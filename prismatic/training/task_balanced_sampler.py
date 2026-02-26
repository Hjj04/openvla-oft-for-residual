"""
任务均衡采样器 (从 OpenPI 移植，适配 OpenVLA-OFT)

功能：
1. 按任务均衡采样 (每个任务被采样的概率相等)
2. 支持 DDP 分布式训练
3. 支持从缓存加载任务索引 (避免每次遍历数据集)
"""

import json
import logging
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import Sampler


class TaskBalancedSampler(Sampler[int]):
    """
    任务均衡采样器

    确保每个 batch 中各任务的样本数量大致相等
    采样方式: 轮流从各任务采样 (round-robin)
    """

    def __init__(
        self,
        dataset,
        task_id_key: str = "task_id",
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        cache_dir: Optional[str] = None,
    ):
        if num_replicas is None:
            if torch.distributed.is_initialized():
                num_replicas = torch.distributed.get_world_size()
            else:
                num_replicas = 1

        if rank is None:
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = 0

        self.dataset = dataset
        self.task_id_key = task_id_key
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.cache_dir = cache_dir

        self._build_task_indices()

    def _build_task_indices(self):
        """构建每个任务的样本索引列表"""
        self.task_indices = {}

        # 尝试从缓存加载
        cache_file = None
        if self.cache_dir:
            cache_file = Path(self.cache_dir) / "task_indices_cache.json"
            if cache_file.exists():
                try:
                    with open(cache_file, "r") as f:
                        cached = json.load(f)
                    if cached.get("dataset_len") == len(self.dataset):
                        self.task_indices = {int(k): v for k, v in cached["task_indices"].items()}
                        logging.info(f"Loaded task indices from cache: {cache_file}")
                        self._finalize_indices()
                        return
                except Exception as e:
                    logging.warning(f"Failed to load cache: {e}")

        logging.info(f"Building task indices for {len(self.dataset)} samples...")

        for idx in range(len(self.dataset)):
            try:
                sample = self.dataset[idx]
                if self.task_id_key in sample:
                    task_id = sample[self.task_id_key]
                    if isinstance(task_id, (np.ndarray, torch.Tensor)):
                        task_id = int(task_id.item() if hasattr(task_id, "item") else task_id[0])
                    else:
                        task_id = int(task_id)
                else:
                    task_id = 0

                if task_id not in self.task_indices:
                    self.task_indices[task_id] = []
                self.task_indices[task_id].append(idx)
            except Exception as e:
                logging.warning(f"Failed to get task_id for sample {idx}: {e}")
                if 0 not in self.task_indices:
                    self.task_indices[0] = []
                self.task_indices[0].append(idx)

        # 保存缓存
        if cache_file:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file, "w") as f:
                    json.dump({"dataset_len": len(self.dataset), "task_indices": self.task_indices}, f)
                logging.info(f"Saved task indices to cache: {cache_file}")
            except Exception as e:
                logging.warning(f"Failed to save cache: {e}")

        self._finalize_indices()

    def _finalize_indices(self):
        """完成索引构建后的处理"""
        self.num_tasks = len(self.task_indices)
        self.task_ids = sorted(self.task_indices.keys())

        for task_id in self.task_ids:
            logging.info(f"  Task {task_id}: {len(self.task_indices[task_id])} samples")

        total_samples = len(self.dataset)
        self.num_samples = total_samples // self.num_replicas
        if not self.drop_last and total_samples % self.num_replicas != 0:
            self.num_samples += 1
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        """生成均衡采样的索引序列"""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        task_iters = {}
        for task_id in self.task_ids:
            indices = self.task_indices[task_id].copy()
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=g).tolist()
                indices = [indices[i] for i in perm]
            task_iters[task_id] = iter(indices)

        indices = []
        current_task_idx = 0

        while len(indices) < self.total_size:
            task_id = self.task_ids[current_task_idx % self.num_tasks]
            try:
                idx = next(task_iters[task_id])
                indices.append(idx)
            except StopIteration:
                task_indices_copy = self.task_indices[task_id].copy()
                if self.shuffle:
                    perm = torch.randperm(len(task_indices_copy), generator=g).tolist()
                    task_indices_copy = [task_indices_copy[i] for i in perm]
                task_iters[task_id] = iter(task_indices_copy)
                idx = next(task_iters[task_id])
                indices.append(idx)
            current_task_idx += 1

        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int):
        """设置 epoch (用于 DDP 同步)"""
        self.epoch = epoch
