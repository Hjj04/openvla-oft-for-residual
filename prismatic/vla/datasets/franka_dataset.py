"""
franka_dataset.py

自定义 PyTorch Map-style Dataset，直接读取 PKL 文件。
输出格式与 RLDSBatchTransform 完全一致，确保 run_forward_pass 无需修改。

PKL 文件格式 (每个 PKL 是一个 episode = list of transitions):
    transition = {
        "observations": {
            "pixels": {"image": np.ndarray(H,W,3), "image2": np.ndarray(H,W,3)},
            "agent_pos": np.ndarray(10,),  # [xyz(3) + rot6d(6) + gripper(1)]
            "state": np.ndarray(10,),
            "task_description": str,
        },
        "action": np.ndarray(10,),  # [dx,dy,dz, rot6d(6), dg]
        "reward": float,
        "dones": bool,
    }

加权模式 (Weighted SFT) 额外字段:
    transition["weight_value"]: float  # 归一化后的权重值
    transition["task_id"]: int         # 任务 ID
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)


class FrankaDataset(Dataset):
    """
    Map-style Dataset 读取 PKL 文件。
    输出格式与 RLDSBatchTransform 完全一致:
    {pixel_values, input_ids, labels, actions, proprio, dataset_name}
    加权模式额外输出: {weight_value, task_id}
    """

    def __init__(
        self,
        data_dirs: List[str],
        task_names: List[str],
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
        action_dim: int = 10,
        chunk_size: int = 10,
        proprio_dim: int = 10,
        use_wrist_image: bool = True,
        use_proprio: bool = True,
        weighted: bool = False,
        norm_stats: Optional[Dict] = None,
    ):
        """
        Args:
            data_dirs: PKL 数据目录列表 (每个目录对应一个任务)
            task_names: 任务名称列表 (与 data_dirs 一一对应)
            action_tokenizer: 动作离散化 tokenizer
            base_tokenizer: LLM tokenizer
            image_transform: 图像预处理 transform
            prompt_builder_fn: Prompt 构建器
            action_dim: 动作维度 (默认 10)
            chunk_size: Action chunk 大小 (默认 10)
            proprio_dim: 本体感知维度 (默认 10)
            use_wrist_image: 是否使用腕部相机
            use_proprio: 是否使用本体感知
            weighted: 是否为加权 SFT 模式
            norm_stats: 预计算的归一化统计量 {action: {q01, q99}, proprio: {q01, q99}}
        """
        super().__init__()
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.proprio_dim = proprio_dim
        self.use_wrist_image = use_wrist_image
        self.use_proprio = use_proprio
        self.weighted = weighted

        # 加载所有 episode 并构建 flat index
        self.episodes = []       # [(episode_data, task_id, task_name)]
        self.flat_index = []     # [(episode_idx, frame_idx)]

        logging.info(f"Loading PKL data from {len(data_dirs)} directories...")
        for task_id, (data_dir, task_name) in enumerate(zip(data_dirs, task_names)):
            pkl_files = sorted(Path(data_dir).glob("*.pkl"))
            logging.info(f"  Task {task_id} ({task_name}): {len(pkl_files)} episodes from {data_dir}")

            for pkl_file in pkl_files:
                with open(pkl_file, "rb") as f:
                    episode = pickle.load(f)

                if not isinstance(episode, list):
                    logging.warning(f"Skipping {pkl_file}: not a list")
                    continue

                ep_idx = len(self.episodes)
                self.episodes.append((episode, task_id, task_name))

                # 每个 frame 都可以作为起始点 (末尾 pad)
                for frame_idx in range(len(episode)):
                    self.flat_index.append((ep_idx, frame_idx))

        logging.info(f"Total: {len(self.episodes)} episodes, {len(self.flat_index)} frames")

        # 计算或加载归一化统计量
        if norm_stats is not None:
            self.norm_stats = norm_stats
        else:
            self.norm_stats = self._compute_norm_stats()

        # 构建 dataset_statistics (用于推理时反归一化)
        self.dataset_statistics = {
            "franka": {
                "action": {
                    "q01": self.norm_stats["action"]["q01"],
                    "q99": self.norm_stats["action"]["q99"],
                    "mean": self.norm_stats["action"].get("mean", np.zeros(action_dim)),
                    "std": self.norm_stats["action"].get("std", np.ones(action_dim)),
                },
                "proprio": {
                    "q01": self.norm_stats["proprio"]["q01"],
                    "q99": self.norm_stats["proprio"]["q99"],
                },
            }
        }

    def _compute_norm_stats(self) -> Dict:
        """计算 action 和 proprio 的 q01/q99 统计量"""
        all_actions = []
        all_proprios = []

        for episode, task_id, task_name in self.episodes:
            for transition in episode:
                action = self._extract_action(transition)
                if action is not None:
                    all_actions.append(action)
                proprio = self._extract_proprio(transition)
                if proprio is not None:
                    all_proprios.append(proprio)

        all_actions = np.array(all_actions)
        all_proprios = np.array(all_proprios) if all_proprios else np.zeros((1, self.proprio_dim))

        stats = {
            "action": {
                "q01": np.percentile(all_actions, 1, axis=0).astype(np.float32),
                "q99": np.percentile(all_actions, 99, axis=0).astype(np.float32),
                "mean": np.mean(all_actions, axis=0).astype(np.float32),
                "std": np.std(all_actions, axis=0).astype(np.float32),
            },
            "proprio": {
                "q01": np.percentile(all_proprios, 1, axis=0).astype(np.float32),
                "q99": np.percentile(all_proprios, 99, axis=0).astype(np.float32),
            },
        }
        logging.info(f"Computed norm stats: action q01={stats['action']['q01']}, q99={stats['action']['q99']}")
        return stats

    def _extract_action(self, transition: Dict) -> Optional[np.ndarray]:
        """从 transition 中提取 action"""
        if "action" in transition:
            return np.array(transition["action"], dtype=np.float32)
        return None

    def _extract_proprio(self, transition: Dict) -> Optional[np.ndarray]:
        """从 transition 中提取 proprio (agent_pos 或 state)"""
        obs = transition.get("observations", transition.get("observation", {}))
        if "agent_pos" in obs:
            return np.array(obs["agent_pos"], dtype=np.float32)[:self.proprio_dim]
        elif "state" in obs:
            return np.array(obs["state"], dtype=np.float32)[:self.proprio_dim]
        return None

    def _extract_image(self, transition: Dict, key: str = "image") -> Optional[np.ndarray]:
        """从 transition 中提取图像"""
        obs = transition.get("observations", transition.get("observation", {}))
        pixels = obs.get("pixels", obs.get("images", {}))
        if key in pixels:
            return np.array(pixels[key], dtype=np.uint8)
        return None

    def _extract_task_description(self, transition: Dict, fallback: str = "") -> str:
        """从 transition 中提取任务描述"""
        obs = transition.get("observations", transition.get("observation", {}))
        desc = obs.get("task_description", transition.get("language_instruction", fallback))
        if isinstance(desc, bytes):
            desc = desc.decode()
        return desc.lower().strip()

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """用 q01/q99 归一化 action 到 [-1, 1]"""
        q01 = self.norm_stats["action"]["q01"]
        q99 = self.norm_stats["action"]["q99"]
        mask = (q99 - q01) > 1e-6
        normalized = np.zeros_like(action)
        normalized[mask] = 2.0 * (action[mask] - q01[mask]) / (q99[mask] - q01[mask]) - 1.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def _normalize_proprio(self, proprio: np.ndarray) -> np.ndarray:
        """用 q01/q99 归一化 proprio 到 [-1, 1]"""
        q01 = self.norm_stats["proprio"]["q01"]
        q99 = self.norm_stats["proprio"]["q99"]
        mask = (q99 - q01) > 1e-6
        normalized = np.zeros_like(proprio)
        normalized[mask] = 2.0 * (proprio[mask] - q01[mask]) / (q99[mask] - q01[mask]) - 1.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def __len__(self) -> int:
        return len(self.flat_index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep_idx, frame_idx = self.flat_index[idx]
        episode, task_id, task_name = self.episodes[ep_idx]
        transition = episode[frame_idx]
        ep_len = len(episode)

        # 1. 提取图像
        img_array = self._extract_image(transition, "image")
        if img_array is None:
            raise ValueError(f"No image found in episode {ep_idx}, frame {frame_idx}")
        img = Image.fromarray(img_array)
        pixel_values = self.image_transform(img)

        # 腕部相机 (可选)
        pixel_values_wrist = None
        if self.use_wrist_image:
            wrist_array = self._extract_image(transition, "image2")
            if wrist_array is not None:
                wrist_img = Image.fromarray(wrist_array)
                pixel_values_wrist = self.image_transform(wrist_img)

        # 2. Action chunking: 取 [frame_idx, frame_idx+1, ..., frame_idx+chunk_size-1]
        actions_chunk = []
        for i in range(self.chunk_size):
            t = min(frame_idx + i, ep_len - 1)  # 末尾 pad (重复最后一帧)
            action = self._extract_action(episode[t])
            if action is None:
                action = np.zeros(self.action_dim, dtype=np.float32)
            actions_chunk.append(action)
        actions_raw = np.stack(actions_chunk, axis=0)  # (chunk_size, action_dim)

        # 3. 归一化 actions
        actions_normalized = np.stack([self._normalize_action(a) for a in actions_raw], axis=0)

        # 4. 归一化 proprio
        proprio = None
        if self.use_proprio:
            proprio_raw = self._extract_proprio(transition)
            if proprio_raw is not None:
                proprio = self._normalize_proprio(proprio_raw)
            else:
                proprio = np.zeros(self.proprio_dim, dtype=np.float32)

        # 5. 构建 prompt + tokenize (与 RLDSBatchTransform 完全一致)
        lang = self._extract_task_description(transition, fallback=task_name)
        prompt_builder = self.prompt_builder_fn("openvla")

        # Action tokenization: 当前帧 + 未来帧
        current_action_string = self.action_tokenizer(actions_normalized[0])
        future_actions_string = "".join(self.action_tokenizer(actions_normalized[1:]))
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

        # 6. Labels masking: 只对 action token 计算 loss
        labels[:-(action_chunk_len + 1)] = IGNORE_INDEX

        # 7. 构建返回 dict
        return_dict = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            dataset_name="franka",
            actions=actions_normalized,  # (chunk_size, action_dim), float32
        )

        if pixel_values_wrist is not None:
            return_dict["pixel_values_wrist"] = pixel_values_wrist

        if self.use_proprio and proprio is not None:
            return_dict["proprio"] = proprio

        # 8. 加权 SFT 模式: 额外输出 weight_value 和 task_id
        if self.weighted:
            wv = transition.get("weight_value", 1.0)
            return_dict["weight_value"] = float(wv)
            return_dict["task_id"] = int(task_id)

        return return_dict
