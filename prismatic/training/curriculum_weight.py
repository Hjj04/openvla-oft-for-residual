"""
Curriculum Learning 权重计算模块 (从 OpenPI 移植，适配 OpenVLA-OFT)

支持三种权重模式:
1. residual_xyz_norm: XYZ 残差范数 (高值=困难样本)
2. residual_full_norm: 全维度残差范数 (高值=困难样本)
3. executed_scale: 动态 scale 值 (高值=高置信度样本) — 默认

权重计算逻辑:
- residual_*_norm: 高斯权重 w = exp(-(r_norm - r_peak)^2 / (2σ^2))
  - r_peak 随训练进程变化 (curriculum learning)
- executed_scale: 直接映射 w = scale^α
"""

import math
import torch
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple


WEIGHT_MODES = ["residual_xyz_norm", "residual_full_norm", "executed_scale"]


@dataclass
class CurriculumConfig:
    """Curriculum Learning 配置"""
    weight_mode: Literal["residual_xyz_norm", "residual_full_norm", "executed_scale"] = "executed_scale"

    # 高斯权重参数 (用于 residual_*_norm 模式)
    sigma: float = 0.3
    w_max: float = 2.0

    # Curriculum Learning 参数
    enabled: bool = True
    by_step: bool = True
    total_steps: int = 50000
    total_epochs: int = 100
    r_peak_start: float = 0.2
    r_peak_end: float = 0.8
    schedule: Literal["default", "linear", "cosine", "step"] = "default"

    # executed_scale 模式参数
    scale_power: float = 1.0
    scale_min_weight: float = 0.1
    scale_use_curriculum: bool = False

    # Batch 权重归一化
    batch_normalize: bool = True


class CurriculumWeightCalculator:
    """Curriculum Learning 权重计算器 (支持多种权重模式 + wandb 指标输出)"""

    def __init__(self, config: Optional[CurriculumConfig] = None, **kwargs):
        if config is not None:
            self.config = config
        else:
            self.config = CurriculumConfig(**kwargs)
        self.current_r_peak = 0.0
        self.current_sigma = self.config.sigma

    @property
    def weight_mode(self) -> str:
        return self.config.weight_mode

    def get_r_peak(self, current_step: int = 0, current_epoch: int = 0) -> float:
        """计算当前的 r_peak 值 (仅用于 residual_*_norm 模式)"""
        if not self.config.enabled:
            return (self.config.r_peak_start + self.config.r_peak_end) / 2

        if self.config.by_step:
            progress = min(current_step / max(self.config.total_steps, 1), 1.0)
        else:
            progress = min(current_epoch / max(self.config.total_epochs, 1), 1.0)

        r_start = self.config.r_peak_start
        r_end = self.config.r_peak_end

        if self.config.schedule == "default":
            if progress < 0.3:
                r_peak = 0.2
            elif progress < 0.7:
                r_peak = 0.35
            else:
                r_peak = 0.5
        elif self.config.schedule == "linear":
            r_peak = r_start + progress * (r_end - r_start)
        elif self.config.schedule == "cosine":
            r_peak = r_start + 0.5 * (r_end - r_start) * (1 - math.cos(math.pi * progress))
        elif self.config.schedule == "step":
            if progress < 0.33:
                r_peak = r_start
            elif progress < 0.66:
                r_peak = (r_start + r_end) / 2
            else:
                r_peak = r_end
        else:
            if progress < 0.3:
                r_peak = 0.2
            elif progress < 0.7:
                r_peak = 0.35
            else:
                r_peak = 0.5

        return r_peak

    def compute_gaussian_weight(self, r_norm: torch.Tensor, r_peak: float) -> torch.Tensor:
        """计算高斯权重: w = exp(-(r_norm - r_peak)^2 / (2σ^2))"""
        sigma = self.config.sigma
        weights = torch.exp(-((r_norm - r_peak) ** 2) / (2 * sigma ** 2))
        weights = torch.clamp(weights, max=self.config.w_max)
        return weights

    def compute_scale_weight(self, scale_values: torch.Tensor) -> torch.Tensor:
        """计算 scale 权重: w = max(scale^α, min_weight)"""
        alpha = self.config.scale_power
        min_weight = self.config.scale_min_weight
        weights = torch.pow(scale_values + 1e-8, alpha)
        weights = torch.clamp(weights, min=min_weight, max=self.config.w_max)
        return weights

    def compute_weights(
        self,
        weight_values: torch.Tensor,
        current_step: int = 0,
        current_epoch: int = 0,
    ) -> torch.Tensor:
        """计算 batch 内所有样本的权重"""
        if self.config.weight_mode == "executed_scale" and not self.config.scale_use_curriculum:
            weights = self.compute_scale_weight(weight_values)
        else:
            r_peak = self.get_r_peak(current_step, current_epoch)
            self.current_r_peak = r_peak
            weights = self.compute_gaussian_weight(weight_values, r_peak)

        if self.config.batch_normalize and weights.numel() > 0:
            weights = weights / (weights.mean() + 1e-8)

        return weights

    def compute_weights_with_task_balance(
        self,
        weight_values: torch.Tensor,
        task_ids: torch.Tensor,
        current_step: int = 0,
        current_epoch: int = 0,
        num_tasks: int = 1,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算权重 + 任务均衡 + wandb 指标

        Returns:
            (weights, metrics_dict)
        """
        if self.config.weight_mode == "executed_scale" and not self.config.scale_use_curriculum:
            weights = self.compute_scale_weight(weight_values)
        else:
            r_peak = self.get_r_peak(current_step, current_epoch)
            self.current_r_peak = r_peak
            weights = self.compute_gaussian_weight(weight_values, r_peak)

        # 任务内归一化
        if self.config.batch_normalize:
            task_ids_flat = task_ids.squeeze(-1) if task_ids.dim() > 1 else task_ids
            weights_flat = weights.squeeze(-1) if weights.dim() > 1 else weights
            normalized_weights = torch.zeros_like(weights_flat)

            for tid in range(num_tasks):
                mask = (task_ids_flat == tid)
                if mask.sum() > 0:
                    task_weights = weights_flat[mask]
                    task_mean = task_weights.mean() + 1e-8
                    normalized_weights[mask] = task_weights / task_mean

            if weights.dim() > 1:
                weights = normalized_weights.unsqueeze(-1)
            else:
                weights = normalized_weights

        # 构建 wandb 指标
        wv_flat = weight_values.squeeze(-1) if weight_values.dim() > 1 else weight_values
        w_flat = weights.squeeze(-1) if weights.dim() > 1 else weights
        tid_flat = task_ids.squeeze(-1) if task_ids.dim() > 1 else task_ids

        progress = current_step / max(self.config.total_steps, 1)
        metrics = {
            "weight_value_mean": wv_flat.mean().item(),
            "weight_value_std": wv_flat.std().item() if wv_flat.numel() > 1 else 0.0,
            "weight_value_min": wv_flat.min().item(),
            "weight_value_max": wv_flat.max().item(),
            "r_peak": self.current_r_peak,
            "sigma": self.current_sigma,
            "progress": progress,
            "final_weight_mean": w_flat.mean().item(),
            "final_weight_std": w_flat.std().item() if w_flat.numel() > 1 else 0.0,
            "final_weight_min": w_flat.min().item(),
            "final_weight_max": w_flat.max().item(),
            "high_weight_ratio": (w_flat > 0.5).float().mean().item(),
            "weight_ratio_max_min": (w_flat.max() / w_flat.min().clamp(min=1e-8)).item(),
        }

        # Per-task 指标
        for tid in range(num_tasks):
            mask = (tid_flat == tid)
            if mask.any():
                metrics[f"task_{tid}_weight_mean"] = w_flat[mask].mean().item()
                metrics[f"task_{tid}_wv_mean"] = wv_flat[mask].mean().item()
            metrics[f"task_{tid}_count"] = mask.sum().item()

        return weights, metrics


def create_curriculum_calculator(
    weight_mode: str = "executed_scale",
    sigma: float = 0.3,
    w_max: float = 2.0,
    r_peak_start: float = 0.2,
    r_peak_end: float = 0.8,
    total_steps: int = 50000,
    schedule: str = "default",
    batch_normalize: bool = True,
    scale_power: float = 1.0,
) -> CurriculumWeightCalculator:
    """创建 Curriculum 权重计算器的便捷函数"""
    config = CurriculumConfig(
        weight_mode=weight_mode,
        sigma=sigma,
        w_max=w_max,
        enabled=True,
        by_step=True,
        total_steps=total_steps,
        r_peak_start=r_peak_start,
        r_peak_end=r_peak_end,
        schedule=schedule,
        batch_normalize=batch_normalize,
        scale_power=scale_power,
    )
    return CurriculumWeightCalculator(config)
