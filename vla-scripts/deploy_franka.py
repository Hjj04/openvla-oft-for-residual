"""
deploy_franka.py

Franka 推理服务，适配 10D EEF_R6 动作空间和残差 RL client。
基于 deploy.py，针对 Franka 双目相机 + 10D action chunk 优化。

支持两种模式:
  1. Fine-tuned 模式 (默认): 加载微调后的 checkpoint，L1 回归 + proprio + 双目
  2. Zero-shot 模式 (--zero_shot): 加载预训练 openvla-7b，离散 token 预测，
     7D 动作自动转换为 10D rot6d 并填充 10 步 chunk

用法:
    # Fine-tuned 模式
    python vla-scripts/deploy_franka.py \
        --pretrained_checkpoint /path/to/checkpoint \
        --unnorm_key franka \
        --port 8777

    # Zero-shot 模式 (预训练 openvla-7b)
    python vla-scripts/deploy_franka.py \
        --pretrained_checkpoint /path/to/openvla-7b \
        --zero_shot True \
        --unnorm_key bridge_orig \
        --port 8777
"""

import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import draccus
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.projectors import ProprioProjector
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM

try:
    import json_numpy
    json_numpy.patch()
except ImportError:
    pass

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def normalize_proprio(proprio: np.ndarray, norm_stats: Dict) -> np.ndarray:
    """Normalize proprio using q01/q99 bounds to [-1, 1]."""
    q01 = np.array(norm_stats["q01"], dtype=np.float32)
    q99 = np.array(norm_stats["q99"], dtype=np.float32)
    mask = (q99 - q01) > 1e-6
    normalized = np.zeros_like(proprio, dtype=np.float32)
    normalized[mask] = 2.0 * (proprio[mask] - q01[mask]) / (q99[mask] - q01[mask]) - 1.0
    return np.clip(normalized, -1.0, 1.0)


def unnormalize_actions(actions: np.ndarray, norm_stats: Dict) -> np.ndarray:
    """Unnormalize actions from [-1, 1] back to original scale using q01/q99."""
    q01 = np.array(norm_stats["q01"], dtype=np.float32)
    q99 = np.array(norm_stats["q99"], dtype=np.float32)
    return q01 + (actions + 1.0) / 2.0 * (q99 - q01)


# ─── Zero-shot helpers: 7D (xyz+euler+gripper) → 10D (xyz+rot6d+gripper) ───

def _euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Euler angles (XYZ convention) → 3x3 rotation matrix."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ], dtype=np.float32)


def euler_to_rot6d(euler: np.ndarray) -> np.ndarray:
    """Euler angles (3,) → rot6d (6,)."""
    R = _euler_to_rotation_matrix(euler[0], euler[1], euler[2])
    return R[:, :2].T.flatten()


def action_7d_to_10d(action_7d: np.ndarray) -> np.ndarray:
    """Map 7D action (xyz+euler+gripper) → 10D (xyz+rot6d+gripper)."""
    xyz = action_7d[:3]
    rot6d = euler_to_rot6d(action_7d[3:6])
    gripper = action_7d[6:7]
    return np.concatenate([xyz, rot6d, gripper]).astype(np.float32)


PIPELINE_ACTION_DIM = 10
PIPELINE_CHUNK_SIZE = 10


class FrankaVLAServer:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.zero_shot = getattr(cfg, "zero_shot", False)

        # Zero-shot mode: monkey-patch constants for openvla-7b (7D, chunk=1)
        if self.zero_shot:
            import prismatic.extern.hf.modeling_prismatic as _mp
            _mp.ACTION_DIM = 7
            _mp.NUM_ACTIONS_CHUNK = 1
            print("Zero-shot mode: patched ACTION_DIM=7, NUM_ACTIONS_CHUNK=1")

        # Register model
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        # Load VLA
        print(f"Loading VLA from {cfg.pretrained_checkpoint}...")
        self.processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            cfg.pretrained_checkpoint,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(DEVICE)
        self.vla.eval()

        if self.zero_shot:
            self.vla.vision_backbone.set_num_images_in_input(1)
            self.action_head = None
            self.proprio_projector = None
        else:
            self.vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

            # Load action head
            self.action_head = None
            if cfg.use_l1_regression:
                self.action_head = L1RegressionActionHead(
                    input_dim=self.vla.llm_dim, hidden_dim=self.vla.llm_dim, action_dim=ACTION_DIM
                )
                head_path = Path(cfg.pretrained_checkpoint) / "action_head--latest_checkpoint.pt"
                if not head_path.exists():
                    import glob
                    candidates = glob.glob(str(Path(cfg.pretrained_checkpoint) / "action_head--*_checkpoint.pt"))
                    if candidates:
                        head_path = Path(sorted(candidates)[-1])
                if head_path.exists():
                    state_dict = torch.load(head_path, weights_only=True, map_location="cpu")
                    clean_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
                    self.action_head.load_state_dict(clean_sd)
                    print(f"Loaded action head from {head_path}")
                self.action_head = self.action_head.to(torch.bfloat16).to(DEVICE)
                self.action_head.eval()

            # Load proprio projector
            self.proprio_projector = None
            if cfg.use_proprio:
                self.proprio_projector = ProprioProjector(llm_dim=self.vla.llm_dim, proprio_dim=PROPRIO_DIM)
                proj_path = Path(cfg.pretrained_checkpoint) / "proprio_projector--latest_checkpoint.pt"
                if not proj_path.exists():
                    import glob
                    candidates = glob.glob(str(Path(cfg.pretrained_checkpoint) / "proprio_projector--*_checkpoint.pt"))
                    if candidates:
                        proj_path = Path(sorted(candidates)[-1])
                if proj_path.exists():
                    state_dict = torch.load(proj_path, weights_only=True, map_location="cpu")
                    clean_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
                    self.proprio_projector.load_state_dict(clean_sd)
                    print(f"Loaded proprio projector from {proj_path}")
                self.proprio_projector = self.proprio_projector.to(DEVICE)
                self.proprio_projector.eval()

        # Load normalization stats
        self.norm_stats = self._load_norm_stats()
        print(f"Loaded norm stats for key: {cfg.unnorm_key}")

        # Inject norm_stats into model so predict_action can unnormalize correctly
        if not hasattr(self.vla, "norm_stats") or self.vla.norm_stats is None:
            self.vla.norm_stats = {}
        self.vla.norm_stats[cfg.unnorm_key] = self.norm_stats

        # Image resize size
        self.resize_size = tuple(self.vla.config.image_sizes)
        mode_str = "zero-shot (7D→10D)" if self.zero_shot else "fine-tuned"
        print(f"[{mode_str}] Image resize: {self.resize_size}, Output: {PIPELINE_ACTION_DIM}D×{PIPELINE_CHUNK_SIZE}")

    def _load_norm_stats(self) -> Dict:
        """Load normalization statistics from checkpoint or use identity for zero-shot."""
        stats_path = Path(self.cfg.pretrained_checkpoint) / "dataset_statistics.json"
        if stats_path.exists():
            with open(stats_path) as f:
                all_stats = json.load(f)
            if self.cfg.unnorm_key in all_stats:
                stats = all_stats[self.cfg.unnorm_key]
                for k in stats:
                    for k2 in stats[k]:
                        if isinstance(stats[k][k2], list):
                            stats[k][k2] = np.array(stats[k][k2], dtype=np.float32)
                return stats

        # Fallback: try vla.norm_stats
        if hasattr(self.vla, "norm_stats") and self.cfg.unnorm_key in self.vla.norm_stats:
            return self.vla.norm_stats[self.cfg.unnorm_key]

        if self.zero_shot:
            # Identity norm_stats: predict_action's _unnormalize_actions becomes a no-op
            print(f"Zero-shot mode: using identity norm_stats for unnorm_key='{self.cfg.unnorm_key}'")
            return {
                "action": {
                    "q01": np.full(7, -1.0, dtype=np.float32),
                    "q99": np.full(7, 1.0, dtype=np.float32),
                    "mean": np.zeros(7, dtype=np.float32),
                    "std": np.ones(7, dtype=np.float32),
                },
            }

        raise ValueError(f"Cannot find norm stats for key '{self.cfg.unnorm_key}'")

    @torch.inference_mode()
    def predict_action(self, observation: Dict[str, Any], instruction: str) -> List[np.ndarray]:
        """Predict action chunk from observation.

        Fine-tuned: returns list of 10D actions (chunk_size items).
        Zero-shot: returns list of 10D actions (10 identical steps from single 7D→10D conversion).
        """
        # Process primary image
        primary_img = Image.fromarray(observation["image"])
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        inputs = self.processor(prompt, primary_img).to(DEVICE, dtype=torch.bfloat16)

        if self.zero_shot:
            # Zero-shot: single image, discrete tokens, no proprio
            action, _ = self.vla.predict_action(
                **inputs,
                unnorm_key=self.cfg.unnorm_key,
                do_sample=False,
                action_head=None,
            )
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            action = action.reshape(-1)  # (7,)

            # Convert 7D → 10D and tile to chunk
            action_10d = action_7d_to_10d(action)
            return [action_10d.copy() for _ in range(PIPELINE_CHUNK_SIZE)]

        # ── Fine-tuned path ──
        # Process wrist image if available
        if self.cfg.num_images_in_input > 1 and "image2" in observation:
            wrist_img = Image.fromarray(observation["image2"])
            wrist_inputs = self.processor(prompt, wrist_img).to(DEVICE, dtype=torch.bfloat16)
            inputs["pixel_values"] = torch.cat(
                [inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1
            )

        # Normalize proprio
        proprio = None
        if self.cfg.use_proprio and "state" in observation:
            proprio_raw = np.array(observation["state"], dtype=np.float32)
            proprio = normalize_proprio(proprio_raw, self.norm_stats["proprio"])
            proprio = torch.tensor(proprio, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # Forward pass — predict_action internally unnormalizes via self.vla.norm_stats
        action, _ = self.vla.predict_action(
            **inputs,
            unnorm_key=self.cfg.unnorm_key,
            do_sample=False,
            proprio=proprio,
            proprio_projector=self.proprio_projector,
            action_head=self.action_head,
        )

        # action is already unnormalized by predict_action, just convert to numpy
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if isinstance(action, np.ndarray):
            if action.ndim == 1:
                action = action.reshape(1, -1)
            return [action[i].astype(np.float32) for i in range(action.shape[0])]

        return [np.array(action, dtype=np.float32)]

    def get_server_action(self, payload: Dict[str, Any]) -> Any:
        try:
            if "encoded" in payload:
                payload = json.loads(payload["encoded"])

            instruction = payload.get("instruction", payload.get("task_description", ""))
            actions = self.predict_action(payload, instruction)

            # Return as list of lists for JSON serialization
            result = {"actions": [a.tolist() for a in actions]}
            return JSONResponse(result)
        except Exception:
            logging.error(traceback.format_exc())
            return JSONResponse({"error": "prediction failed"}, status_code=500)

    def run(self, host: str = "0.0.0.0", port: int = 8777) -> None:
        app = FastAPI()
        app.post("/act")(self.get_server_action)
        print(f"Starting Franka VLA server on {host}:{port}")
        uvicorn.run(app, host=host, port=port)


@dataclass
class DeployConfig:
    # fmt: off
    host: str = "0.0.0.0"
    port: int = 8777
    pretrained_checkpoint: str = ""
    use_l1_regression: bool = True
    use_film: bool = False
    num_images_in_input: int = 2                # primary + wrist
    use_proprio: bool = True
    lora_rank: int = 32
    unnorm_key: str = "franka"

    # Zero-shot mode: use pretrained openvla-7b (discrete tokens, 7D, no chunking)
    zero_shot: bool = False
    # fmt: on


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = FrankaVLAServer(cfg)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
