"""
websocket_openvla_server.py

OpenVLA-OFT WebSocket 推理服务，协议兼容 OpenPI 的 websocket_pi_server.py。
RLVLA 的 OpenPiOfficialWeb 客户端可以直接连接本服务。

支持两种模式:
  1. Fine-tuned 模式 (默认): 加载微调后的 checkpoint，L1 回归 + proprio + 双目
  2. Zero-shot 模式 (--zero_shot): 加载预训练 openvla-7b，离散 token 预测，
     7D 动作自动转换为 10D rot6d 并填充 10 步 chunk

协议:
  - WebSocket on port 55555, no compression, no size limit
  - 序列化: MessagePack + NumPy (msgpack_numpy)
  - 握手: 服务端先发 metadata dict
  - 请求: {observation/image, observation/wrist_image, observation/state, prompt}
  - 响应: {actions: np.ndarray(chunk_size, action_dim)}

用法:
    # Fine-tuned 模式
    CUDA_VISIBLE_DEVICES=0 python vla-scripts/websocket_openvla_server.py \
        --pretrained_checkpoint /path/to/finetuned_checkpoint \
        --unnorm_key franka

    # Zero-shot 模式 (预训练 openvla-7b)
    CUDA_VISIBLE_DEVICES=0 python vla-scripts/websocket_openvla_server.py \
        --pretrained_checkpoint /path/to/openvla-7b \
        --zero_shot True \
        --unnorm_key bridge_orig
"""

import asyncio
import json
import logging
import socket
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import draccus
import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

import websockets.asyncio.server
import websockets.frames

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.projectors import ProprioProjector
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM
from prismatic.util import msgpack_numpy

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
    """Euler angles (3,) → rot6d (6,). Same convention as pkl_to_franka_dataset.py."""
    R = _euler_to_rotation_matrix(euler[0], euler[1], euler[2])
    return R[:, :2].T.flatten()  # first 2 columns, transposed, flattened → (6,)


def action_7d_to_10d(action_7d: np.ndarray) -> np.ndarray:
    """Map 7D action (xyz+euler+gripper) → 10D (xyz+rot6d+gripper).

    openvla-7b outputs: [dx, dy, dz, droll, dpitch, dyaw, gripper]
    Pipeline expects:   [dx, dy, dz, r1, r2, r3, r4, r5, r6, gripper]
    """
    xyz = action_7d[:3]
    rot6d = euler_to_rot6d(action_7d[3:6])
    gripper = action_7d[6:7]
    return np.concatenate([xyz, rot6d, gripper]).astype(np.float32)


# Pipeline 输出常量 (不管 zero-shot 还是 fine-tuned，对外都是 10D × 10 步)
PIPELINE_ACTION_DIM = 10
PIPELINE_CHUNK_SIZE = 10


class OpenVLAPolicy:
    """Wraps OpenVLA-OFT model with an `infer()` interface matching OpenPI's policy protocol.

    Input:  {observation/image, observation/wrist_image, observation/state, prompt}
    Output: {actions: np.ndarray(chunk_size, action_dim)}
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.zero_shot = getattr(cfg, "zero_shot", False)

        # ── Zero-shot mode: monkey-patch constants for openvla-7b (7D, chunk=1) ──
        if self.zero_shot:
            import prismatic.extern.hf.modeling_prismatic as _mp
            _mp.ACTION_DIM = 7
            _mp.NUM_ACTIONS_CHUNK = 1
            print("Zero-shot mode: patched ACTION_DIM=7, NUM_ACTIONS_CHUNK=1")

        # Register HF model classes
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        # Load VLA backbone
        print(f"Loading OpenVLA from {cfg.pretrained_checkpoint}...")
        self.processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            cfg.pretrained_checkpoint,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(DEVICE)
        self.vla.eval()

        # Zero-shot: single image, no action_head, no proprio
        if self.zero_shot:
            self.vla.vision_backbone.set_num_images_in_input(1)
            self.action_head = None
            self.proprio_projector = None
        else:
            self.vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

            # Load action head (L1 regression)
            self.action_head = None
            if cfg.use_l1_regression:
                self.action_head = L1RegressionActionHead(
                    input_dim=self.vla.llm_dim, hidden_dim=self.vla.llm_dim, action_dim=ACTION_DIM
                )
                head_path = self._find_checkpoint(cfg.pretrained_checkpoint, "action_head")
                if head_path and head_path.exists():
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
                proj_path = self._find_checkpoint(cfg.pretrained_checkpoint, "proprio_projector")
                if proj_path and proj_path.exists():
                    state_dict = torch.load(proj_path, weights_only=True, map_location="cpu")
                    clean_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
                    self.proprio_projector.load_state_dict(clean_sd)
                    print(f"Loaded proprio projector from {proj_path}")
                self.proprio_projector = self.proprio_projector.to(DEVICE)
                self.proprio_projector.eval()

        # Load normalization stats and inject into model
        self.norm_stats = self._load_norm_stats(cfg)

        if not hasattr(self.vla, "norm_stats") or self.vla.norm_stats is None:
            self.vla.norm_stats = {}
        self.vla.norm_stats[cfg.unnorm_key] = self.norm_stats

        mode_str = "zero-shot (7D→10D)" if self.zero_shot else "fine-tuned"
        print(f"OpenVLA ready [{mode_str}]: output={PIPELINE_ACTION_DIM}D×{PIPELINE_CHUNK_SIZE}, device={DEVICE}")

    @staticmethod
    def _find_checkpoint(ckpt_dir: str, prefix: str) -> Optional[Path]:
        """Find latest checkpoint file for a given prefix (action_head / proprio_projector)."""
        latest = Path(ckpt_dir) / f"{prefix}--latest_checkpoint.pt"
        if latest.exists():
            return latest
        import glob
        candidates = glob.glob(str(Path(ckpt_dir) / f"{prefix}--*_checkpoint.pt"))
        return Path(sorted(candidates)[-1]) if candidates else None

    @staticmethod
    def _load_norm_stats(cfg) -> Dict:
        """Load normalization statistics.

        Fine-tuned mode: from dataset_statistics.json in checkpoint dir.
        Zero-shot mode: from model's built-in norm_stats (e.g. bridge_orig).
        """
        zero_shot = getattr(cfg, "zero_shot", False)

        # Try dataset_statistics.json first (fine-tuned checkpoints)
        stats_path = Path(cfg.pretrained_checkpoint) / "dataset_statistics.json"
        if stats_path.exists():
            with open(stats_path) as f:
                all_stats = json.load(f)
            if cfg.unnorm_key in all_stats:
                stats = all_stats[cfg.unnorm_key]
                for k in stats:
                    for k2 in stats[k]:
                        if isinstance(stats[k][k2], list):
                            stats[k][k2] = np.array(stats[k][k2], dtype=np.float32)
                return stats

        if zero_shot:
            # For zero-shot, we construct a passthrough norm_stats so predict_action's
            # _unnormalize_actions works correctly. The pretrained model already outputs
            # actions in a normalized [-1, 1] range via bin_centers, and _unnormalize_actions
            # does: q01 + (action + 1) / 2 * (q99 - q01). With q01=-1, q99=1 this is identity.
            print(f"Zero-shot mode: using identity norm_stats for unnorm_key='{cfg.unnorm_key}'")
            identity_stats = {
                "action": {
                    "q01": np.full(7, -1.0, dtype=np.float32),
                    "q99": np.full(7, 1.0, dtype=np.float32),
                    "mean": np.zeros(7, dtype=np.float32),
                    "std": np.ones(7, dtype=np.float32),
                },
            }
            return identity_stats

        raise ValueError(f"Cannot find norm stats for key '{cfg.unnorm_key}' in {stats_path}")

    @property
    def metadata(self) -> Dict:
        """Server metadata sent on connection (matches OpenPI protocol).

        Always reports pipeline dimensions (10D × 10 steps) regardless of mode,
        since zero-shot 7D→10D conversion is handled internally.
        """
        return {
            "model": "openvla-7b-zeroshot" if self.zero_shot else "openvla-oft",
            "action_dim": PIPELINE_ACTION_DIM,
            "chunk_size": PIPELINE_CHUNK_SIZE,
            "unnorm_key": self.cfg.unnorm_key,
            "zero_shot": self.zero_shot,
        }

    @torch.inference_mode()
    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference on a single observation. Matches OpenPI's policy.infer() interface.

        Fine-tuned mode: L1 regression → (chunk_size, 10) directly.
        Zero-shot mode:  discrete tokens → (1, 7) → convert to (1, 10) → tile to (10, 10).

        Args:
            obs: {
                "observation/image": np.ndarray (H,W,3) uint8
                "observation/wrist_image": np.ndarray (H,W,3) uint8  (ignored in zero-shot)
                "observation/state": np.ndarray (10,) float32         (ignored in zero-shot)
                "prompt": str
            }

        Returns:
            {"actions": np.ndarray (PIPELINE_CHUNK_SIZE, PIPELINE_ACTION_DIM) float32}
        """
        t0 = time.time()

        # Extract fields
        primary_image = obs["observation/image"]  # (H, W, 3) uint8
        prompt_text = obs.get("prompt", "")
        if isinstance(prompt_text, np.ndarray):
            prompt_text = str(prompt_text.item()) if prompt_text.ndim == 0 else str(prompt_text[0])

        # Process primary image
        primary_pil = Image.fromarray(primary_image)
        prompt = f"In: What action should the robot take to {prompt_text.lower()}?\nOut:"
        inputs = self.processor(prompt, primary_pil).to(DEVICE, dtype=torch.bfloat16)

        if self.zero_shot:
            # ── Zero-shot path: discrete tokens, single image, no proprio ──
            action, _ = self.vla.predict_action(
                **inputs,
                unnorm_key=self.cfg.unnorm_key,
                do_sample=False,
                action_head=None,
            )
            # action shape: (1, 7) — already "unnormalized" (identity norm_stats → raw bin_centers)
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            action = action.reshape(-1)  # (7,)

            # Convert 7D (xyz+euler+gripper) → 10D (xyz+rot6d+gripper)
            action_10d = action_7d_to_10d(action)  # (10,)

            # Tile single-step action to fill the 10-step chunk
            actions = np.tile(action_10d, (PIPELINE_CHUNK_SIZE, 1))  # (10, 10)

        else:
            # ── Fine-tuned path: L1 regression with proprio + dual cameras ──
            # Process wrist image if available
            if self.cfg.num_images_in_input > 1 and "observation/wrist_image" in obs:
                wrist_pil = Image.fromarray(obs["observation/wrist_image"])
                wrist_inputs = self.processor(prompt, wrist_pil).to(DEVICE, dtype=torch.bfloat16)
                inputs["pixel_values"] = torch.cat(
                    [inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1
                )

            # Normalize proprio
            proprio = None
            if self.cfg.use_proprio and "observation/state" in obs:
                proprio_raw = np.array(obs["observation/state"], dtype=np.float32)
                proprio_norm = normalize_proprio(proprio_raw, self.norm_stats["proprio"])
                proprio = torch.tensor(proprio_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # Forward pass — predict_action internally unnormalizes via self.vla.norm_stats
            action, _ = self.vla.predict_action(
                **inputs,
                unnorm_key=self.cfg.unnorm_key,
                do_sample=False,
                proprio=proprio,
                proprio_projector=self.proprio_projector,
                action_head=self.action_head,
            )

            # Post-process: to numpy (already unnormalized by predict_action)
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            if action.ndim == 1:
                action = action.reshape(1, -1)
            actions = action.astype(np.float32)

        infer_ms = (time.time() - t0) * 1000
        return {
            "actions": actions.astype(np.float32),
            "server_timing": {"infer_ms": infer_ms},
        }


class WebsocketPolicyServer:
    """WebSocket server matching OpenPI's protocol exactly.

    Handshake: send metadata on connect.
    Loop: recv msgpack obs → policy.infer(obs) → send msgpack result.
    """

    def __init__(self, policy: OpenVLAPolicy, host: str = "0.0.0.0", port: int = 55555) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            print(f"OpenVLA WebSocket server listening on ws://{local_ip}:{self._port}")
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        # Handshake: send metadata first (same as OpenPI)
        await websocket.send(packer.pack(self._policy.metadata))

        prev_total_ms = 0.0
        while True:
            try:
                t_total = time.time()

                t0 = time.time()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                trans_ms = (time.time() - t0) * 1000

                t0 = time.time()
                result = self._policy.infer(obs)
                infer_ms = (time.time() - t0) * 1000

                # Add timing info (compatible with OpenPI)
                result["server_timing"] = {
                    "infer_ms": infer_ms,
                    "trans_ms": trans_ms,
                    "prev_total_ms": prev_total_ms,
                }

                await websocket.send(packer.pack(result))

                prev_total_ms = (time.time() - t_total) * 1000
                print(f"trans: {trans_ms:.1f}ms | infer: {infer_ms:.1f}ms | total: {prev_total_ms:.1f}ms")

            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                tb = traceback.format_exc()
                logging.error(tb)
                await websocket.send(tb)  # Send error as string (client expects bytes, string = error)
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


@dataclass
class ServerConfig:
    # fmt: off
    pretrained_checkpoint: str = ""
    unnorm_key: str = "franka"
    host: str = "0.0.0.0"
    port: int = 55555
    use_l1_regression: bool = True
    num_images_in_input: int = 2        # primary + wrist
    use_proprio: bool = True

    # Zero-shot mode: use pretrained openvla-7b (discrete tokens, 7D, no chunking)
    zero_shot: bool = False
    # fmt: on


@draccus.wrap()
def main(cfg: ServerConfig) -> None:
    policy = OpenVLAPolicy(cfg)
    server = WebsocketPolicyServer(policy, host=cfg.host, port=cfg.port)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
