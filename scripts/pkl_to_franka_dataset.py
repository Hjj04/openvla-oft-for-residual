"""
pkl_to_franka_dataset.py

Stage 2: Demo PKL 预处理脚本
将原始 demo PKL 转换为 OpenVLA-OFT 训练格式:
- 四元数 → rot6d (如果尚未转换)
- State: xyz(3) + rot6d(6) + gripper(1) = 10D
- Action: next_state 绝对位姿 (10D)
- 保持 PKL list-of-transitions 格式

用法:
    python scripts/pkl_to_franka_dataset.py \
        --input_dir /path/to/raw_demos \
        --output_dir /path/to/processed_demos \
        --task_name "pick red chili pepper"
"""

import argparse
import glob
import os
import pickle

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


def normalize_vector(v: np.ndarray) -> np.ndarray:
    v_mag = np.linalg.norm(v, axis=-1, keepdims=True)
    v_mag = np.maximum(v_mag, 1e-8)
    return v / v_mag


def matrix_to_rotation6d(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim == 2:
        matrix = matrix[np.newaxis, ...]
    rot_6d = matrix[..., :2]
    batch_dim = matrix.shape[0]
    rot_6d = rot_6d.swapaxes(1, 2).reshape(batch_dim, 6)
    return rot_6d.squeeze()


def quaternion_to_rotation6d(quat: np.ndarray) -> np.ndarray:
    """四元数 [x,y,z,w] → 6D rotation"""
    quat = np.atleast_2d(quat)
    r = R.from_quat(quat)
    matrix = r.as_matrix()
    return matrix_to_rotation6d(matrix)


def is_already_converted(transition: dict) -> bool:
    """检查 transition 是否已经转换为 10D 格式"""
    obs = transition.get("observations", {})
    if "state" in obs and len(obs["state"]) == 10:
        action = transition.get("action", np.array([]))
        if len(action) == 10:
            return True
    return False


def process_episode(episode: list, task_name: str) -> list:
    """处理单个 episode"""
    # 检查是否已转换
    if len(episode) > 0 and is_already_converted(episode[0]):
        # 已经是 10D 格式，只需确保 task_description 存在
        for frame in episode:
            obs = frame.get("observations", frame.get("observation", {}))
            if "task_description" not in obs:
                obs["task_description"] = task_name
        return episode

    processed = []
    for i, frame in enumerate(episode):
        obs = frame["observations"]
        next_obs = frame.get("next_observations", {})

        # 提取 tcp_pose
        orin = obs.get("orin_state", {})
        curr_tcp = orin.get("tcp_pose", None)

        if curr_tcp is None:
            # 尝试从 agent_pos 提取 (8D: 7 joints + gripper 或 7D: xyz + quat)
            agent_pos = obs.get("agent_pos", None)
            if agent_pos is not None and len(agent_pos) >= 7:
                curr_pos = agent_pos[:3]
                curr_quat = agent_pos[3:7]
                curr_gripper = np.array([agent_pos[7] if len(agent_pos) > 7 else 0.0])
            else:
                raise ValueError(f"Cannot extract pose from frame {i}")
        else:
            curr_pos = curr_tcp[:3]
            curr_quat = curr_tcp[3:]
            curr_gripper = np.array([orin.get("gripper_pose", 0.0)]).flatten()

        # 转换四元数 → rot6d
        curr_rot6d = quaternion_to_rotation6d(curr_quat)
        new_state = np.concatenate([curr_pos, curr_rot6d, curr_gripper]).astype(np.float32)

        # Action = next state (绝对位姿)
        if next_obs and "orin_state" in next_obs:
            next_tcp = next_obs["orin_state"]["tcp_pose"]
            next_pos = next_tcp[:3]
            next_quat = next_tcp[3:]
            next_gripper = np.array([next_obs["orin_state"].get("gripper_pose", 0.0)]).flatten()
            next_rot6d = quaternion_to_rotation6d(next_quat)
            new_action = np.concatenate([next_pos, next_rot6d, next_gripper]).astype(np.float32)
        elif i + 1 < len(episode):
            # 用下一帧的 state 作为 action
            next_frame = episode[i + 1]
            next_obs_inner = next_frame.get("observations", {})
            next_orin = next_obs_inner.get("orin_state", {})
            if "tcp_pose" in next_orin:
                next_tcp = next_orin["tcp_pose"]
                next_pos = next_tcp[:3]
                next_quat = next_tcp[3:]
                next_gripper = np.array([next_orin.get("gripper_pose", 0.0)]).flatten()
                next_rot6d = quaternion_to_rotation6d(next_quat)
                new_action = np.concatenate([next_pos, next_rot6d, next_gripper]).astype(np.float32)
            else:
                new_action = new_state.copy()  # 最后一帧 pad
        else:
            new_action = new_state.copy()  # 最后一帧 pad

        # 更新 frame
        obs["state"] = new_state
        obs["agent_pos"] = new_state
        if "task_description" not in obs:
            obs["task_description"] = task_name
        frame["action"] = new_action

        # 提取力/力矩 (可选)
        if "orin_state" in obs:
            orin = obs["orin_state"]
            if "tcp_force" in orin and "tcp_torque" in orin:
                wrench = np.concatenate([orin["tcp_force"], orin["tcp_torque"]]).astype(np.float32)
                obs["tcp_wrench"] = wrench
            if "tau_J" in orin:
                obs["effort"] = np.array(orin["tau_J"], dtype=np.float32)

        processed.append(frame)

    return processed


def main():
    parser = argparse.ArgumentParser(description="Demo PKL → OpenVLA-OFT 训练格式转换")
    parser.add_argument("--input_dir", type=str, required=True, help="输入原始 PKL 文件夹")
    parser.add_argument("--output_dir", type=str, required=True, help="输出处理后的文件夹")
    parser.add_argument("--task_name", type=str, default="", help="任务描述 (如果 PKL 中没有)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pkl_files = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.pkl"), recursive=True))
    if not pkl_files:
        print(f"Warning: no .pkl files found in {args.input_dir}")
        return

    print(f"Found {len(pkl_files)} PKL files, processing...")

    stats = {"total_episodes": 0, "total_frames": 0, "skipped": 0}
    for pkl_file in tqdm(pkl_files):
        try:
            with open(pkl_file, "rb") as f:
                episode = pickle.load(f)

            if not isinstance(episode, list):
                stats["skipped"] += 1
                continue

            processed = process_episode(episode, args.task_name)

            rel_path = os.path.relpath(pkl_file, args.input_dir)
            save_path = os.path.join(args.output_dir, rel_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, "wb") as f:
                pickle.dump(processed, f)

            stats["total_episodes"] += 1
            stats["total_frames"] += len(processed)
        except Exception as e:
            print(f"Error processing {pkl_file}: {e}")
            stats["skipped"] += 1

    print(f"\nDone! Episodes: {stats['total_episodes']}, Frames: {stats['total_frames']}, Skipped: {stats['skipped']}")


if __name__ == "__main__":
    main()
