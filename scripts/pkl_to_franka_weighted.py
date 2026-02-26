"""
pkl_to_franka_weighted.py

Stage 6: 残差轨迹 PKL → 加权 SFT 训练格式转换
从残差 RL 收集的轨迹中提取 weight_value，并进行分位数归一化。

输入: 残差 RL 收集的轨迹 PKL (包含 base_action, executed_scale 等)
输出: 带 weight_value 和 task_id 的训练 PKL

支持的 weight_mode:
- executed_scale: 使用动态 scale 值作为权重 (默认)
- residual_xyz_norm: 使用 XYZ 残差范数
- residual_full_norm: 使用全维度残差范数

用法:
    python scripts/pkl_to_franka_weighted.py \
        --input_dirs /path/to/task1_trajs /path/to/task2_trajs \
        --task_names "pick red chili pepper" "put objects into basket" \
        --output_dir /path/to/weighted_data \
        --weight_mode executed_scale
"""

import argparse
import glob
import os
import pickle

import numpy as np
from tqdm import tqdm


def extract_weight_value(transition: dict, weight_mode: str) -> float:
    """从 transition 中提取权重值"""
    if weight_mode == "executed_scale":
        # 动态 scale 值 (由残差 RL 的 Q-value advantage 决定)
        scale = transition.get("executed_scale", transition.get("scale", 1.0))
        if isinstance(scale, (np.ndarray,)):
            scale = float(scale.item() if scale.size == 1 else scale[0])
        return float(scale)

    elif weight_mode == "residual_xyz_norm":
        # XYZ 残差范数
        action = np.array(transition.get("action", np.zeros(10)), dtype=np.float32)
        base_action = np.array(transition.get("base_action", action), dtype=np.float32)
        residual = action[:3] - base_action[:3]
        return float(np.linalg.norm(residual))

    elif weight_mode == "residual_full_norm":
        # 全维度残差范数
        action = np.array(transition.get("action", np.zeros(10)), dtype=np.float32)
        base_action = np.array(transition.get("base_action", action), dtype=np.float32)
        residual = action - base_action
        return float(np.linalg.norm(residual))

    else:
        return 1.0


def quantile_normalize(values: np.ndarray, q_low: float = 20.0, q_high: float = 80.0) -> np.ndarray:
    """分位数归一化到 [0, 1]"""
    low = np.percentile(values, q_low)
    high = np.percentile(values, q_high)
    if high - low < 1e-8:
        return np.ones_like(values) * 0.5
    normalized = (values - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description="残差轨迹 PKL → 加权 SFT 格式转换")
    parser.add_argument("--input_dirs", type=str, nargs="+", required=True, help="输入轨迹 PKL 文件夹列表")
    parser.add_argument("--task_names", type=str, nargs="+", required=True, help="任务名称列表")
    parser.add_argument("--output_dir", type=str, required=True, help="输出加权数据文件夹")
    parser.add_argument("--weight_mode", type=str, default="executed_scale",
                        choices=["executed_scale", "residual_xyz_norm", "residual_full_norm"])
    parser.add_argument("--q_low", type=float, default=20.0, help="分位数归一化下界百分位 (P20)")
    parser.add_argument("--q_high", type=float, default=80.0, help="分位数归一化上界百分位 (P80)")
    args = parser.parse_args()

    assert len(args.input_dirs) == len(args.task_names), "input_dirs 和 task_names 数量必须一致"
    os.makedirs(args.output_dir, exist_ok=True)

    # Pass 1: 收集每个任务的 weight_value (按任务分组)
    print(f"Pass 1: Collecting weight values (mode={args.weight_mode})...")
    task_weight_values = {tid: [] for tid in range(len(args.task_names))}  # task_id -> [wv, ...]
    all_episodes = []  # [(episode, task_id, task_name, pkl_file, ep_wvs)]

    for task_id, (input_dir, task_name) in enumerate(zip(args.input_dirs, args.task_names)):
        pkl_files = sorted(glob.glob(os.path.join(input_dir, "**", "*.pkl"), recursive=True))
        print(f"  Task {task_id} ({task_name}): {len(pkl_files)} files from {input_dir}")

        for pkl_file in pkl_files:
            try:
                with open(pkl_file, "rb") as f:
                    episode = pickle.load(f)
                if not isinstance(episode, list):
                    continue

                ep_wvs = []
                for transition in episode:
                    wv = extract_weight_value(transition, args.weight_mode)
                    ep_wvs.append(wv)
                    task_weight_values[task_id].append(wv)

                all_episodes.append((episode, task_id, task_name, pkl_file, ep_wvs))
            except Exception as e:
                print(f"  Error loading {pkl_file}: {e}")

    # 按任务分位数归一化 (P20-P80)
    print(f"\nPer-task quantile normalization (P{args.q_low}-P{args.q_high}):")
    task_norm_params = {}  # task_id -> (low, high)
    for tid in range(len(args.task_names)):
        wvs = np.array(task_weight_values[tid], dtype=np.float32)
        if len(wvs) == 0:
            task_norm_params[tid] = (0.0, 1.0)
            continue
        low = np.percentile(wvs, args.q_low)
        high = np.percentile(wvs, args.q_high)
        task_norm_params[tid] = (low, high)
        print(f"  Task {tid} ({args.task_names[tid]}): {len(wvs)} frames, "
              f"raw=[{wvs.min():.4f}, {wvs.max():.4f}], P{args.q_low}={low:.4f}, P{args.q_high}={high:.4f}")

    # Pass 2: 写入按任务归一化后的 weight_value 和 task_id
    print("\nPass 2: Writing weighted PKL files...")
    stats = {"total_episodes": 0, "total_frames": 0}

    for episode, task_id, task_name, pkl_file, ep_wvs in tqdm(all_episodes):
        low, high = task_norm_params[task_id]
        for idx_in_ep, transition in enumerate(episode):
            raw_wv = ep_wvs[idx_in_ep]
            if high - low < 1e-8:
                norm_wv = 0.5
            else:
                norm_wv = float(np.clip((raw_wv - low) / (high - low), 0.0, 1.0))
            transition["weight_value"] = norm_wv
            transition["task_id"] = int(task_id)

            # 确保 task_description 存在
            obs = transition.get("observations", transition.get("observation", {}))
            if "task_description" not in obs:
                obs["task_description"] = task_name

        # 保存到 output_dir/task_{task_id}/
        task_dir = os.path.join(args.output_dir, f"task_{task_id}")
        os.makedirs(task_dir, exist_ok=True)
        save_name = os.path.basename(pkl_file)
        save_path = os.path.join(task_dir, save_name)

        with open(save_path, "wb") as f:
            pickle.dump(episode, f)

        stats["total_episodes"] += 1
        stats["total_frames"] += len(episode)

    print(f"\nDone! Episodes: {stats['total_episodes']}, Frames: {stats['total_frames']}")
    print(f"Output: {args.output_dir}")

    # 保存归一化统计量
    all_wvs = np.concatenate([np.array(v) for v in task_weight_values.values() if len(v) > 0])
    norm_info = {
        "weight_mode": args.weight_mode,
        "normalization": "per_task",
        "q_low": args.q_low,
        "q_high": args.q_high,
        "global_raw_mean": float(all_wvs.mean()),
        "global_raw_std": float(all_wvs.std()),
        "global_raw_min": float(all_wvs.min()),
        "global_raw_max": float(all_wvs.max()),
        "task_names": args.task_names,
        "per_task_params": {
            str(tid): {"p_low": float(task_norm_params[tid][0]), "p_high": float(task_norm_params[tid][1]),
                        "n_frames": len(task_weight_values[tid])}
            for tid in range(len(args.task_names))
        },
    }
    import json
    with open(os.path.join(args.output_dir, "weight_norm_info.json"), "w") as f:
        json.dump(norm_info, f, indent=2)
    print(f"Saved normalization info to {args.output_dir}/weight_norm_info.json")


if __name__ == "__main__":
    main()
