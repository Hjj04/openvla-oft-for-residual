#!/usr/bin/env python3
"""
offline_merge.py

训练完成后，将 LoRA adapter 合并到 base model 并保存完整模型。

用法:
    python offline_merge.py \
        --base_model_path /mnt/bos/kg35ez/openvla_weights/pretrained \
        --checkpoint_dir /mnt/bos/kg35ez/openvla_runs/xxx--5000_chkpt \
        --output_dir /mnt/bos/kg35ez/openvla_runs/xxx--5000_chkpt/merged

    # 如果不指定 output_dir，默认保存到 checkpoint_dir/merged
"""

import argparse
import json
import os
import shutil

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Offline LoRA merge for OpenVLA-OFT")
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="预训练 base model 路径，例如 /mnt/bos/kg35ez/openvla_weights/pretrained",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="训练产生的 checkpoint 目录，例如 /mnt/bos/kg35ez/openvla_runs/xxx--5000_chkpt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="合并后模型的保存路径，默认为 checkpoint_dir/merged",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 默认输出路径
    if args.output_dir is None:
        args.output_dir = os.path.join(args.checkpoint_dir, "merged")

    adapter_dir = os.path.join(args.checkpoint_dir, "lora_adapter")

    # 路径检查
    assert os.path.isdir(args.base_model_path), f"base_model_path 不存在: {args.base_model_path}"
    assert os.path.isdir(adapter_dir), f"lora_adapter 目录不存在: {adapter_dir}"

    print(f"Base model : {args.base_model_path}")
    print(f"LoRA adapter: {adapter_dir}")
    print(f"Output dir  : {args.output_dir}")

    # 注册 OpenVLA 类到 HuggingFace Auto Classes
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # 在 CPU 上加载 base model，避免 GPU OOM
    print("\n[1/4] 加载 base model (CPU)...")
    base_vla = AutoModelForVision2Seq.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )

    # 加载 LoRA adapter
    print("[2/4] 加载 LoRA adapter...")
    merged_vla = PeftModel.from_pretrained(
        base_vla,
        adapter_dir,
        torch_dtype=torch.bfloat16,
    )

    # 合并权重
    print("[3/4] 合并 LoRA 权重到 base model...")
    merged_vla = merged_vla.merge_and_unload()

    # 保存合并后的模型
    print(f"[4/4] 保存合并模型到 {args.output_dir} ...")
    os.makedirs(args.output_dir, exist_ok=True)
    merged_vla.save_pretrained(args.output_dir)

    # 同时复制 processor / tokenizer 文件
    # NOTE: config.json 已由 save_pretrained 正确写入，不要从 checkpoint_dir 覆盖
    processor_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "preprocessor_config.json",
    ]
    print("复制 processor 文件...")
    for fname in processor_files:
        src = os.path.join(args.checkpoint_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(args.output_dir, fname))

    # 复制 dataset_statistics.json（推理时反归一化动作需要）
    stats_src = os.path.join(args.checkpoint_dir, "dataset_statistics.json")
    if os.path.isfile(stats_src):
        shutil.copy2(stats_src, os.path.join(args.output_dir, "dataset_statistics.json"))
        print("已复制 dataset_statistics.json")

    # 复制 proprio_projector 和 action_head 权重
    for pattern in ["proprio_projector", "action_head"]:
        for fname in os.listdir(args.checkpoint_dir):
            if fname.startswith(pattern) and fname.endswith(".pt"):
                shutil.copy2(
                    os.path.join(args.checkpoint_dir, fname),
                    os.path.join(args.output_dir, fname),
                )
                print(f"已复制 {fname}")

    print(f"\n完成！合并模型保存于: {args.output_dir}")


if __name__ == "__main__":
    main()