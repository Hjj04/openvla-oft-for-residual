"""
finetune_franka.py

Fine-tunes OpenVLA-OFT on Franka PKL data via LoRA.
Supports both Base SFT (Stage 3) and Weighted SFT (Stage 7) with curriculum learning.

Usage:
    # Base SFT (Stage 3)
    torchrun --nproc_per_node 1 vla-scripts/finetune_franka.py \
        --vla_path openvla/openvla-7b \
        --stage sft \
        --pkl_data_dirs /path/to/task1 /path/to/task2 \
        --task_names "pick red chili pepper" "put objects into basket" \
        --max_steps 50000

    # Weighted SFT (Stage 7)
    torchrun --nproc_per_node 1 vla-scripts/finetune_franka.py \
        --vla_path /path/to/base_sft_checkpoint \
        --stage weighted_sft \
        --pkl_data_dirs /path/to/weighted_task1 /path/to/weighted_task2 \
        --task_names "pick red chili pepper" "put objects into basket" \
        --weight_mode executed_scale \
        --max_steps 30000
"""

import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import draccus
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from accelerate import PartialState
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import ProprioProjector
from prismatic.training.curriculum_weight import CurriculumConfig, CurriculumWeightCalculator
from prismatic.training.task_balanced_sampler import TaskBalancedSampler
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)
from prismatic.vla.datasets.franka_dataset import FrankaDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"

    # Stage: "sft" (Base SFT) or "weighted_sft" (Weighted SFT with curriculum)
    stage: str = "sft"

    # Dataset
    pkl_data_dirs: List[str] = field(default_factory=list)   # PKL 数据目录列表
    task_names: List[str] = field(default_factory=list)      # 任务名称列表
    run_root_dir: Path = Path("runs")

    # Algorithm
    use_l1_regression: bool = True
    use_film: bool = False
    num_images_in_input: int = 2                             # 双目: primary + wrist
    use_proprio: bool = True

    # Training
    batch_size: int = 8
    learning_rate: float = 5e-4
    lr_warmup_steps: int = 0
    num_steps_before_decay: int = 100_000
    grad_accumulation_steps: int = 1
    max_steps: int = 200_000
    max_epochs: int = 1000                                   # Map-style dataset 需要 epoch 循环
    save_freq: int = 5000
    save_latest_checkpoint_only: bool = False
    resume: bool = False
    resume_step: Optional[int] = None
    image_aug: bool = True

    # LoRA
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    merge_lora_during_training: bool = True

    # Weighted SFT / Curriculum
    weight_mode: str = "executed_scale"                      # executed_scale | residual_xyz_norm | residual_full_norm
    curriculum_sigma: float = 0.3
    curriculum_w_max: float = 2.0
    curriculum_r_peak_start: float = 0.2
    curriculum_r_peak_end: float = 0.8
    curriculum_schedule: str = "default"                      # default | linear | cosine | step
    scale_power: float = 1.0
    scale_use_curriculum: bool = True                         # executed_scale 模式是否启用高斯 curriculum

    # Wandb
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"
    run_id_note: Optional[str] = None
    run_id_override: Optional[str] = None
    wandb_log_freq: int = 10
    # fmt: on


# === Utility Functions (reused from finetune.py) ===

def remove_ddp_in_checkpoint(state_dict) -> dict:
    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == "module.":
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def get_run_id(cfg: FinetuneConfig) -> str:
    if cfg.run_id_override is not None:
        return cfg.run_id_override
    elif cfg.resume:
        run_id = cfg.vla_path.split("/")[-1]
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
        return run_id
    else:
        task_str = "_".join(cfg.task_names[:2]) if cfg.task_names else "franka"
        run_id = (
            f"{cfg.vla_path.split('/')[-1]}+{task_str}"
            f"+{cfg.stage}+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_lora:
            run_id += f"+lora-r{cfg.lora_rank}"
        if cfg.stage == "weighted_sft":
            run_id += f"+{cfg.weight_mode}"
        if cfg.run_id_note:
            run_id += f"--{cfg.run_id_note}"
        return run_id


def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)


def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)


def count_parameters(module: nn.Module, name: str) -> None:
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")


def init_module(module_class, module_name, cfg, device_id, module_args, to_bf16=False, find_unused=False):
    module = module_class(**module_args)
    count_parameters(module, module_name)
    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)
    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)
    return wrap_ddp(module, device_id, find_unused)


def compute_smoothened_metrics(metrics_deques) -> dict:
    smoothened = {}
    for name, d in metrics_deques.items():
        if d and len(d) > 0:
            smoothened[name] = sum(d) / len(d)
    return smoothened


def log_metrics_to_wandb(metrics, prefix, step, wandb_entity) -> None:
    log_dict = {}
    for name, value in metrics.items():
        if name == "loss_value":
            log_dict[f"{prefix}/Loss"] = value
        else:
            log_dict[f"{prefix}/{name.replace('_', ' ').title()}"] = value
    wandb_entity.log(log_dict, step=step)


def run_forward_pass(
    vla,
    action_head,
    proprio_projector,
    batch,
    action_tokenizer,
    device_id,
    use_l1_regression,
    use_proprio,
    use_film,
    num_patches,
    # Weighted SFT 参数
    curriculum_calculator=None,
    current_step=0,
    num_tasks=1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Forward pass with optional weighted L1 loss for curriculum learning.

    Returns:
        (loss, metrics_dict)
    """
    metrics = {}

    # Ground-truth actions
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)

    # VLA forward pass
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"],
            output_hidden_states=True,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            use_film=use_film,
        )

    # Action masks
    ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

    if use_l1_regression:
        # Extract action hidden states
        last_hidden_states = output.hidden_states[-1]
        text_hidden_states = last_hidden_states[:, num_patches:-1]
        batch_size = batch["input_ids"].shape[0]
        actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )

        # Predict actions
        predicted_actions = action_head.module.predict_action(actions_hidden_states)

        # === Weighted L1 Loss ===
        weight_values = batch.get("weight_values", None)
        task_ids = batch.get("task_ids", None)

        if weight_values is not None and curriculum_calculator is not None:
            weight_values = weight_values.to(device_id)
            task_ids = task_ids.to(device_id) if task_ids is not None else torch.zeros(batch_size, device=device_id, dtype=torch.long)

            # Per-sample L1 loss
            per_sample_l1 = F.l1_loss(ground_truth_actions, predicted_actions, reduction="none")
            per_sample_l1 = per_sample_l1.mean(dim=(-1, -2))  # (B,)

            # Curriculum weights + metrics
            weights, curriculum_metrics = curriculum_calculator.compute_weights_with_task_balance(
                weight_values, task_ids, current_step=current_step, num_tasks=num_tasks
            )
            weights = weights.to(device_id).to(per_sample_l1.dtype)
            if weights.dim() > 1:
                weights = weights.squeeze(-1)

            # Weighted loss
            loss = (per_sample_l1 * weights).sum() / weights.sum()

            # Unweighted loss for comparison
            unweighted_loss = per_sample_l1.mean()
            metrics["unweighted_loss"] = unweighted_loss.item()

            # Per-task L1 loss
            for tid in range(num_tasks):
                mask = task_ids == tid
                if mask.any():
                    curriculum_metrics[f"task_{tid}_loss"] = per_sample_l1[mask].mean().item()

            metrics["curriculum_metrics"] = curriculum_metrics
        else:
            # Standard L1 loss (Base SFT)
            loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)

        metrics["loss_value"] = loss.item()

        # Detailed L1 losses
        ground_truth_curr = ground_truth_actions[:, 0]
        predicted_curr = predicted_actions[:, 0]
        ground_truth_next = ground_truth_actions[:, 1:]
        predicted_next = predicted_actions[:, 1:]
        metrics["curr_action_l1_loss"] = torch.nn.L1Loss()(ground_truth_curr, predicted_curr).item()
        metrics["next_actions_l1_loss"] = torch.nn.L1Loss()(ground_truth_next, predicted_next).item()
    else:
        # Discrete token prediction (fallback)
        loss = output.loss
        metrics["loss_value"] = loss.item()

    return loss, metrics


def save_training_checkpoint(cfg, run_dir, log_step, vla, processor, proprio_projector, action_head, dataset_statistics, distributed_state):
    """Save checkpoint."""
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        suffix = f"{log_step}_checkpoint.pt"

    adapter_dir = checkpoint_dir / "lora_adapter"

    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)

        # Save dataset statistics for inference denormalization
        import json
        stats_path = checkpoint_dir / "dataset_statistics.json"
        serializable_stats = {}
        for k, v in dataset_statistics.items():
            serializable_stats[k] = {}
            for k2, v2 in v.items():
                serializable_stats[k][k2] = {}
                for k3, v3 in v2.items():
                    if hasattr(v3, "tolist"):
                        serializable_stats[k][k2][k3] = v3.tolist()
                    else:
                        serializable_stats[k][k2][k3] = v3
        with open(stats_path, "w") as f:
            json.dump(serializable_stats, f, indent=2)

        # Inject Franka norm_stats into model config so predict_action works after loading
        if not hasattr(vla.module, "norm_stats") or vla.module.norm_stats is None:
            vla.module.norm_stats = {}
        vla.module.norm_stats.update(dataset_statistics)
        vla.module.config.norm_stats = vla.module.norm_stats

        print(f"Saving checkpoint for step {log_step}")

    dist.barrier()

    if distributed_state.is_main_process:
        processor.save_pretrained(checkpoint_dir)
        vla.module.save_pretrained(adapter_dir)

        if cfg.use_proprio and proprio_projector is not None:
            torch.save(proprio_projector.state_dict(), checkpoint_dir / f"proprio_projector--{suffix}")

        if cfg.use_l1_regression and action_head is not None:
            torch.save(action_head.state_dict(), checkpoint_dir / f"action_head--{suffix}")

        if cfg.use_film:
            torch.save(vla.module.vision_backbone.state_dict(), checkpoint_dir / f"vision_backbone--{suffix}")

    dist.barrier()

    # Merge LoRA
    if cfg.use_lora and cfg.merge_lora_during_training:
        base_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        )
        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        merged_vla = merged_vla.merge_and_unload()
        if distributed_state.is_main_process:
            merged_vla.save_pretrained(checkpoint_dir)
            print(f"Saved merged model for step {log_step} at: {checkpoint_dir}")
        dist.barrier()


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    """Fine-tunes OpenVLA-OFT on Franka PKL data. Supports Base SFT and Weighted SFT."""
    assert cfg.use_lora, "Only LoRA fine-tuning is supported."
    assert len(cfg.pkl_data_dirs) > 0, "Must specify --pkl_data_dirs"
    assert len(cfg.pkl_data_dirs) == len(cfg.task_names), "pkl_data_dirs and task_names must match"

    cfg.vla_path = cfg.vla_path.rstrip("/")
    is_weighted = cfg.stage == "weighted_sft"
    num_tasks = len(cfg.task_names)

    print(f"Fine-tuning OpenVLA on Franka | Stage: {cfg.stage} | Tasks: {cfg.task_names}")

    # Run ID and directory
    run_id = get_run_id(cfg)
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    # GPU setup
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    # Wandb init with resume support
    if distributed_state.is_main_process:
        wandb_id_path = run_dir / "wandb_id.txt"
        if cfg.resume and wandb_id_path.exists():
            wandb.init(
                id=wandb_id_path.read_text().strip(),
                resume="must",
                entity=cfg.wandb_entity,
                project=cfg.wandb_project,
            )
        else:
            wandb.init(
                entity=cfg.wandb_entity,
                project=cfg.wandb_project,
                name=f"ft+{run_id}",
                config={
                    "stage": cfg.stage,
                    "task_names": cfg.task_names,
                    "weight_mode": cfg.weight_mode if is_weighted else "none",
                    "batch_size": cfg.batch_size,
                    "learning_rate": cfg.learning_rate,
                    "max_steps": cfg.max_steps,
                    "lora_rank": cfg.lora_rank,
                },
            )
            wandb_id_path.write_text(wandb.run.id)

    # Print constants
    print(
        f"Constants: NUM_ACTIONS_CHUNK={NUM_ACTIONS_CHUNK}, ACTION_DIM={ACTION_DIM}, "
        f"PROPRIO_DIM={PROPRIO_DIM}, NORM_TYPE={ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )

    # Load model
    if model_is_on_hf_hub(cfg.vla_path):
        cfg.vla_path = snapshot_download(repo_id=cfg.vla_path)
    else:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    if distributed_state.is_main_process:
        update_auto_map(cfg.vla_path)
        check_model_logic_mismatch(cfg.vla_path)
    dist.barrier()

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device_id)

    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    # LoRA
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # FiLM (optional)
    if cfg.use_film:
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone, llm_dim=vla.llm_dim
        )
        if cfg.resume:
            state_dict = load_checkpoint("vision_backbone", cfg.vla_path, cfg.resume_step)
            vla.model.vision_backbone.load_state_dict(state_dict)
        vla.model.vision_backbone = vla.model.vision_backbone.to(device_id)

    vla = wrap_ddp(vla, device_id, find_unused=True)

    # Proprio projector
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector, "proprio_projector", cfg, device_id,
            {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM},
        )

    # L1 regression action head
    action_head = None
    if cfg.use_l1_regression:
        action_head = init_module(
            L1RegressionActionHead, "action_head", cfg, device_id,
            {"input_dim": vla.module.llm_dim, "hidden_dim": vla.module.llm_dim, "action_dim": ACTION_DIM},
            to_bf16=True,
        )

    # Num patches
    NUM_PATCHES = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()
    if cfg.use_proprio:
        NUM_PATCHES += 1

    # Optimizer
    trainable_params = [p for p in vla.parameters() if p.requires_grad]
    if action_head is not None:
        trainable_params += [p for p in action_head.parameters() if p.requires_grad]
    if proprio_projector is not None:
        trainable_params += [p for p in proprio_projector.parameters() if p.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    original_lr = optimizer.param_groups[0]["lr"]
    scheduler = MultiStepLR(optimizer, milestones=[cfg.num_steps_before_decay], gamma=0.1)

    # Action tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # === Create FrankaDataset ===
    use_wrist_image = cfg.num_images_in_input > 1
    train_dataset = FrankaDataset(
        data_dirs=cfg.pkl_data_dirs,
        task_names=cfg.task_names,
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        action_dim=ACTION_DIM,
        chunk_size=NUM_ACTIONS_CHUNK,
        proprio_dim=PROPRIO_DIM,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
        weighted=is_weighted,
    )

    # Save dataset statistics
    if distributed_state.is_main_process:
        import json
        stats_path = run_dir / "dataset_statistics.json"
        serializable = {}
        for k, v in train_dataset.dataset_statistics.items():
            serializable[k] = {}
            for k2, v2 in v.items():
                serializable[k][k2] = {k3: v3.tolist() if hasattr(v3, "tolist") else v3 for k3, v3 in v2.items()}
        with open(stats_path, "w") as f:
            json.dump(serializable, f, indent=2)

    # Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )

    # Sampler: TaskBalancedSampler for weighted SFT, DistributedSampler for base SFT
    if is_weighted:
        sampler = TaskBalancedSampler(
            train_dataset, task_id_key="task_id",
            shuffle=True, seed=42, cache_dir=str(run_dir / "sampler_cache"),
        )
    else:
        sampler = DistributedSampler(train_dataset, shuffle=True) if dist.is_initialized() else None

    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Curriculum calculator (weighted SFT only)
    curriculum_calculator = None
    if is_weighted:
        curriculum_config = CurriculumConfig(
            weight_mode=cfg.weight_mode,
            sigma=cfg.curriculum_sigma,
            w_max=cfg.curriculum_w_max,
            enabled=True,
            by_step=True,
            total_steps=cfg.max_steps,
            r_peak_start=cfg.curriculum_r_peak_start,
            r_peak_end=cfg.curriculum_r_peak_end,
            schedule=cfg.curriculum_schedule,
            scale_power=cfg.scale_power,
            scale_use_curriculum=cfg.scale_use_curriculum,
            batch_normalize=True,
        )
        curriculum_calculator = CurriculumWeightCalculator(curriculum_config)

    # Metrics deques
    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "grad_norm": deque(maxlen=cfg.grad_accumulation_steps),
    }
    if is_weighted:
        recent_metrics["unweighted_loss"] = deque(maxlen=cfg.grad_accumulation_steps)

    # Log camera views at step 0
    logged_images = False

    # === Training Loop (epoch-based for map-style dataset) ===
    global_batch_idx = 0
    global_step = 0  # gradient steps

    print(f"Starting training: {len(train_dataset)} samples, ~{len(dataloader)} batches/epoch")

    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()

        for epoch in range(cfg.max_epochs):
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            for batch in dataloader:
                # Log camera images once
                if not logged_images and distributed_state.is_main_process:
                    try:
                        pv = batch["pixel_values"]
                        n_log = min(3, pv.shape[0])
                        images_to_log = []
                        for i in range(n_log):
                            # pixel_values may have multiple images concatenated
                            img_np = pv[i].permute(1, 2, 0).cpu().numpy()
                            img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8) * 255).astype("uint8")
                            images_to_log.append(wandb.Image(img_np))
                        wandb.log({"camera_views": images_to_log}, step=0)
                    except Exception:
                        pass
                    logged_images = True

                # Forward pass
                loss, metrics = run_forward_pass(
                    vla=vla,
                    action_head=action_head,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    batch=batch,
                    action_tokenizer=action_tokenizer,
                    device_id=device_id,
                    use_l1_regression=cfg.use_l1_regression,
                    use_proprio=cfg.use_proprio,
                    use_film=cfg.use_film,
                    num_patches=NUM_PATCHES,
                    curriculum_calculator=curriculum_calculator,
                    current_step=global_step,
                    num_tasks=num_tasks,
                )

                # Backward
                normalized_loss = loss / cfg.grad_accumulation_steps
                normalized_loss.backward()

                # Store metrics
                for name, value in metrics.items():
                    if name in recent_metrics:
                        recent_metrics[name].append(value)

                # Gradient norm (compute before optimizer step)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                recent_metrics["grad_norm"].append(grad_norm.item())

                # Gradient step
                gradient_step_idx = global_batch_idx // cfg.grad_accumulation_steps
                log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx

                # Log to wandb
                if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                    smoothened = compute_smoothened_metrics(recent_metrics)
                    log_metrics_to_wandb(smoothened, "VLA Train", log_step, wandb)

                    # Log learning rate
                    wandb.log({"VLA Train/Learning Rate": scheduler.get_last_lr()[0]}, step=log_step)

                    # Log grad norm
                    if "grad_norm" in smoothened:
                        wandb.log({"VLA Train/Grad Norm": smoothened["grad_norm"]}, step=log_step)

                    # Log curriculum metrics (weighted SFT only)
                    if is_weighted and "curriculum_metrics" in metrics:
                        cm = metrics["curriculum_metrics"]
                        curriculum_log = {}
                        for k, v in cm.items():
                            if k.startswith("task_"):
                                curriculum_log[f"per_task/{k}"] = v
                            else:
                                curriculum_log[f"curriculum/{k}"] = v
                        wandb.log(curriculum_log, step=log_step)

                # LR warmup
                if cfg.lr_warmup_steps > 0:
                    lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)
                    current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                    for pg in optimizer.param_groups:
                        pg["lr"] = current_lr

                # Optimizer step
                if (global_batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    progress.update()
                    global_step += 1

                # Save checkpoint
                if global_step > 0 and log_step % cfg.save_freq == 0 and (global_batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                    save_training_checkpoint(
                        cfg=cfg, run_dir=run_dir, log_step=log_step,
                        vla=vla, processor=processor,
                        proprio_projector=proprio_projector if cfg.use_proprio else None,
                        action_head=action_head,
                        dataset_statistics=train_dataset.dataset_statistics,
                        distributed_state=distributed_state,
                    )

                global_batch_idx += 1

                # Stop at max_steps
                if global_step >= cfg.max_steps:
                    break

            if global_step >= cfg.max_steps:
                print(f"Reached max_steps={cfg.max_steps}. Stopping.")
                break

            print(f"Epoch {epoch + 1} done. Global step: {global_step}")

    # Final save
    if global_step > 0:
        save_training_checkpoint(
            cfg=cfg, run_dir=run_dir, log_step=global_step,
            vla=vla, processor=processor,
            proprio_projector=proprio_projector if cfg.use_proprio else None,
            action_head=action_head,
            dataset_statistics=train_dataset.dataset_statistics,
            distributed_state=distributed_state,
        )

    print("Training complete!")


if __name__ == "__main__":
    finetune()
