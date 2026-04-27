import json
import math
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from peft import PeftModel
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    get_linear_schedule_with_warmup,
    set_seed,
)

from train_lora_full import (
    SIQA_S_FAMILY,
    SIQA_U_FAMILY,
    FractionalFamilySampler,
    MixedTrainCollator,
    ScoreTokenHelper,
    SiqaScoreDataset,
    SiqaScoreEvalCollator,
    SiqaUnderstandDataset,
    SiqaUnderstandEvalCollator,
    clear_cuda_cache,
    compute_combined_score,
    compute_mixed_loss,
    configure_image_processor,
    evaluate_siqa_s_model,
    evaluate_siqa_u_model,
    is_multi_node_run,
    load_json_file,
    load_resume_metadata,
    log_lora_setup,
    resolve_lora_target_modules,
    save_checkpoint,
    save_resume_checkpoint,
    summarize_parameter_counts,
    write_history,
)
from train_lora_siqa_s_from_domain import validate_init_adapter_config


RESUME_CONFIG_KEYS = (
    "model",
    "train_siqa_s_jsonl",
    "train_siqa_u_jsonl",
    "val_siqa_s_jsonl",
    "val_siqa_u_jsonl",
    "root",
    "epochs",
    "lr",
    "weight_decay",
    "warmup_ratio",
    "max_length",
    "per_device_batch_size",
    "gradient_accumulation_steps",
    "gradient_clip_norm",
    "score_loss_weight",
    "score_loss_type",
    "score_huber_delta",
    "seed",
    "use_bf16",
    "max_new_tokens",
    "mixed_precision",
    "init_adapter_dir",
    "image_min_pixels",
    "image_max_pixels",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "lora_bias",
    "enable_text_lora",
    "enable_vision_lora",
    "lora_text_target_modules",
    "lora_vision_target_modules",
    "lora_target_modules",
    "train_sampling_mode",
    "siqa_s_epoch_multiplier",
    "siqa_u_epoch_multiplier",
    "best_metric_name",
)


def validate_resume_config(current_config: dict[str, Any], saved_config: dict[str, Any] | None) -> None:
    if saved_config is None:
        raise ValueError(
            "Refusing to auto-resume because resume state exists but no saved training configuration was found.\n"
            f"Resume directory: {current_config['resume_dir']}\n"
            "Expected either `trainer_state.json['config']` or `train_config.json`."
        )

    mismatches = [
        f"{key}: saved={saved_config.get(key)!r} current={current_config.get(key)!r}"
        for key in RESUME_CONFIG_KEYS
        if current_config.get(key) != saved_config.get(key)
    ]
    if mismatches:
        mismatch_text = "\n".join(mismatches)
        raise ValueError(
            "Refusing to auto-resume because the saved training configuration does not match the current run.\n"
            f"Resume directory: {current_config['resume_dir']}\n"
            f"Mismatched fields:\n{mismatch_text}"
        )


def main() -> None:
    model_name = "Qwen/Qwen3-VL-2B-Instruct"
    train_siqa_s_jsonl = "TrainSet/train_SIQA-S.jsonl"
    train_siqa_u_jsonl = "TrainSet/train_SIQA-U.jsonl"
    val_siqa_s_jsonl = "TrainSet/SIQA-S-valid.jsonl"
    val_siqa_u_jsonl = "TrainSet/SIQA-U-valid.jsonl"
    root = "TrainSet"
    output_dir = Path("checkpoints/qwen3-vl-2b-siqa-mixed-lora_from_domain")
    resume_dir = output_dir / "resume_state"
    init_adapter_dir = "checkpoints/qwen3-vl-2b-domain-lora_v3/best"
    epochs = 3
    lr = 1e-4
    weight_decay = 0.01
    warmup_ratio = 0.1
    max_length = 2048
    per_device_batch_size = 4
    gradient_accumulation_steps = 8
    gradient_clip_norm = 1.0
    score_loss_weight = 1.0
    score_loss_type = "huber"
    score_huber_delta = 0.5
    seed = 42
    num_workers = 1
    use_bf16 = True
    image_min_pixels = 256 * 256
    image_max_pixels = 28 * 28 * 1280
    distributed_timeout = timedelta(hours=2)
    max_new_tokens = 20
    log_every = 100
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    lora_bias = "none"
    enable_text_lora = True
    enable_vision_lora = True
    train_sampling_mode = "fractional_family"
    siqa_s_epoch_multiplier = 1.0
    siqa_u_epoch_multiplier = 0.5
    best_metric_name = "Combined_Score"

    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    mixed_precision = "no"
    if use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        mixed_precision = "bf16"

    process_group_kwargs = InitProcessGroupKwargs(timeout=distributed_timeout)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        kwargs_handlers=[process_group_kwargs],
    )
    shared_storage_available = False
    multi_node_run = is_multi_node_run(accelerator)
    use_resume_state = not (multi_node_run and not shared_storage_available)
    save_last_on_each_node = multi_node_run and not shared_storage_available

    if accelerator.is_main_process:
        print(f"Using mixed precision: {mixed_precision}")
        print(f"Output dir: {output_dir}")
        print(f"Distributed timeout: {distributed_timeout}")
        if not use_resume_state:
            print(
                "Multi-node run without shared storage detected: "
                "disabling exact resume-state save/load and saving `last` on each node locally."
            )

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    configure_image_processor(
        processor=processor,
        min_pixels=image_min_pixels,
        max_pixels=image_max_pixels,
    )
    eval_processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    configure_image_processor(
        processor=eval_processor,
        min_pixels=image_min_pixels,
        max_pixels=image_max_pixels,
    )
    eval_processor.tokenizer.padding_side = "left"
    score_tokens = ScoreTokenHelper(processor)

    if accelerator.is_main_process:
        print(
            "Configured processor image limits: "
            f"min_pixels={image_min_pixels} max_pixels={image_max_pixels}"
        )

    train_siqa_s_dataset = SiqaScoreDataset(train_siqa_s_jsonl, root)
    train_siqa_u_dataset = SiqaUnderstandDataset(train_siqa_u_jsonl, root)
    val_siqa_s_dataset = SiqaScoreDataset(val_siqa_s_jsonl, root)
    val_siqa_u_dataset = SiqaUnderstandDataset(val_siqa_u_jsonl, root)
    train_dataset = ConcatDataset([train_siqa_s_dataset, train_siqa_u_dataset])
    train_sampler = FractionalFamilySampler(
        siqa_s_count=len(train_siqa_s_dataset),
        siqa_u_count=len(train_siqa_u_dataset),
        siqa_u_indices_by_type=train_siqa_u_dataset.indices_by_type,
        sampling_mode=train_sampling_mode,
        siqa_s_epoch_multiplier=siqa_s_epoch_multiplier,
        siqa_u_epoch_multiplier=siqa_u_epoch_multiplier,
        seed=seed,
    )

    train_collator = MixedTrainCollator(processor, max_length, score_tokens)
    score_eval_collator = SiqaScoreEvalCollator(eval_processor, max_length)
    uqa_eval_collator = SiqaUnderstandEvalCollator(eval_processor, max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_device_batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=train_collator,
    )
    val_siqa_s_loader = DataLoader(
        val_siqa_s_dataset,
        batch_size=per_device_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=score_eval_collator,
    )
    val_siqa_u_loader = DataLoader(
        val_siqa_u_dataset,
        batch_size=per_device_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=uqa_eval_collator,
    )

    if accelerator.is_main_process:
        train_family_counts = {
            SIQA_S_FAMILY: len(train_siqa_s_dataset),
            SIQA_U_FAMILY: len(train_siqa_u_dataset),
        }
        val_family_counts = {
            SIQA_S_FAMILY: len(val_siqa_s_dataset),
            SIQA_U_FAMILY: len(val_siqa_u_dataset),
        }
        train_score_task_counts = {"perception": 0, "knowledge": 0}
        for example in train_siqa_s_dataset.examples:
            if example.score_task is not None:
                train_score_task_counts[example.score_task] += 1
        val_score_task_counts = {"perception": 0, "knowledge": 0}
        for example in val_siqa_s_dataset.examples:
            if example.score_task is not None:
                val_score_task_counts[example.score_task] += 1
        train_uqa_type_counts = {
            question_type: len(indices)
            for question_type, indices in train_siqa_u_dataset.indices_by_type.items()
        }
        val_uqa_type_counts = {
            question_type: len(indices)
            for question_type, indices in val_siqa_u_dataset.indices_by_type.items()
        }

        print(f"Train examples: {len(train_dataset)} {train_family_counts}")
        print(f"Val examples:   {sum(val_family_counts.values())} {val_family_counts}")
        print(f"Train SIQA-S task counts: {train_score_task_counts}")
        print(f"Val SIQA-S task counts:   {val_score_task_counts}")
        print(f"Train SIQA-U type counts: {train_uqa_type_counts}")
        print(f"Val SIQA-U type counts:   {val_uqa_type_counts}")
        print(
            "Train sampling: "
            f"mode={train_sampling_mode} "
            f"siqa_s_multiplier={siqa_s_epoch_multiplier} "
            f"siqa_u_multiplier={siqa_u_epoch_multiplier} "
            f"samples_per_epoch={len(train_sampler)}"
        )
        print(f"Train sampler summary: {train_sampler.summary()}")

    load_dtype = torch.bfloat16 if mixed_precision == "bf16" else None
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=load_dtype,
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    resolved_targets = resolve_lora_target_modules(
        model=model,
        enable_text_lora=enable_text_lora,
        enable_vision_lora=enable_vision_lora,
    )
    init_adapter_path = Path(init_adapter_dir) if init_adapter_dir is not None else None
    if init_adapter_path is None:
        raise ValueError("This script requires `init_adapter_dir` to point to a trained domain adapter.")
    if not init_adapter_path.exists():
        raise FileNotFoundError(f"init_adapter_dir does not exist: {init_adapter_path}")

    validate_init_adapter_config(
        init_adapter_dir=init_adapter_path,
        model_name=model_name,
        resolved_targets=resolved_targets,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_bias=lora_bias,
    )
    model = PeftModel.from_pretrained(model, str(init_adapter_path), is_trainable=True)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.train()

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("LoRA setup completed, but no trainable parameters were found.")

    total_parameters, trainable_parameter_count = summarize_parameter_counts(model)
    log_lora_setup(
        accelerator=accelerator,
        resolved_targets=resolved_targets,
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameter_count,
    )

    optimizer = AdamW(trainable_parameters, lr=lr, weight_decay=weight_decay)

    model, optimizer, train_loader = accelerator.prepare(
        model,
        optimizer,
        train_loader,
    )

    steps_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    total_train_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = int(total_train_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )
    scheduler = accelerator.prepare_scheduler(scheduler)

    config_record = {
        "model": model_name,
        "train_siqa_s_jsonl": train_siqa_s_jsonl,
        "train_siqa_u_jsonl": train_siqa_u_jsonl,
        "val_siqa_s_jsonl": val_siqa_s_jsonl,
        "val_siqa_u_jsonl": val_siqa_u_jsonl,
        "root": root,
        "output_dir": str(output_dir),
        "resume_dir": str(resume_dir),
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "max_length": max_length,
        "per_device_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clip_norm": gradient_clip_norm,
        "score_loss_weight": score_loss_weight,
        "score_loss_type": score_loss_type,
        "score_huber_delta": score_huber_delta,
        "seed": seed,
        "num_workers": num_workers,
        "use_bf16": use_bf16,
        "image_min_pixels": image_min_pixels,
        "image_max_pixels": image_max_pixels,
        "max_new_tokens": max_new_tokens,
        "log_every": log_every,
        "mixed_precision": mixed_precision,
        "init_adapter_dir": str(init_adapter_path),
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_bias": lora_bias,
        "enable_text_lora": enable_text_lora,
        "enable_vision_lora": enable_vision_lora,
        "lora_text_target_modules": resolved_targets["text"],
        "lora_vision_target_modules": resolved_targets["vision"],
        "lora_target_modules": resolved_targets["all"],
        "train_sampling_mode": train_sampling_mode,
        "siqa_s_epoch_multiplier": siqa_s_epoch_multiplier,
        "siqa_u_epoch_multiplier": siqa_u_epoch_multiplier,
        "best_metric_name": best_metric_name,
    }

    resume_metadata = load_resume_metadata(resume_dir) if use_resume_state else None
    saved_resume_config = None
    if resume_metadata is not None:
        saved_resume_config = resume_metadata.get("config")
        if not isinstance(saved_resume_config, dict):
            saved_resume_config = load_json_file(output_dir / "train_config.json")
        validate_resume_config(config_record, saved_resume_config)

    if accelerator.is_main_process:
        with open(output_dir / "train_config.json", "w", encoding="utf-8") as handle:
            json.dump(config_record, handle, indent=2, ensure_ascii=False)

    microsteps_per_epoch = len(train_loader)
    optimizer_steps_per_epoch = max(1, math.ceil(microsteps_per_epoch / gradient_accumulation_steps))
    total_optimizer_steps = optimizer_steps_per_epoch * epochs

    if accelerator.is_main_process:
        print(f"Train microsteps per epoch: {microsteps_per_epoch}")
        print(f"Optimizer steps per epoch: {optimizer_steps_per_epoch}")
        print(f"Total optimizer steps: {total_optimizer_steps}")

    best_metric = -float("inf")
    global_step = 0
    optimizer_steps = 0
    start_epoch = 1

    if resume_metadata is not None:
        accelerator.load_state(str(resume_dir))
        start_epoch = int(resume_metadata.get("epoch", 0)) + 1
        global_step = int(resume_metadata.get("global_step", 0))
        optimizer_steps = int(resume_metadata.get("optimizer_steps", 0))
        best_metric = float(resume_metadata.get("best_metric", -float("inf")))
        if accelerator.is_main_process:
            print(
                f"Resuming training from {resume_dir} at epoch={start_epoch} "
                f"global_step={global_step} optimizer_steps={optimizer_steps}"
            )

    for epoch in range(start_epoch, epochs + 1):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        if hasattr(train_loader, "set_epoch"):
            train_loader.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        epoch_lm_loss = 0.0
        epoch_score_loss = 0.0
        epoch_steps = 0
        accumulation_loss = 0.0
        accumulation_lm_loss = 0.0
        accumulation_score_loss = 0.0
        accumulation_microsteps = 0
        epoch_optimizer_steps = 0
        train_progress = tqdm(
            total=optimizer_steps_per_epoch,
            desc=f"Epoch {epoch}/{epochs}",
            disable=not accelerator.is_main_process,
            leave=True,
        )

        for step, batch in enumerate(train_loader, start=1):
            target_scores = batch.pop("target_scores")
            quality_positions = batch.pop("quality_positions")
            siqa_s_mask = batch.pop("siqa_s_mask")
            model_inputs = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}

            with accelerator.accumulate(model):
                outputs = model(**model_inputs)
                total_loss, lm_loss, score_loss = compute_mixed_loss(
                    outputs=outputs,
                    siqa_s_mask=siqa_s_mask,
                    quality_positions=quality_positions,
                    target_scores=target_scores,
                    score_tokens=score_tokens,
                    score_loss_weight=score_loss_weight,
                    score_loss_type=score_loss_type,
                    score_huber_delta=score_huber_delta,
                )
                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_parameters, gradient_clip_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            reduced_total_loss = accelerator.gather(total_loss.detach().unsqueeze(0)).mean().item()
            reduced_lm_loss = accelerator.gather(lm_loss.detach().unsqueeze(0)).mean().item()
            reduced_score_loss = accelerator.gather(score_loss.detach().unsqueeze(0)).mean().item()

            epoch_loss += reduced_total_loss
            epoch_lm_loss += reduced_lm_loss
            epoch_score_loss += reduced_score_loss
            epoch_steps += 1
            global_step += 1
            accumulation_loss += reduced_total_loss
            accumulation_lm_loss += reduced_lm_loss
            accumulation_score_loss += reduced_score_loss
            accumulation_microsteps += 1

            if accelerator.sync_gradients:
                optimizer_steps += 1
                epoch_optimizer_steps += 1
                window_loss = accumulation_loss / max(accumulation_microsteps, 1)
                window_lm_loss = accumulation_lm_loss / max(accumulation_microsteps, 1)
                window_score_loss = accumulation_score_loss / max(accumulation_microsteps, 1)
                if accelerator.is_main_process:
                    train_progress.update(1)
                    train_progress.set_postfix(
                        loss=f"{window_loss:.4f}",
                        lm=f"{window_lm_loss:.4f}",
                        score=f"{window_score_loss:.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    )
                if accelerator.is_main_process and optimizer_steps % log_every == 0:
                    batch_progress = 100.0 * step / max(microsteps_per_epoch, 1)
                    epoch_progress = 100.0 * epoch_optimizer_steps / max(optimizer_steps_per_epoch, 1)
                    total_progress = 100.0 * optimizer_steps / max(total_optimizer_steps, 1)
                    train_progress.write(
                        f"epoch={epoch}/{epochs} "
                        f"batch_step={step}/{microsteps_per_epoch} ({batch_progress:.1f}%) "
                        f"epoch_optimizer_step={epoch_optimizer_steps}/{optimizer_steps_per_epoch} "
                        f"({epoch_progress:.1f}%) "
                        f"optimizer_step={optimizer_steps}/{total_optimizer_steps} ({total_progress:.1f}%) "
                        f"loss={window_loss:.4f} lm_loss={window_lm_loss:.4f} "
                        f"score_loss={window_score_loss:.4f}"
                    )
                accumulation_loss = 0.0
                accumulation_lm_loss = 0.0
                accumulation_score_loss = 0.0
                accumulation_microsteps = 0

        train_progress.close()
        accelerator.wait_for_everyone()

        average_loss = epoch_loss / max(epoch_steps, 1)
        average_lm_loss = epoch_lm_loss / max(epoch_steps, 1)
        average_score_loss = epoch_score_loss / max(epoch_steps, 1)

        raw_model = accelerator.unwrap_model(model)
        metrics: dict[str, float] = {}
        clear_cuda_cache()
        if accelerator.is_main_process:
            siqa_s_metrics = evaluate_siqa_s_model(
                model=raw_model,
                processor=eval_processor,
                dataloader=val_siqa_s_loader,
                device=accelerator.device,
                score_tokens=score_tokens,
                max_new_tokens=max_new_tokens,
                show_progress=True,
            )
            siqa_u_metrics = evaluate_siqa_u_model(
                model=raw_model,
                processor=eval_processor,
                dataloader=val_siqa_u_loader,
                device=accelerator.device,
                max_new_tokens=max_new_tokens,
                show_progress=True,
            )
            metrics = {
                **siqa_s_metrics,
                **siqa_u_metrics,
                "Combined_Score": compute_combined_score(siqa_s_metrics, siqa_u_metrics),
            }
        accelerator.wait_for_everyone()
        clear_cuda_cache()

        last_dir = output_dir / "last"
        if save_last_on_each_node:
            if accelerator.local_process_index == 0:
                save_checkpoint(raw_model, eval_processor, last_dir)
        elif accelerator.is_main_process:
            save_checkpoint(raw_model, eval_processor, last_dir)
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            current_metric = metrics[best_metric_name]
            if math.isfinite(current_metric) and current_metric > best_metric:
                best_metric = current_metric
                best_dir = output_dir / "best"
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                shutil.copytree(last_dir, best_dir)

            record = {
                "epoch": epoch,
                "global_step": global_step,
                "optimizer_steps": optimizer_steps,
                "train_loss": average_loss,
                "train_lm_loss": average_lm_loss,
                "train_score_loss": average_score_loss,
                **metrics,
                f"best_{best_metric_name}": best_metric,
            }
            write_history(output_dir, record)
            print(json.dumps(record, ensure_ascii=False))

        accelerator.wait_for_everyone()
        if use_resume_state:
            save_resume_checkpoint(
                accelerator=accelerator,
                processor=processor,
                resume_dir=resume_dir,
                metadata={
                    "epoch": epoch,
                    "global_step": global_step,
                    "optimizer_steps": optimizer_steps,
                    "best_metric": best_metric,
                    "config": config_record,
                },
            )
            accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
