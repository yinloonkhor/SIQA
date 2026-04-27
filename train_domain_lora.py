import json
import math
import os
import shutil
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path, PurePosixPath
from random import Random
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    get_linear_schedule_with_warmup,
    set_seed,
)


IMAGE_PLACEHOLDER = "<image>"
SECTION_MARKER_REPLACEMENTS = {
    "<|content|>": "Content",
    "<|context|>": "Context",
    "<|outline|>": "Outline",
}
PROMPT_FORMAT = "qwen_chat_v1"
TEXT_LORA_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
VISION_LORA_SUFFIXES = ("qkv", "proj")
LORA_SUPPORTED_MODULE_TYPES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
RESUME_CONFIG_KEYS = (
    "model",
    "train_jsonl",
    "val_jsonl",
    "image_root",
    "epochs",
    "lr",
    "weight_decay",
    "warmup_ratio",
    "early_stopping_patience",
    "max_length",
    "per_device_batch_size",
    "gradient_accumulation_steps",
    "gradient_clip_norm",
    "seed",
    "use_bf16",
    "mixed_precision",
    "image_min_pixels",
    "image_max_pixels",
    "data_fraction",
    "subset_seed",
    "require_images",
    "prompt_format",
    "init_adapter_dir",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "lora_bias",
    "enable_text_lora",
    "enable_vision_lora",
    "lora_text_target_modules",
    "lora_vision_target_modules",
    "lora_target_modules",
)
ROLE_ALIASES = {
    "assistant": "assistant",
    "bot": "assistant",
    "gpt": "assistant",
    "human": "user",
    "model": "assistant",
    "system": "system",
    "user": "user",
}
KNOWN_BROKEN_IMAGE_PATHS = frozenset(
    {
        "imgs/1709.01620v4/figures/VeryLate.png",
        "imgs/1902.02449v3/figures/diag10.png",
        "imgs/2001.05284v1/tables/table_2.png",
        "imgs/2003.00355v1/figures/flchain_cluster_scatter.png",
        "imgs/2009.01225v1/figures/pipeline.png",
        "imgs/2009.11050v1/figures/description_steps_cropped.png",
        "imgs/2101.09536v2/figures/tinyimnet.png",
        "imgs/2104.02323v1/figures/key_to_depth.png",
        "imgs/2106.07540v2/tables/table_1.png",
        "imgs/2112.05485v2/figures/precision_recall_f1_clipped.png",
        "imgs/2202.13013v4/figures/phi_pca_signnet11.png",
        "imgs/2203.09326v1/tables/table_3.png",
        "imgs/2203.10202v1/figures/preprocessing.png",
        "imgs/2210.03122v1/figures/air_quality-eps-converted-to.png",
        "imgs/2210.06015v2/figures/icml.png",
        "imgs/2302.06229v1/figures/patterns.png",
        "imgs/2303.09817v1/figures/model_comparison.png",
        "imgs/2306.01853v1/figures/Figure_4.png",
    }
)


def load_trusted_rgb_image(image_path: str | Path) -> Image.Image:
    # This training pipeline reads local dataset images, so allow oversized inputs
    # without triggering Pillow's decompression-bomb protection for trusted files.
    previous_max_pixels = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(image_path) as image_file:
                if image_file.mode == "P" and "transparency" in image_file.info:
                    # Pillow warns when palette transparency is stored as bytes;
                    # normalizing through RGBA avoids the warning and preserves appearance.
                    image_file = image_file.convert("RGBA")
                return image_file.convert("RGB").copy()
    finally:
        Image.MAX_IMAGE_PIXELS = previous_max_pixels


def filter_jsonl_by_image_health(
    jsonl_path: str | Path,
    image_root: str | Path,
    output_path: str | Path,
    max_image_pixels: int | None = None,
) -> str:
    if max_image_pixels is not None and max_image_pixels <= 0:
        raise ValueError(f"max_image_pixels must be positive when set, got {max_image_pixels!r}")

    jsonl_path = Path(jsonl_path)
    image_root = Path(image_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    kept_rows = 0
    dropped_rows = 0
    drop_reasons: Counter[str] = Counter()
    temporary_output_path = output_path.with_name(
        f"{output_path.stem}.pid{os.getpid()}.tmp{output_path.suffix}"
    )

    previous_max_pixels = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with open(jsonl_path, "r", encoding="utf-8") as source_handle, open(
                temporary_output_path, "w", encoding="utf-8"
            ) as target_handle:
                for line in source_handle:
                    total_rows += 1

                    if not line.strip():
                        dropped_rows += 1
                        drop_reasons["empty_line"] += 1
                        continue

                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        dropped_rows += 1
                        drop_reasons["invalid_json"] += 1
                        continue
                    if not isinstance(item, dict):
                        dropped_rows += 1
                        drop_reasons["invalid_json_object"] += 1
                        continue

                    image_paths = item.get("image") or []
                    if not isinstance(image_paths, list) or any(
                        not isinstance(image_path, str) or not image_path for image_path in image_paths
                    ):
                        dropped_rows += 1
                        drop_reasons["invalid_image_paths"] += 1
                        continue

                    keep_row = True
                    for image_path in image_paths:
                        if normalize_image_path_key(image_path) in KNOWN_BROKEN_IMAGE_PATHS:
                            keep_row = False
                            drop_reasons["known_broken_image"] += 1
                            break

                        absolute_path = Path(image_path)
                        if not absolute_path.is_absolute():
                            absolute_path = image_root / absolute_path

                        if not absolute_path.exists():
                            keep_row = False
                            drop_reasons["missing_image"] += 1
                            break

                        try:
                            with Image.open(absolute_path) as image_file:
                                width, height = image_file.size
                                pixel_count = int(width) * int(height)
                                aspect_ratio = max(
                                    float(width) / max(float(height), 1.0),
                                    float(height) / max(float(width), 1.0),
                                )
                                if max_image_pixels is not None and pixel_count > max_image_pixels:
                                    keep_row = False
                                    drop_reasons["image_too_large"] += 1
                                    break
                                if aspect_ratio >= 200.0:
                                    keep_row = False
                                    drop_reasons["image_aspect_ratio_too_large"] += 1
                                    break
                                image_file.load()
                        except Exception:
                            keep_row = False
                            drop_reasons["broken_image"] += 1
                            break

                    if not keep_row:
                        dropped_rows += 1
                        continue

                    target_handle.write(line)
                    kept_rows += 1

        os.replace(temporary_output_path, output_path)
    finally:
        Image.MAX_IMAGE_PIXELS = previous_max_pixels
        if temporary_output_path.exists():
            temporary_output_path.unlink()

    if kept_rows == 0:
        raise ValueError(
            f"Filtering removed every row from {jsonl_path}. "
            f"drop_reasons={dict(drop_reasons)!r} max_image_pixels={max_image_pixels!r}"
        )

    if int(os.environ.get("RANK", "0")) == 0:
        print(
            f"Filtered {jsonl_path} -> {output_path}: "
            f"kept={kept_rows} dropped={dropped_rows} total={total_rows} "
            f"drop_reasons={dict(drop_reasons)} max_image_pixels={max_image_pixels!r}"
        )

    return str(output_path)


def _is_text_module(module_name: str) -> bool:
    return module_name.startswith("language_model.") or ".language_model." in module_name


def _is_vision_module(module_name: str) -> bool:
    return module_name.startswith("visual.") or ".visual." in module_name


def _is_lora_supported_module(module: nn.Module) -> bool:
    return isinstance(module, LORA_SUPPORTED_MODULE_TYPES)


def resolve_lora_target_modules(
    model: Any,
    enable_text_lora: bool,
    enable_vision_lora: bool,
) -> dict[str, list[str]]:
    if not enable_text_lora and not enable_vision_lora:
        raise ValueError("At least one of enable_text_lora or enable_vision_lora must be True.")

    text_targets: list[str] = []
    vision_targets: list[str] = []

    for module_name, module in model.named_modules():
        if not module_name or not _is_lora_supported_module(module):
            continue

        if enable_text_lora and _is_text_module(module_name) and module_name.endswith(TEXT_LORA_SUFFIXES):
            text_targets.append(module_name)
            continue

        if enable_vision_lora and _is_vision_module(module_name) and module_name.endswith(VISION_LORA_SUFFIXES):
            vision_targets.append(module_name)

    text_targets = sorted(set(text_targets))
    vision_targets = sorted(set(vision_targets))
    target_modules = sorted(set(text_targets + vision_targets))

    if enable_text_lora and not text_targets:
        raise ValueError("Text LoRA is enabled, but no matching language-model target modules were found.")
    if enable_vision_lora and not vision_targets:
        raise ValueError("Vision LoRA is enabled, but no matching vision target modules were found.")
    if not target_modules:
        raise ValueError("No LoRA target modules were resolved from the loaded model.")

    return {
        "text": text_targets,
        "vision": vision_targets,
        "all": target_modules,
    }


def summarize_parameter_counts(model: Any) -> tuple[int, int]:
    total_parameters = 0
    trainable_parameters = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total_parameters += count
        if parameter.requires_grad:
            trainable_parameters += count
    return total_parameters, trainable_parameters


def log_lora_setup(
    accelerator: Accelerator,
    resolved_targets: dict[str, list[str]],
    total_parameters: int,
    trainable_parameters: int,
) -> None:
    if not accelerator.is_main_process:
        return

    print("Resolved LoRA target modules:")
    print(f"  Text ({len(resolved_targets['text'])})")
    print(f"  Vision ({len(resolved_targets['vision'])})")

    trainable_ratio = 0.0
    if total_parameters > 0:
        trainable_ratio = (trainable_parameters / total_parameters) * 100.0
    print(
        f"Trainable parameters: {trainable_parameters:,} / {total_parameters:,} "
        f"({trainable_ratio:.4f}%)"
    )


def configure_image_processor(processor: Any, min_pixels: int, max_pixels: int) -> None:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None or not hasattr(image_processor, "size"):
        raise ValueError("Loaded processor does not expose image_processor.size for pixel limits.")
    if min_pixels <= 0 or max_pixels <= 0:
        raise ValueError("Image pixel limits must be positive integers.")
    if min_pixels > max_pixels:
        raise ValueError("image_min_pixels cannot be greater than image_max_pixels.")

    image_processor.size = {
        "shortest_edge": int(min_pixels),
        "longest_edge": int(max_pixels),
    }


def load_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_resume_metadata(resume_dir: Path) -> dict[str, Any] | None:
    metadata_path = resume_dir / "trainer_state.json"
    return load_json_file(metadata_path)


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


def save_resume_checkpoint(
    accelerator: Accelerator,
    processor: Any,
    resume_dir: Path,
    metadata: dict[str, Any],
) -> None:
    resume_dir.mkdir(parents=True, exist_ok=True)
    accelerator.save_state(str(resume_dir))
    if accelerator.is_main_process:
        processor.save_pretrained(resume_dir)
        with open(resume_dir / "trainer_state.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)


def save_checkpoint(model: Any, processor: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)


def write_history(output_dir: Path, record: dict[str, Any]) -> None:
    history_path = output_dir / "history.jsonl"
    with open(history_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_multi_node_run(accelerator: Accelerator) -> bool:
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    return accelerator.num_processes > local_world_size


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def normalize_role(raw_role: Any) -> str | None:
    if not isinstance(raw_role, str):
        return None
    return ROLE_ALIASES.get(raw_role.strip().lower())


def normalize_image_path_key(image_path: str) -> str:
    return str(PurePosixPath(image_path.replace("\\", "/")))


def validate_init_adapter_config(
    init_adapter_dir: Path,
    model_name: str,
    resolved_targets: dict[str, list[str]],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_bias: str,
) -> None:
    adapter_config = load_json_file(init_adapter_dir / "adapter_config.json")
    if adapter_config is None:
        raise ValueError(f"init_adapter_dir does not contain adapter_config.json: {init_adapter_dir}")

    saved_model_name = adapter_config.get("base_model_name_or_path")
    if saved_model_name != model_name:
        raise ValueError(
            "The init adapter was trained for a different base model.\n"
            f"init_adapter_dir: {init_adapter_dir}\n"
            f"saved base model: {saved_model_name!r}\n"
            f"current base model: {model_name!r}"
        )

    saved_target_suffixes = {
        str(module_name).split(".")[-1]
        for module_name in (adapter_config.get("target_modules") or [])
    }
    expected_target_suffixes = {
        module_name.split(".")[-1]
        for module_name in resolved_targets["all"]
    }
    if saved_target_suffixes != expected_target_suffixes:
        raise ValueError(
            "The init adapter target-module layout does not match the current LoRA setup.\n"
            f"init_adapter_dir: {init_adapter_dir}\n"
            f"saved target suffixes: {sorted(saved_target_suffixes)!r}\n"
            f"expected target suffixes: {sorted(expected_target_suffixes)!r}"
        )

    saved_r = int(adapter_config.get("r"))
    saved_alpha = int(adapter_config.get("lora_alpha"))
    saved_dropout = float(adapter_config.get("lora_dropout"))
    saved_bias = str(adapter_config.get("bias"))
    if saved_r != lora_r or saved_alpha != lora_alpha or saved_dropout != lora_dropout or saved_bias != lora_bias:
        raise ValueError(
            "The init adapter LoRA hyperparameters do not match the current run config.\n"
            f"init_adapter_dir: {init_adapter_dir}\n"
            f"saved: r={saved_r} alpha={saved_alpha} dropout={saved_dropout} bias={saved_bias!r}\n"
            f"current: r={lora_r} alpha={lora_alpha} dropout={lora_dropout} bias={lora_bias!r}"
        )


@dataclass
class DomainSftExample:
    example_id: str
    task_type: str
    image_paths: list[str]
    system_instruction: str | list[str]
    prompt_turns: list[tuple[str, str]]
    target_text: str


class MPaperSftDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        image_root: str,
        split_name: str,
        data_fraction: float,
        subset_seed: int,
        require_images: bool = True,
        apply_fraction: bool = True,
    ) -> None:
        if not (0.0 < data_fraction <= 1.0):
            raise ValueError(f"data_fraction must be in (0, 1], got {data_fraction!r}")

        self.jsonl_path = str(jsonl_path)
        self.image_root = Path(image_root)
        self.split_name = split_name
        self.data_fraction = float(data_fraction)
        self.subset_seed = int(subset_seed)
        self.require_images = bool(require_images)
        self.apply_fraction = bool(apply_fraction)

        self._file_handle = None
        self._file_handle_pid = None
        self.total_rows = 0
        self.skip_reasons: Counter[str] = Counter()
        self.eligible_counts_by_task: dict[str, int] = {}
        self.selected_counts_by_task: dict[str, int] = {}
        self.selected_offsets: list[int] = []

        self._index_dataset()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_file_handle"] = None
        state["_file_handle_pid"] = None
        return state

    def __del__(self) -> None:
        file_handle = getattr(self, "_file_handle", None)
        if file_handle is not None:
            file_handle.close()

    def _get_file_handle(self):
        current_pid = os.getpid()
        if self._file_handle is None or self._file_handle_pid != current_pid:
            if self._file_handle is not None:
                self._file_handle.close()
            self._file_handle = open(self.jsonl_path, "r", encoding="utf-8")
            self._file_handle_pid = current_pid
        return self._file_handle

    def _load_item_at_offset(self, offset: int) -> dict[str, Any]:
        handle = self._get_file_handle()
        handle.seek(offset)
        line = handle.readline()
        if not line:
            raise IndexError(f"Failed to read JSONL row at offset {offset} from {self.jsonl_path}")
        return json.loads(line)

    def _index_dataset(self) -> None:
        eligible_offsets_by_task: dict[str, list[int]] = defaultdict(list)

        with open(self.jsonl_path, "r", encoding="utf-8") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break

                self.total_rows += 1
                if not line.strip():
                    self.skip_reasons["empty_line"] += 1
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    self.skip_reasons["invalid_json"] += 1
                    continue

                is_valid, task_type, reason = self._inspect_item(item)
                if not is_valid:
                    self.skip_reasons[reason] += 1
                    continue

                eligible_offsets_by_task[task_type].append(offset)

        self.eligible_counts_by_task = {
            task_type: len(offsets)
            for task_type, offsets in sorted(eligible_offsets_by_task.items())
        }
        self.selected_offsets, self.selected_counts_by_task = self._select_offsets(eligible_offsets_by_task)

        if not self.selected_offsets:
            raise ValueError(
                f"No usable examples were selected from {self.jsonl_path}. "
                f"eligible_counts_by_task={self.eligible_counts_by_task!r} "
                f"skip_reasons={dict(self.skip_reasons)!r}"
            )

    def _inspect_item(self, item: dict[str, Any]) -> tuple[bool, str, str]:
        task_type = str(item.get("task_type") or "unknown_task")
        image_paths = item.get("image") or []

        if self.require_images and not image_paths:
            return False, task_type, "no_images"

        if not isinstance(image_paths, list) or any(not isinstance(path, str) or not path for path in image_paths):
            return False, task_type, "invalid_image_paths"
        if any(normalize_image_path_key(path) in KNOWN_BROKEN_IMAGE_PATHS for path in image_paths):
            return False, task_type, "known_broken_image"

        conversations = item.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            return False, task_type, "invalid_conversations"

        normalized_roles: list[str] = []
        for message in conversations:
            if not isinstance(message, dict):
                return False, task_type, "invalid_conversations"
            role = normalize_role(message.get("from"))
            if role is None:
                return False, task_type, "unknown_role"
            value = message.get("value")
            if not isinstance(value, str):
                return False, task_type, "invalid_message_value"
            normalized_roles.append(role)

        if normalized_roles[-1] != "assistant":
            return False, task_type, "missing_final_assistant"

        target_text = conversations[-1]["value"]
        if not target_text.strip():
            return False, task_type, "empty_assistant_target"
        if IMAGE_PLACEHOLDER in target_text:
            return False, task_type, "assistant_contains_image_placeholder"

        prompt_placeholder_count = sum(
            str(message.get("value", "")).count(IMAGE_PLACEHOLDER)
            for message in conversations[:-1]
        )
        if self.require_images and prompt_placeholder_count != len(image_paths):
            return False, task_type, "image_placeholder_mismatch"

        return True, task_type, ""

    def _select_offsets(
        self,
        eligible_offsets_by_task: dict[str, list[int]],
    ) -> tuple[list[int], dict[str, int]]:
        selected_offsets: list[int] = []
        selected_counts_by_task: dict[str, int] = {}

        for task_type in sorted(eligible_offsets_by_task):
            offsets = list(eligible_offsets_by_task[task_type])
            if not offsets:
                selected_counts_by_task[task_type] = 0
                continue

            if not self.apply_fraction or self.data_fraction >= 1.0:
                chosen_offsets = offsets
            else:
                sample_count = max(1, int(math.ceil(len(offsets) * self.data_fraction)))
                sample_count = min(sample_count, len(offsets))
                task_rng = Random(f"{self.subset_seed}:{task_type}")
                shuffled_offsets = offsets[:]
                task_rng.shuffle(shuffled_offsets)
                chosen_offsets = sorted(shuffled_offsets[:sample_count])

            selected_offsets.extend(chosen_offsets)
            selected_counts_by_task[task_type] = len(chosen_offsets)

        return sorted(selected_offsets), selected_counts_by_task

    def assert_selected_images_exist(self, max_missing_paths: int = 10) -> None:
        missing_paths: list[str] = []

        with open(self.jsonl_path, "r", encoding="utf-8") as handle:
            for offset in self.selected_offsets:
                handle.seek(offset)
                line = handle.readline()
                if not line:
                    continue
                item = json.loads(line)
                for image_path in item.get("image") or []:
                    absolute_path = Path(image_path)
                    if not absolute_path.is_absolute():
                        absolute_path = self.image_root / absolute_path
                    if absolute_path.exists():
                        continue
                    missing_paths.append(str(absolute_path))
                    if len(missing_paths) >= max_missing_paths:
                        break
                if missing_paths:
                    break

        if missing_paths:
            raise FileNotFoundError(
                f"Selected {self.split_name} examples reference images that are missing under image_root={self.image_root}.\n"
                f"First missing paths:\n" + "\n".join(missing_paths)
            )

    def _item_to_example(self, item: dict[str, Any]) -> DomainSftExample:
        conversations = item["conversations"]
        prompt_turns: list[tuple[str, str]] = []

        for message in conversations[:-1]:
            role = normalize_role(message.get("from"))
            if role is None:
                raise ValueError(f"Unsupported conversation role: {message.get('from')!r}")
            text = str(message.get("value", ""))
            prompt_turns.append((role, text))

        image_paths: list[str] = []
        for image_path in item.get("image") or []:
            absolute_path = Path(image_path)
            if not absolute_path.is_absolute():
                absolute_path = self.image_root / absolute_path
            image_paths.append(str(absolute_path))

        return DomainSftExample(
            example_id=str(item.get("id") or ""),
            task_type=str(item.get("task_type") or "unknown_task"),
            image_paths=image_paths,
            system_instruction=item.get("system_instruction") or "",
            prompt_turns=prompt_turns,
            target_text=str(conversations[-1]["value"]).strip(),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "split": self.split_name,
            "rows": self.total_rows,
            "selected": len(self.selected_offsets),
            "eligible_counts_by_task": self.eligible_counts_by_task,
            "selected_counts_by_task": self.selected_counts_by_task,
            "skip_reasons": dict(self.skip_reasons),
        }

    def __len__(self) -> int:
        return len(self.selected_offsets)

    def __getitem__(self, index: int) -> DomainSftExample:
        item = self._load_item_at_offset(self.selected_offsets[index])
        return self._item_to_example(item)


def resolve_system_instruction(system_instruction: str | list[str]) -> str:
    if isinstance(system_instruction, str):
        return system_instruction.strip()
    if isinstance(system_instruction, list):
        for candidate in system_instruction:
            text = str(candidate).strip()
            if text:
                return text
    return ""


def normalize_section_markers(text: str) -> str:
    normalized_text = str(text)
    for marker, label in SECTION_MARKER_REPLACEMENTS.items():
        normalized_text = normalized_text.replace(f"{marker}:", f"{label}:")
        normalized_text = normalized_text.replace(marker, label)
    return normalized_text


def _join_message_text(existing_text: str, incoming_text: str) -> str:
    existing_text = str(existing_text)
    incoming_text = str(incoming_text)
    if not existing_text:
        return incoming_text
    if not incoming_text:
        return existing_text
    return f"{existing_text.rstrip()}\n{incoming_text.lstrip()}"


def _inline_text_to_qwen_content(
    text: str,
    *,
    image_count: int,
    image_cursor: int,
) -> tuple[list[dict[str, Any]], int]:
    normalized_text = normalize_section_markers(text)
    segments = normalized_text.split(IMAGE_PLACEHOLDER)
    content: list[dict[str, Any]] = []

    for segment_index, segment_text in enumerate(segments):
        if segment_text:
            content.append({"type": "text", "text": segment_text})
        if segment_index >= len(segments) - 1:
            continue
        if image_cursor >= image_count:
            raise ValueError(
                f"Encountered more {IMAGE_PLACEHOLDER!r} placeholders than images while building Qwen chat content."
            )
        content.append({"type": "image"})
        image_cursor += 1

    return content, image_cursor


def build_qwen_chat_messages(
    example: DomainSftExample,
    *,
    include_target: bool,
) -> list[dict[str, Any]]:
    prompt_messages: list[tuple[str, str]] = []
    system_instruction = resolve_system_instruction(example.system_instruction)
    if system_instruction:
        prompt_messages.append(("system", system_instruction))

    for role, text in example.prompt_turns:
        if prompt_messages and prompt_messages[-1][0] == role:
            merged_text = _join_message_text(prompt_messages[-1][1], text)
            prompt_messages[-1] = (role, merged_text)
            continue
        prompt_messages.append((role, str(text)))

    image_cursor = 0
    qwen_messages: list[dict[str, Any]] = []
    for role, text in prompt_messages:
        content, image_cursor = _inline_text_to_qwen_content(
            text,
            image_count=len(example.image_paths),
            image_cursor=image_cursor,
        )
        if not content:
            continue
        qwen_messages.append({"role": role, "content": content})

    if image_cursor != len(example.image_paths):
        raise ValueError(
            f"Expected {len(example.image_paths)} images but only used {image_cursor} placeholders while building chat messages."
        )

    if include_target:
        target_content, target_image_cursor = _inline_text_to_qwen_content(
            example.target_text.rstrip(),
            image_count=len(example.image_paths),
            image_cursor=image_cursor,
        )
        if target_image_cursor != image_cursor:
            raise ValueError("Assistant target text must not introduce additional image placeholders.")
        qwen_messages.append({"role": "assistant", "content": target_content or [{"type": "text", "text": ""}]})

    return qwen_messages


def build_sft_transcripts(example: DomainSftExample, processor: Any) -> tuple[str, str]:
    if not hasattr(processor, "apply_chat_template"):
        raise ValueError("Loaded processor does not expose apply_chat_template required for native Qwen chat formatting.")

    prompt_messages = build_qwen_chat_messages(example, include_target=False)
    full_messages = build_qwen_chat_messages(example, include_target=True)
    prompt_text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return prompt_text, full_text


class DomainSftCollator:
    def __init__(self, processor: Any, max_length: int) -> None:
        self.processor = processor
        self.max_length = int(max_length)
        self.pad_token_id = int(processor.tokenizer.pad_token_id)
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None or not hasattr(image_processor, "size"):
            raise ValueError("Loaded processor does not expose image_processor.size for multimodal truncation control.")
        self.image_processor = image_processor
        self.default_image_size = {
            "shortest_edge": int(image_processor.size["shortest_edge"]),
            "longest_edge": int(image_processor.size["longest_edge"]),
        }
        self.merge_size = int(getattr(image_processor, "merge_size", 2))
        self.image_token = str(getattr(processor, "image_token", "<|image_pad|>"))

    @staticmethod
    def _shared_prefix_length(prompt_ids: torch.Tensor, full_ids: torch.Tensor) -> int:
        shared_length = 0
        max_compare = min(int(prompt_ids.numel()), int(full_ids.numel()))
        while shared_length < max_compare and int(prompt_ids[shared_length]) == int(full_ids[shared_length]):
            shared_length += 1
        return shared_length

    def _set_image_budget(self, max_pixels: int) -> None:
        self.image_processor.size = {
            "shortest_edge": int(self.default_image_size["shortest_edge"]),
            "longest_edge": int(max_pixels),
        }

    def _iter_image_budget_candidates(self) -> list[int]:
        pixels_per_token = 28 * 28
        min_token_budget = max(1, int(math.ceil(self.default_image_size["shortest_edge"] / pixels_per_token)))
        default_token_budget = max(min_token_budget, int(self.default_image_size["longest_edge"] // pixels_per_token))
        token_candidates = [
            default_token_budget,
            768,
            640,
            512,
            384,
            256,
            192,
            128,
            96,
            min_token_budget,
        ]

        unique_candidates: list[int] = []
        seen: set[int] = set()
        for token_budget in token_candidates:
            bounded_budget = max(min_token_budget, min(default_token_budget, int(token_budget)))
            candidate = pixels_per_token * bounded_budget
            if candidate in seen:
                continue
            unique_candidates.append(candidate)
            seen.add(candidate)
        return unique_candidates

    def _expand_image_tokens(self, text: str, image_grid_thw: torch.Tensor) -> str:
        expanded_text = str(text)
        placeholder_index = 0
        merge_length = self.merge_size**2
        while self.image_token in expanded_text:
            if placeholder_index >= int(image_grid_thw.size(0)):
                raise ValueError("More image placeholders were found in text than processed images in the sample.")
            num_image_tokens = int(image_grid_thw[placeholder_index].prod().item()) // merge_length
            expanded_text = expanded_text.replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
            placeholder_index += 1
        return expanded_text.replace("<|placeholder|>", self.image_token)

    def _encode_example(
        self,
        prompt_text: str,
        full_text: str,
        loaded_images: list[Image.Image],
    ) -> dict[str, torch.Tensor] | None:
        # Qwen3-VL cannot safely use tokenizer-side truncation because it expands
        # image placeholders into repeated image tokens before tokenization.
        for max_pixels in self._iter_image_budget_candidates():
            self._set_image_budget(max_pixels)
            full_inputs = self.processor(
                text=[full_text],
                images=[loaded_images],
                return_tensors="pt",
                padding=False,
                truncation=False,
            )

            input_ids = full_inputs["input_ids"][0]
            attention_mask = full_inputs["attention_mask"][0]
            sequence_end = int(attention_mask.sum().item())
            if sequence_end > self.max_length:
                continue

            expanded_prompt_text = self._expand_image_tokens(prompt_text, full_inputs["image_grid_thw"])
            prompt_ids = self.processor.tokenizer(
                expanded_prompt_text,
                return_tensors="pt",
                padding=False,
                truncation=False,
            )["input_ids"][0]
            prompt_length = self._shared_prefix_length(prompt_ids, input_ids[:sequence_end])
            if prompt_length >= sequence_end:
                continue

            labels = input_ids[:sequence_end].clone()
            labels[:prompt_length] = -100
            return {
                "input_ids": input_ids[:sequence_end],
                "attention_mask": attention_mask[:sequence_end],
                "mm_token_type_ids": full_inputs["mm_token_type_ids"][0][:sequence_end],
                "labels": labels,
                "pixel_values": full_inputs["pixel_values"],
                "image_grid_thw": full_inputs["image_grid_thw"],
            }
        return None

    def __call__(self, batch: list[DomainSftExample]) -> dict[str, Any]:
        encoded_examples: list[dict[str, torch.Tensor]] = []

        try:
            for example in batch:
                loaded_images: list[Image.Image] = []
                for image_path in example.image_paths:
                    loaded_images.append(load_trusted_rgb_image(image_path))

                prompt_text, full_text = build_sft_transcripts(example, self.processor)
                encoded_example = self._encode_example(prompt_text, full_text, loaded_images)
                if encoded_example is None:
                    continue
                encoded_examples.append(encoded_example)
        finally:
            self._set_image_budget(self.default_image_size["longest_edge"])

        if not encoded_examples:
            raise ValueError(
                "No examples in the batch could fit within the configured multimodal token budget. "
                "Reduce image_max_pixels, reduce prompt length, or increase max_length."
            )

        return {
            "input_ids": pad_sequence(
                [example["input_ids"] for example in encoded_examples],
                batch_first=True,
                padding_value=self.pad_token_id,
            ),
            "attention_mask": pad_sequence(
                [example["attention_mask"] for example in encoded_examples],
                batch_first=True,
                padding_value=0,
            ),
            "mm_token_type_ids": pad_sequence(
                [example["mm_token_type_ids"] for example in encoded_examples],
                batch_first=True,
                padding_value=0,
            ),
            "labels": pad_sequence(
                [example["labels"] for example in encoded_examples],
                batch_first=True,
                padding_value=-100,
            ),
            "pixel_values": torch.cat([example["pixel_values"] for example in encoded_examples], dim=0),
            "image_grid_thw": torch.cat([example["image_grid_thw"] for example in encoded_examples], dim=0),
        }


def evaluate_model_loss(
    model: Any,
    dataloader: DataLoader,
    device: torch.device,
    show_progress: bool = False,
) -> float:
    model.eval()
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        progress_bar = tqdm(
            dataloader,
            desc="Validation",
            leave=True,
            disable=not show_progress,
        )
        for batch in progress_bar:
            batch = move_batch_to_device(batch, device)
            model_inputs = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
            outputs = model(**model_inputs)
            total_loss += float(outputs.loss.detach().float().cpu().item())
            total_steps += 1

    return total_loss / max(total_steps, 1)


def print_dataset_summary(split_name: str, dataset: MPaperSftDataset) -> None:
    summary = dataset.summary()
    print(
        f"{split_name} rows={summary['rows']} selected={summary['selected']} "
        f"eligible_by_task={summary['eligible_counts_by_task']}"
    )
    print(f"{split_name} selected_by_task={summary['selected_counts_by_task']}")
    if summary["skip_reasons"]:
        print(f"{split_name} skipped={summary['skip_reasons']}")


def main() -> None:
    model_name = "Qwen/Qwen3-VL-2B-Instruct"
    train_jsonl = "M-Paper/sft/3tasks_train.jsonl"
    val_jsonl = "M-Paper/sft/3tasks_val.jsonl"
    image_root = "M-Paper"
    output_dir = Path("checkpoints/qwen3-vl-2b-domain-lora")
    resume_dir = output_dir / "resume_state"
    init_adapter_dir = None
    epochs = 3
    lr = 1e-4
    weight_decay = 0.01
    warmup_ratio = 0.05
    early_stopping_patience: int | None = None
    max_length = 2048
    per_device_batch_size = 2
    gradient_accumulation_steps = 8
    gradient_clip_norm = 1.0
    seed = 42
    num_workers = 1
    use_bf16 = True
    image_min_pixels = 256 * 256
    # Keep multi-image PaperOwl-style prompts within a 2048-token text budget for Qwen3-VL.
    image_max_pixels = 28 * 28 * 896
    dataset_filter_max_image_pixels = image_max_pixels
    distributed_timeout = timedelta(hours=2)
    log_every = 100
    data_fraction = 0.4
    subset_seed = 42
    require_images = True
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    lora_bias = "none"
    enable_text_lora = True
    enable_vision_lora = True

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

    if accelerator.is_main_process:
        print(
            "Configured processor image limits: "
            f"min_pixels={image_min_pixels} max_pixels={image_max_pixels}"
        )

    train_jsonl_path = Path(train_jsonl)
    val_jsonl_path = Path(val_jsonl)
    max_pixels_tag = "none" if dataset_filter_max_image_pixels is None else str(dataset_filter_max_image_pixels)
    filtered_train_jsonl = train_jsonl_path.with_name(
        f"{train_jsonl_path.stem}.filtered_maxpixels_{max_pixels_tag}{train_jsonl_path.suffix}"
    )
    filtered_val_jsonl = val_jsonl_path.with_name(
        f"{val_jsonl_path.stem}.filtered_maxpixels_{max_pixels_tag}{val_jsonl_path.suffix}"
    )
    should_filter_on_this_process = accelerator.is_main_process
    if multi_node_run and not shared_storage_available:
        should_filter_on_this_process = accelerator.is_local_main_process

    if should_filter_on_this_process:
        filter_jsonl_by_image_health(
            jsonl_path=train_jsonl_path,
            image_root=image_root,
            output_path=filtered_train_jsonl,
            max_image_pixels=dataset_filter_max_image_pixels,
        )
        filter_jsonl_by_image_health(
            jsonl_path=val_jsonl_path,
            image_root=image_root,
            output_path=filtered_val_jsonl,
            max_image_pixels=dataset_filter_max_image_pixels,
        )

    accelerator.wait_for_everyone()

    if not filtered_train_jsonl.exists() or not filtered_val_jsonl.exists():
        raise FileNotFoundError(
            "Filtered JSONL files were not created before dataset initialization.\n"
            f"train_jsonl={filtered_train_jsonl}\n"
            f"val_jsonl={filtered_val_jsonl}"
        )

    train_jsonl = str(filtered_train_jsonl)
    val_jsonl = str(filtered_val_jsonl)

    train_dataset = MPaperSftDataset(
        jsonl_path=train_jsonl,
        image_root=image_root,
        split_name="train",
        data_fraction=data_fraction,
        subset_seed=subset_seed,
        require_images=require_images,
        apply_fraction=True,
    )
    val_dataset = MPaperSftDataset(
        jsonl_path=val_jsonl,
        image_root=image_root,
        split_name="val",
        data_fraction=1.0,
        subset_seed=subset_seed,
        require_images=require_images,
        apply_fraction=False,
    )

    if accelerator.is_main_process:
        print_dataset_summary("Train", train_dataset)
        print_dataset_summary("Val", val_dataset)

    train_dataset.assert_selected_images_exist()
    val_dataset.assert_selected_images_exist()

    train_collator = DomainSftCollator(processor, max_length)
    val_collator = DomainSftCollator(processor, max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_device_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=train_collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=per_device_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=val_collator,
    )

    load_dtype = torch.bfloat16 if mixed_precision == "bf16" else None
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=load_dtype,
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
    if init_adapter_path is not None:
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
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=lora_bias,
                target_modules=resolved_targets["all"],
            ),
        )

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
        "train_jsonl": train_jsonl,
        "val_jsonl": val_jsonl,
        "image_root": image_root,
        "output_dir": str(output_dir),
        "resume_dir": str(resume_dir),
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "early_stopping_patience": early_stopping_patience,
        "max_length": max_length,
        "per_device_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clip_norm": gradient_clip_norm,
        "seed": seed,
        "num_workers": num_workers,
        "use_bf16": use_bf16,
        "image_min_pixels": image_min_pixels,
        "image_max_pixels": image_max_pixels,
        "log_every": log_every,
        "mixed_precision": mixed_precision,
        "data_fraction": data_fraction,
        "subset_seed": subset_seed,
        "require_images": require_images,
        "prompt_format": PROMPT_FORMAT,
        "init_adapter_dir": str(init_adapter_path) if init_adapter_path is not None else None,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_bias": lora_bias,
        "enable_text_lora": enable_text_lora,
        "enable_vision_lora": enable_vision_lora,
        "lora_text_target_modules": resolved_targets["text"],
        "lora_vision_target_modules": resolved_targets["vision"],
        "lora_target_modules": resolved_targets["all"],
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

    best_metric = float("inf")
    epochs_without_improvement = 0
    global_step = 0
    optimizer_steps = 0
    start_epoch = 1

    if resume_metadata is not None:
        accelerator.load_state(str(resume_dir))
        start_epoch = int(resume_metadata.get("epoch", 0)) + 1
        global_step = int(resume_metadata.get("global_step", 0))
        optimizer_steps = int(resume_metadata.get("optimizer_steps", 0))
        best_metric = float(resume_metadata.get("best_metric", float("inf")))
        epochs_without_improvement = int(resume_metadata.get("epochs_without_improvement", 0))
        if accelerator.is_main_process:
            print(
                f"Resuming training from {resume_dir} at epoch={start_epoch} "
                f"global_step={global_step} optimizer_steps={optimizer_steps}"
            )

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        accumulation_loss = 0.0
        accumulation_microsteps = 0
        epoch_optimizer_steps = 0
        train_progress = tqdm(
            total=optimizer_steps_per_epoch,
            desc=f"Epoch {epoch}/{epochs}",
            disable=not accelerator.is_main_process,
            leave=True,
        )

        for step, batch in enumerate(train_loader, start=1):
            model_inputs = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}

            with accelerator.accumulate(model):
                outputs = model(**model_inputs)
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_parameters, gradient_clip_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            reduced_loss = accelerator.gather(loss.detach().unsqueeze(0)).mean().item()
            epoch_loss += reduced_loss
            epoch_steps += 1
            global_step += 1
            accumulation_loss += reduced_loss
            accumulation_microsteps += 1

            if accelerator.sync_gradients:
                optimizer_steps += 1
                epoch_optimizer_steps += 1
                window_loss = accumulation_loss / max(accumulation_microsteps, 1)
                if accelerator.is_main_process:
                    train_progress.update(1)
                    train_progress.set_postfix(
                        loss=f"{window_loss:.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    )
                if accelerator.is_main_process and optimizer_steps % log_every == 0:
                    batch_progress = 100.0 * step / max(microsteps_per_epoch, 1)
                    epoch_progress = 100.0 * epoch_optimizer_steps / max(optimizer_steps_per_epoch, 1)
                    total_progress = 100.0 * optimizer_steps / max(total_optimizer_steps, 1)
                    train_progress.write(
                        f"epoch={epoch}/{epochs} "
                        f"batch_step={step}/{microsteps_per_epoch} ({batch_progress:.1f}%) "
                        f"epoch_optimizer_step={epoch_optimizer_steps}/{optimizer_steps_per_epoch} ({epoch_progress:.1f}%) "
                        f"optimizer_step={optimizer_steps}/{total_optimizer_steps} ({total_progress:.1f}%) "
                        f"loss={window_loss:.4f}"
                    )
                accumulation_loss = 0.0
                accumulation_microsteps = 0

        train_progress.close()
        accelerator.wait_for_everyone()

        average_train_loss = epoch_loss / max(epoch_steps, 1)

        raw_model = accelerator.unwrap_model(model)
        val_loss = float("nan")
        if accelerator.is_main_process:
            val_loss = evaluate_model_loss(
                model=raw_model,
                dataloader=val_loader,
                device=accelerator.device,
                show_progress=True,
            )
        accelerator.wait_for_everyone()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        last_dir = output_dir / "last"
        if save_last_on_each_node:
            if accelerator.local_process_index == 0:
                save_checkpoint(raw_model, processor, last_dir)
        elif accelerator.is_main_process:
            save_checkpoint(raw_model, processor, last_dir)
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            improved = math.isfinite(val_loss) and val_loss < best_metric
            if improved:
                best_metric = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if improved:
                best_dir = output_dir / "best"
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                shutil.copytree(last_dir, best_dir)

            record = {
                "epoch": epoch,
                "global_step": global_step,
                "optimizer_steps": optimizer_steps,
                "train_loss": average_train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_metric,
                "epochs_without_improvement": epochs_without_improvement,
            }
            write_history(output_dir, record)
            print(json.dumps(record, ensure_ascii=False))

        accelerator.wait_for_everyone()
        should_stop = (
            accelerator.is_main_process
            and early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        )
        stop_signal = torch.tensor(int(should_stop), device=accelerator.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stop_signal, op=dist.ReduceOp.MAX)

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
                    "epochs_without_improvement": epochs_without_improvement,
                    "config": config_record,
                },
            )
            accelerator.wait_for_everyone()

        if int(stop_signal.item()) > 0:
            if accelerator.is_main_process:
                print(
                    f"Early stopping triggered after {epochs_without_improvement} "
                    f"consecutive non-improving validation epoch(s)."
                )
            break


if __name__ == "__main__":
    main()
