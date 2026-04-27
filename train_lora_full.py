import gc
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler
from tqdm.auto import tqdm
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    get_linear_schedule_with_warmup,
    set_seed,
)


QUALITY_WORDS = ["Bad", "Poor", "Fair", "Good", "Excellent"]
QUALITY_VALUES = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
QUALITY_WORD_TO_VALUE = {word.lower(): idx + 1.0 for idx, word in enumerate(QUALITY_WORDS)}
QUALITY_PATTERN = re.compile(r"\b(bad|poor|fair|good|excellent)\b", flags=re.IGNORECASE)
VALID_ANSWER_LETTERS = frozenset({"A", "B", "C", "D"})
VALID_SIQA_U_TYPES = frozenset({"yes-or-no", "what", "how"})
SIQA_S_FAMILY = "siqa_s"
SIQA_U_FAMILY = "siqa_u"
PERCEPTION_TASK_ID = 0
KNOWLEDGE_TASK_ID = 1
TEXT_LORA_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
VISION_LORA_SUFFIXES = ("qkv", "proj")
LORA_SUPPORTED_MODULE_TYPES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
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


def float_to_rating_word(score: float) -> str:
    score = max(1.0, min(5.0, float(score)))
    # This only builds the teacher-forced answer text. The actual SIQA-S score supervision uses
    # the score-token logits -> softmax -> expected value path, matching BaseModel.ScoreSoftHead.
    if score < 1.5:
        return "Bad"
    if score < 2.5:
        return "Poor"
    if score < 3.5:
        return "Fair"
    if score < 4.5:
        return "Good"
    return "Excellent"


def parse_quality_word(text: str) -> float | None:
    match = QUALITY_PATTERN.search(text or "")
    if not match:
        return None
    return QUALITY_WORD_TO_VALUE[match.group(1).lower()]


def normalize_answer_letter(text: str) -> str:
    raw_text = str(text or "")
    if not raw_text:
        return ""
    candidate = raw_text[:1].strip().upper()
    return candidate if candidate in VALID_ANSWER_LETTERS else ""


def normalize_gold_answer_letter(text: str) -> str:
    candidate = str(text or "").strip().upper()
    return candidate if candidate in VALID_ANSWER_LETTERS else ""


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
    print(f"  Text ({len(resolved_targets['text'])}):")
    print(f"  Vision ({len(resolved_targets['vision'])}):")

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


def load_rgb_image_with_pixel_cap(image_path: str, max_pixels: int | None) -> Image.Image:
    with Image.open(image_path) as image_file:
        if max_pixels is not None:
            width, height = image_file.size
            pixel_count = width * height
            if pixel_count > max_pixels:
                scale = math.sqrt(max_pixels / float(pixel_count))
                target_size = (
                    max(1, int(width * scale)),
                    max(1, int(height * scale)),
                )
                try:
                    image_file.draft("RGB", target_size)
                except Exception:
                    pass

        image = image_file.convert("RGB")

    if max_pixels is not None:
        width, height = image.size
        pixel_count = width * height
        if pixel_count > max_pixels:
            scale = math.sqrt(max_pixels / float(pixel_count))
            target_size = (
                max(1, int(width * scale)),
                max(1, int(height * scale)),
            )
            image.thumbnail(target_size, Image.Resampling.LANCZOS)

    return image


def build_siqa_s_prompt(score_task: str) -> tuple[str, str]:
    if score_task == "perception":
        system_prompt = """You are an expert in scientific image analysis. Evaluate the given image on **Subjective Quality** only:

- Consider technical quality (sharpness, lighting, legibility) and aesthetic quality (visual appeal, layout balance, information density).
- Ignore scientific correctness.

Use exactly one of these five terms: [Bad, Poor, Fair, Good, Excellent]. Respond ONLY as:

Subjective: [Quality Word]
"""
        question = "How would you rate the subjective quality of this image?"
    elif score_task == "knowledge":
        system_prompt = """You are an expert in scientific image analysis. Evaluate the given image on **Objective Quality** only:

- Assess scientific rigor: completeness (e.g., scale bars, axis labels, units), correctness of data, and avoidance of redundancy.
- Ignore aesthetics or technical rendering.

Use exactly one of these five terms: [Bad, Poor, Fair, Good, Excellent]. Respond ONLY as:

Objective: [Quality Word]
"""
        question = "How would you rate the objective quality of this image?"
    else:
        raise ValueError(f"Unknown SIQA-S task: {score_task}")
    return system_prompt, question


def build_siqa_u_prompt(question: str, option: str) -> tuple[str, str]:
    system_prompt = (
        "You are an expert in scientific image analysis. Your task is to answer visual question "
        "answering (VQA) questions based on the given image. Respond with ONLY a single uppercase "
        "letter: A, B, C, or D. Do not include any explanations, punctuation, spaces, or "
        "additional characters."
    )
    user_prompt = (
        "<image>Answer the following question based on the image.\n"
        f"Question: {question}\n"
        f"Choices: {option}\n"
        "Respond with ONLY one uppercase letter: A, B, C, or D."
    )
    return system_prompt, user_prompt


@dataclass
class MixedSiqaExample:
    family: str
    image_path: str
    target_text: str
    score_task: str | None = None
    target_score: float | None = None
    question: str | None = None
    option: str | None = None
    answer: str | None = None
    uqa_type: str | None = None


def make_messages(
    image: Image.Image,
    example: MixedSiqaExample,
    assistant_text: str | None,
) -> list[dict[str, Any]]:
    if example.family == SIQA_S_FAMILY:
        if example.score_task is None:
            raise ValueError("SIQA-S examples must define score_task.")
        system_prompt, question = build_siqa_s_prompt(example.score_task)
    elif example.family == SIQA_U_FAMILY:
        if example.question is None or example.option is None:
            raise ValueError("SIQA-U examples must define question and option.")
        system_prompt, question = build_siqa_u_prompt(example.question, example.option)
    else:
        raise ValueError(f"Unknown example family: {example.family!r}")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image", "image": image},
            ],
        },
    ]
    if assistant_text is not None:
        messages.append({"role": "assistant", "content": assistant_text})
    return messages


class SiqaScoreDataset(Dataset):
    def __init__(self, jsonl_path: str, root: str) -> None:
        self.jsonl_path = jsonl_path
        self.root = root
        self.examples = self._load_examples()

    def _load_examples(self) -> list[MixedSiqaExample]:
        examples: list[MixedSiqaExample] = []
        with open(self.jsonl_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                image_path = os.path.join(self.root, item["image_path"])
                perception_score = float(item["perception_raing"])
                knowledge_score = float(item["knowledge_rating"])

                perception_word = float_to_rating_word(perception_score)
                knowledge_word = float_to_rating_word(knowledge_score)

                examples.append(
                    MixedSiqaExample(
                        family=SIQA_S_FAMILY,
                        image_path=image_path,
                        score_task="perception",
                        target_score=perception_score,
                        target_text=f"Subjective: {perception_word}",
                    )
                )
                examples.append(
                    MixedSiqaExample(
                        family=SIQA_S_FAMILY,
                        image_path=image_path,
                        score_task="knowledge",
                        target_score=knowledge_score,
                        target_text=f"Objective: {knowledge_word}",
                    )
                )
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> MixedSiqaExample:
        return self.examples[index]


class SiqaUnderstandDataset(Dataset):
    """Lazy-loading SIQA-U dataset that keeps only byte offsets and types in RAM."""

    def __init__(self, jsonl_path: str, root: str) -> None:
        self.jsonl_path = jsonl_path
        self.root = root
        self._offsets: list[int] = []
        self._types: list[str] = []
        self._scan_jsonl()
        self.indices_by_type = self._build_indices_by_type()

    def _scan_jsonl(self) -> None:
        with open(self.jsonl_path, "rb") as handle:
            while True:
                offset = handle.tell()
                raw = handle.readline()
                if not raw:
                    break
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                item = json.loads(line)
                answer = normalize_gold_answer_letter(item["answer"])
                if answer not in VALID_ANSWER_LETTERS:
                    raise ValueError(f"Invalid SIQA-U answer {item['answer']!r} in {self.jsonl_path}")
                question_type = str(item["type"]).strip()
                if question_type not in VALID_SIQA_U_TYPES:
                    raise ValueError(
                        f"Invalid SIQA-U type {item['type']!r} in {self.jsonl_path}. "
                        f"Expected one of: {sorted(VALID_SIQA_U_TYPES)!r}"
                    )
                self._offsets.append(offset)
                self._types.append(question_type)

    def _read_example(self, index: int) -> MixedSiqaExample:
        with open(self.jsonl_path, "rb") as handle:
            handle.seek(self._offsets[index])
            item = json.loads(handle.readline().decode("utf-8"))
        answer = normalize_gold_answer_letter(item["answer"])
        return MixedSiqaExample(
            family=SIQA_U_FAMILY,
            image_path=os.path.join(self.root, item["image_path"]),
            target_text=answer,
            question=str(item["question"]).strip(),
            option=str(item["option"]).strip(),
            answer=answer,
            uqa_type=str(item["type"]).strip(),
        )

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int) -> MixedSiqaExample:
        return self._read_example(index)

    def _build_indices_by_type(self) -> dict[str, list[int]]:
        indices_by_type: dict[str, list[int]] = {}
        for index, question_type in enumerate(self._types):
            indices_by_type.setdefault(question_type, []).append(index)
        return {question_type: sorted(indices) for question_type, indices in sorted(indices_by_type.items())}


class ScoreTokenHelper:
    def __init__(self, processor: Any) -> None:
        tokenizer = processor.tokenizer
        self.class_token_ids: list[list[int]] = []
        self.all_token_ids: set[int] = set()

        for word in QUALITY_WORDS:
            candidates: list[int] = []
            for form in (f" {word}", word):
                token_ids = tokenizer.encode(form, add_special_tokens=False)
                if len(token_ids) == 1 and token_ids[0] != tokenizer.unk_token_id:
                    token_id = token_ids[0]
                    if token_id not in candidates:
                        candidates.append(token_id)
            if not candidates:
                raise RuntimeError(f"Cannot find single-token representation for {word!r}")
            self.class_token_ids.append(candidates)
            self.all_token_ids.update(candidates)

        self.score_values = QUALITY_VALUES.clone()
        self._tensor_cache: dict[tuple[str, torch.dtype | None], list[torch.Tensor]] = {}

    def _get_class_token_tensors(self, device: torch.device) -> list[torch.Tensor]:
        cache_key = (str(device), None)
        cached = self._tensor_cache.get(cache_key)
        if cached is None:
            cached = [torch.tensor(ids, dtype=torch.long, device=device) for ids in self.class_token_ids]
            self._tensor_cache[cache_key] = cached
        return cached

    def compute_class_logits(self, token_logits: torch.Tensor) -> torch.Tensor:
        device = token_logits.device
        class_logits = torch.full(
            (token_logits.size(0), len(self.class_token_ids)),
            -1e9,
            device=device,
            dtype=token_logits.dtype,
        )
        for class_idx, token_ids in enumerate(self._get_class_token_tensors(device)):
            class_logits[:, class_idx] = token_logits.index_select(dim=-1, index=token_ids).max(dim=-1).values
        return class_logits

    def expected_scores(self, token_logits: torch.Tensor) -> torch.Tensor:
        class_logits = self.compute_class_logits(token_logits)
        score_values = self.score_values.to(device=token_logits.device, dtype=token_logits.dtype)
        probs = torch.softmax(class_logits, dim=-1)
        return (probs * score_values).sum(dim=-1)


class MixedTrainCollator:
    def __init__(self, processor: Any, max_length: int, score_tokens: ScoreTokenHelper) -> None:
        self.processor = processor
        self.max_length = max_length
        self.score_tokens = score_tokens
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None or not hasattr(image_processor, "size"):
            raise ValueError("Loaded processor does not expose image_processor.size for pixel limits.")
        self.max_image_pixels = int(image_processor.size["longest_edge"])

    def __call__(self, batch: list[MixedSiqaExample]) -> dict[str, Any]:
        images: list[Image.Image] = []
        prompt_texts: list[str] = []
        full_texts: list[str] = []

        for example in batch:
            image = load_rgb_image_with_pixel_cap(example.image_path, self.max_image_pixels)
            images.append(image)
            prompt_messages = make_messages(image, example, assistant_text=None)
            full_messages = make_messages(image, example, assistant_text=example.target_text)
            prompt_texts.append(
                self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            )
            full_texts.append(
                self.processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
            )

        full_inputs = self.processor(
            text=full_texts,
            images=[[image] for image in images],
            return_tensors="pt",
            padding=True,
            truncation=False,
            max_length=self.max_length,
        )
        prompt_inputs = self.processor(
            text=prompt_texts,
            images=[[image] for image in images],
            return_tensors="pt",
            padding=True,
            truncation=False,
            max_length=self.max_length,
        )

        input_ids = full_inputs["input_ids"]
        attention_mask = full_inputs["attention_mask"]
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        quality_positions: list[int] = []
        target_scores: list[float] = []
        siqa_s_mask: list[bool] = []

        for batch_index, example in enumerate(batch):
            prompt_len = int(prompt_lengths[batch_index].item())
            labels[batch_index, :prompt_len] = -100

            if example.family == SIQA_S_FAMILY:
                sequence_end = int(attention_mask[batch_index].sum().item())
                answer_ids = input_ids[batch_index, prompt_len:sequence_end].tolist()
                quality_offset = self._find_quality_position(answer_ids)
                if quality_offset is None:
                    raise ValueError(
                        f"Could not locate quality token for task={example.score_task} "
                        f"target={example.target_text!r}"
                    )
                quality_position = prompt_len + quality_offset
                labels[batch_index, quality_position] = -100
                quality_positions.append(quality_position)
                if example.target_score is None:
                    raise ValueError("SIQA-S examples must define target_score.")
                target_scores.append(float(example.target_score))
                siqa_s_mask.append(True)
            else:
                quality_positions.append(0)
                target_scores.append(0.0)
                siqa_s_mask.append(False)

        full_inputs["labels"] = labels
        full_inputs["quality_positions"] = torch.tensor(quality_positions, dtype=torch.long)
        full_inputs["target_scores"] = torch.tensor(target_scores, dtype=torch.float32)
        full_inputs["siqa_s_mask"] = torch.tensor(siqa_s_mask, dtype=torch.bool)
        return full_inputs

    def _find_quality_position(self, answer_ids: list[int]) -> int | None:
        for index, token_id in enumerate(answer_ids):
            if token_id in self.score_tokens.all_token_ids:
                return index
        return None


class SiqaScoreEvalCollator:
    def __init__(self, processor: Any, max_length: int) -> None:
        self.processor = processor
        self.max_length = max_length
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None or not hasattr(image_processor, "size"):
            raise ValueError("Loaded processor does not expose image_processor.size for pixel limits.")
        self.max_image_pixels = int(image_processor.size["longest_edge"])

    def __call__(self, batch: list[MixedSiqaExample]) -> dict[str, Any]:
        images: list[Image.Image] = []
        prompt_texts: list[str] = []
        task_ids: list[int] = []
        target_scores: list[float] = []

        for example in batch:
            image = load_rgb_image_with_pixel_cap(example.image_path, self.max_image_pixels)
            images.append(image)
            prompt_messages = make_messages(image, example, assistant_text=None)
            prompt_texts.append(
                self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            )
            if example.score_task == "perception":
                task_ids.append(PERCEPTION_TASK_ID)
            elif example.score_task == "knowledge":
                task_ids.append(KNOWLEDGE_TASK_ID)
            else:
                raise ValueError(f"Unknown SIQA-S score task: {example.score_task!r}")

            if example.target_score is None:
                raise ValueError("SIQA-S validation examples must define target_score.")
            target_scores.append(float(example.target_score))

        inputs = self.processor(
            text=prompt_texts,
            images=[[image] for image in images],
            return_tensors="pt",
            padding=True,
            truncation=False,
            max_length=self.max_length,
        )
        inputs["task_ids"] = torch.tensor(task_ids, dtype=torch.long)
        inputs["target_scores"] = torch.tensor(target_scores, dtype=torch.float32)
        return inputs


class SiqaUnderstandEvalCollator:
    def __init__(self, processor: Any, max_length: int) -> None:
        self.processor = processor
        self.max_length = max_length
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None or not hasattr(image_processor, "size"):
            raise ValueError("Loaded processor does not expose image_processor.size for pixel limits.")
        self.max_image_pixels = int(image_processor.size["longest_edge"])

    def __call__(self, batch: list[MixedSiqaExample]) -> dict[str, Any]:
        images: list[Image.Image] = []
        prompt_texts: list[str] = []
        target_answers: list[str] = []
        uqa_types: list[str] = []

        for example in batch:
            image = load_rgb_image_with_pixel_cap(example.image_path, self.max_image_pixels)
            images.append(image)
            prompt_messages = make_messages(image, example, assistant_text=None)
            prompt_texts.append(
                self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            )
            target_answers.append(str(example.answer or "").strip().upper())
            uqa_types.append(str(example.uqa_type or "").strip())

        inputs = self.processor(
            text=prompt_texts,
            images=[[image] for image in images],
            return_tensors="pt",
            padding=True,
            truncation=False,
            max_length=self.max_length,
        )
        inputs["target_answers"] = target_answers
        inputs["uqa_types"] = uqa_types
        return inputs


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def clear_cuda_cache() -> None:
    if not torch.cuda.is_available():
        return
    gc.collect()
    torch.cuda.empty_cache()


def reset_cuda_memory() -> None:
    """Aggressive memory cleanup that also resets the CUDA caching allocator
    peak-stats and forces synchronization to clear pending async ops."""
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def compute_mixed_loss(
    outputs: Any,
    siqa_s_mask: torch.Tensor,
    quality_positions: torch.Tensor,
    target_scores: torch.Tensor,
    score_tokens: ScoreTokenHelper,
    score_loss_weight: float,
    score_loss_type: str,
    score_huber_delta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lm_loss = outputs.loss
    siqa_s_mask = siqa_s_mask.to(dtype=torch.bool)

    if not torch.any(siqa_s_mask):
        regression_loss = lm_loss.new_zeros(())
        return lm_loss, lm_loss.detach(), regression_loss.detach()

    batch_indices = torch.arange(outputs.logits.size(0), device=outputs.logits.device)[siqa_s_mask]
    score_logit_positions = quality_positions[siqa_s_mask] - 1
    if torch.any(score_logit_positions < 0):
        raise ValueError("quality_positions must be at least 1 to align teacher-forced logits with target tokens")

    quality_logits = outputs.logits[batch_indices, score_logit_positions]
    predicted_scores = score_tokens.expected_scores(quality_logits)
    target_scores = target_scores[siqa_s_mask]

    score_loss_type = score_loss_type.lower()
    if score_loss_type == "mse":
        regression_loss = F.mse_loss(predicted_scores.float(), target_scores.float())
    elif score_loss_type == "huber":
        regression_loss = F.huber_loss(
            predicted_scores.float(),
            target_scores.float(),
            delta=score_huber_delta,
        )
    else:
        raise ValueError(f"Unsupported score_loss_type: {score_loss_type!r}")

    total_loss = lm_loss + (score_loss_weight * regression_loss)
    return total_loss, lm_loss.detach(), regression_loss.detach()


def extract_predicted_scores(
    generated_ids: torch.Tensor,
    scores: list[torch.Tensor],
    score_tokens: ScoreTokenHelper,
) -> list[float]:
    if not scores:
        return [float("nan")] * generated_ids.size(0)

    logits = torch.stack(scores, dim=1)
    predicted_scores: list[float] = []

    for row_index in range(generated_ids.size(0)):
        token_list = generated_ids[row_index].tolist()
        score_value = float("nan")
        match_count = 0
        for token_index, token_id in enumerate(token_list):
            if token_id not in score_tokens.all_token_ids:
                continue
            token_logits = logits[row_index, token_index].unsqueeze(0)
            score_value = score_tokens.expected_scores(token_logits)[0].float().item()
            match_count += 1
            if match_count == 2:
                break

        predicted_scores.append(float(score_value) if math.isfinite(score_value) else float("nan"))

    del logits
    return predicted_scores


def compute_task_correlations(gt_values: list[float], pred_values: list[float]) -> tuple[float, float]:
    valid_pairs = [
        (float(gt), float(pred))
        for gt, pred in zip(gt_values, pred_values)
        if math.isfinite(pred)
    ]
    if len(valid_pairs) < 2:
        return float("nan"), float("nan")

    gt_clean = [pair[0] for pair in valid_pairs]
    pred_clean = [pair[1] for pair in valid_pairs]
    srcc = float(spearmanr(gt_clean, pred_clean).statistic)
    plcc = float(pearsonr(gt_clean, pred_clean).statistic)
    return srcc, plcc


def compute_siqa_s_metrics(
    perception_gt: list[float],
    perception_pred: list[float],
    knowledge_gt: list[float],
    knowledge_pred: list[float],
) -> dict[str, float]:
    srcc_p, plcc_p = compute_task_correlations(perception_gt, perception_pred)
    srcc_k, plcc_k = compute_task_correlations(knowledge_gt, knowledge_pred)
    score_p = max((srcc_p + plcc_p) / 2.0, 0.0) * 100.0
    score_k = max((srcc_k + plcc_k) / 2.0, 0.0) * 100.0
    final_score = (score_p + score_k) / 2.0

    return {
        "Perceptual_SRCC": srcc_p,
        "Perceptual_PLCC": plcc_p,
        "Perceptual_Score": score_p,
        "Knowledge_SRCC": srcc_k,
        "Knowledge_PLCC": plcc_k,
        "Factual_Score": score_k,
        "SIQA_S_Score": final_score,
    }


def compute_siqa_u_metrics(
    question_types: list[str],
    target_answers: list[str],
    predicted_answers: list[str],
) -> dict[str, float]:
    correct = {"yes-or-no": 0, "what": 0, "how": 0}
    total = {"yes-or-no": 0, "what": 0, "how": 0}

    for question_type, target_answer, predicted_answer in zip(question_types, target_answers, predicted_answers):
        if question_type not in total:
            continue

        target_answer = str(target_answer).strip().upper()
        predicted_answer = str(predicted_answer).strip().upper()

        total[question_type] += 1
        if target_answer == predicted_answer:
            correct[question_type] += 1

    accuracy = {
        question_type: (correct[question_type] / total[question_type]) if total[question_type] > 0 else 0.0
        for question_type in total
    }
    score_u = (
        0.2 * accuracy.get("yes-or-no", 0.0)
        + 0.3 * accuracy.get("what", 0.0)
        + 0.5 * accuracy.get("how", 0.0)
    )

    return {
        "ACC_yes-or-no": accuracy.get("yes-or-no", 0.0),
        "ACC_what": accuracy.get("what", 0.0),
        "ACC_how": accuracy.get("how", 0.0),
        "SIQA_U_Score": score_u,
    }


def compute_combined_score(siqa_s_metrics: dict[str, float], siqa_u_metrics: dict[str, float]) -> float:
    return (
        float(siqa_s_metrics["SIQA_S_Score"]) + (100.0 * float(siqa_u_metrics["SIQA_U_Score"]))
    ) / 2.0


class FractionalFamilySampler(Sampler[int]):
    # Each family multiplier applies to that family's own dataset size:
    # `1.0x SIQA-S + 0.5x SIQA-U` means `1.0 * len(SIQA-S)` score examples plus
    # `0.5 * len(SIQA-U)` understanding examples in each epoch.
    def __init__(
        self,
        siqa_s_count: int,
        siqa_u_count: int,
        siqa_u_indices_by_type: dict[str, list[int]],
        sampling_mode: str,
        siqa_s_epoch_multiplier: float,
        siqa_u_epoch_multiplier: float,
        seed: int,
    ) -> None:
        if sampling_mode != "fractional_family":
            raise ValueError(f"Unsupported train_sampling_mode: {sampling_mode!r}")
        if siqa_s_count <= 0 or siqa_u_count <= 0:
            raise ValueError("Both SIQA-S and SIQA-U datasets must be non-empty for fractional sampling.")
        if siqa_s_epoch_multiplier < 0 or siqa_u_epoch_multiplier < 0:
            raise ValueError("Epoch multipliers must be non-negative.")

        self.siqa_s_count = int(siqa_s_count)
        self.siqa_u_count = int(siqa_u_count)
        self.siqa_s_indices = list(range(self.siqa_s_count))
        self.siqa_u_start_index = int(siqa_s_count)
        self.seed = int(seed)
        self.epoch = 0
        self.siqa_s_epoch_multiplier = float(siqa_s_epoch_multiplier)
        self.siqa_u_epoch_multiplier = float(siqa_u_epoch_multiplier)
        self.siqa_u_indices_by_type = self._normalize_siqa_u_indices_by_type(siqa_u_indices_by_type)
        self.num_siqa_s_samples = self._resolve_target_count(
            family_size=self.siqa_s_count,
            multiplier=self.siqa_s_epoch_multiplier,
        )
        self.siqa_u_samples_by_type = {
            question_type: self._resolve_target_count(
                family_size=len(type_indices),
                multiplier=self.siqa_u_epoch_multiplier,
            )
            for question_type, type_indices in self.siqa_u_indices_by_type.items()
        }
        self.num_siqa_u_samples = sum(self.siqa_u_samples_by_type.values())

        if self.num_siqa_s_samples == 0 and self.num_siqa_u_samples == 0:
            raise ValueError("At least one family must contribute examples to each training epoch.")

    def _resolve_target_count(self, family_size: int, multiplier: float) -> int:
        if multiplier == 0.0:
            return 0
        # Round up so a positive fractional multiplier never undershoots the intended fraction
        # because of integer truncation on odd-sized datasets.
        target_count = int(math.ceil(family_size * multiplier))
        return max(1, target_count)

    def _normalize_siqa_u_indices_by_type(
        self,
        siqa_u_indices_by_type: dict[str, list[int]],
    ) -> dict[str, list[int]]:
        if not siqa_u_indices_by_type:
            raise ValueError("SIQA-U type indices must be provided for fractional sampling.")

        normalized: dict[str, list[int]] = {}
        seen_indices: set[int] = set()

        for question_type, type_indices in sorted(siqa_u_indices_by_type.items()):
            cleaned_indices = sorted({int(index) for index in type_indices})
            if not cleaned_indices:
                continue
            for index in cleaned_indices:
                if index < 0 or index >= self.siqa_u_count:
                    raise ValueError(
                        f"SIQA-U type index {index} for question type {question_type!r} is out of range "
                        f"for dataset size {self.siqa_u_count}."
                    )
                if index in seen_indices:
                    raise ValueError(
                        f"SIQA-U type index {index} appears in more than one question type group."
                    )
                seen_indices.add(index)
            normalized[str(question_type).strip()] = cleaned_indices

        if len(seen_indices) != self.siqa_u_count:
            raise ValueError(
                "SIQA-U type indices must cover every SIQA-U example exactly once for fractional sampling."
            )

        return normalized

    def _sample_family_indices(
        self,
        start_index: int,
        base_indices: list[int],
        target_count: int,
        generator: torch.Generator,
    ) -> list[int]:
        if target_count <= 0:
            return []

        if not base_indices:
            raise ValueError("Cannot sample from an empty family index list.")

        relative_indices = torch.tensor(base_indices, dtype=torch.long)
        sampled_indices: list[int] = []
        family_size = len(base_indices)
        full_repeats, remainder = divmod(target_count, family_size)

        for _ in range(full_repeats):
            perm = torch.randperm(family_size, generator=generator)
            sampled_indices.extend((relative_indices.index_select(0, perm) + start_index).tolist())

        if remainder > 0:
            perm = torch.randperm(family_size, generator=generator)[:remainder]
            sampled_indices.extend((relative_indices.index_select(0, perm) + start_index).tolist())

        return sampled_indices

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def summary(self) -> dict[str, Any]:
        return {
            "mode": "fractional_family",
            "siqa_s_epoch_multiplier": self.siqa_s_epoch_multiplier,
            "siqa_u_epoch_multiplier": self.siqa_u_epoch_multiplier,
            "siqa_s_samples_per_epoch": self.num_siqa_s_samples,
            "siqa_u_samples_per_epoch": self.num_siqa_u_samples,
            "siqa_u_type_counts": {
                question_type: len(type_indices)
                for question_type, type_indices in self.siqa_u_indices_by_type.items()
            },
            "siqa_u_samples_by_type": dict(self.siqa_u_samples_by_type),
            "total_samples_per_epoch": len(self),
        }

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        siqa_s_indices = self._sample_family_indices(
            start_index=0,
            base_indices=self.siqa_s_indices,
            target_count=self.num_siqa_s_samples,
            generator=generator,
        )
        siqa_u_indices: list[int] = []
        for question_type, type_indices in self.siqa_u_indices_by_type.items():
            siqa_u_indices.extend(
                self._sample_family_indices(
                    start_index=self.siqa_u_start_index,
                    base_indices=type_indices,
                    target_count=self.siqa_u_samples_by_type[question_type],
                    generator=generator,
                )
            )
        combined_indices = siqa_s_indices + siqa_u_indices

        if combined_indices:
            shuffle_order = torch.randperm(len(combined_indices), generator=generator).tolist()
            combined_indices = [combined_indices[index] for index in shuffle_order]

        return iter(combined_indices)

    def __len__(self) -> int:
        return self.num_siqa_s_samples + self.num_siqa_u_samples


def save_checkpoint(model: Any, processor: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)


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


def is_multi_node_run(accelerator: Accelerator) -> bool:
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    return accelerator.num_processes > local_world_size


def evaluate_siqa_s_model(
    model: Any,
    processor: Any,
    dataloader: DataLoader,
    device: torch.device,
    score_tokens: ScoreTokenHelper,
    max_new_tokens: int,
    show_progress: bool = False,
) -> dict[str, float]:
    model.eval()
    perception_gt: list[float] = []
    perception_pred: list[float] = []
    knowledge_gt: list[float] = []
    knowledge_pred: list[float] = []

    with torch.no_grad():
        progress_bar = tqdm(
            dataloader,
            desc="Validation SIQA-S",
            leave=False,
            disable=not show_progress,
        )
        for batch in progress_bar:
            batch = move_batch_to_device(batch, device)
            task_ids = batch.pop("task_ids")
            target_scores = batch.pop("target_scores")

            model_inputs = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
            outputs = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
                use_cache=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
            input_length = model_inputs["input_ids"].shape[1]
            generated_ids = outputs.sequences[:, input_length:].detach().cpu()
            scores_cpu = [s.detach().cpu() for s in outputs.scores]
            del outputs
            torch.cuda.empty_cache()
            predicted_scores = extract_predicted_scores(generated_ids, scores_cpu, score_tokens)
            del generated_ids, scores_cpu
            task_id_list = task_ids.detach().cpu().tolist()
            target_score_list = target_scores.detach().cpu().tolist()

            for task_id, target_score, predicted_score in zip(task_id_list, target_score_list, predicted_scores):
                if int(task_id) == PERCEPTION_TASK_ID:
                    perception_gt.append(float(target_score))
                    perception_pred.append(float(predicted_score))
                else:
                    knowledge_gt.append(float(target_score))
                    knowledge_pred.append(float(predicted_score))

    return compute_siqa_s_metrics(perception_gt, perception_pred, knowledge_gt, knowledge_pred)


def evaluate_siqa_u_model(
    model: Any,
    processor: Any,
    dataloader: DataLoader,
    device: torch.device,
    max_new_tokens: int,
    show_progress: bool = False,
) -> dict[str, float]:
    model.eval()
    question_types: list[str] = []
    target_answers: list[str] = []
    predicted_answers: list[str] = []

    with torch.no_grad():
        progress_bar = tqdm(
            dataloader,
            desc="Validation SIQA-U",
            leave=False,
            disable=not show_progress,
        )
        for batch in progress_bar:
            batch = move_batch_to_device(batch, device)
            batch_target_answers = [str(answer).strip().upper() for answer in batch.pop("target_answers")]
            batch_uqa_types = [str(question_type).strip() for question_type in batch.pop("uqa_types")]

            model_inputs = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
            outputs = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                do_sample=False,
                use_cache=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
            input_length = model_inputs["input_ids"].shape[1]
            generated_ids = outputs.sequences[:, input_length:].detach().cpu()
            del outputs
            torch.cuda.empty_cache()
            decoded_answers = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            del generated_ids
            normalized_answers = [normalize_answer_letter(text) for text in decoded_answers]

            question_types.extend(batch_uqa_types)
            target_answers.extend(batch_target_answers)
            predicted_answers.extend(normalized_answers)

    return compute_siqa_u_metrics(question_types, target_answers, predicted_answers)


def write_history(output_dir: Path, record: dict[str, Any]) -> None:
    history_path = output_dir / "history.jsonl"
    with open(history_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    model_name = "Qwen/Qwen3-VL-2B-Instruct"
    train_siqa_s_jsonl = "TrainSet/train_SIQA-S.jsonl"
    train_siqa_u_jsonl = "TrainSet/train_SIQA-U.jsonl"
    val_siqa_s_jsonl = "TrainSet/SIQA-S-valid.jsonl"
    val_siqa_u_jsonl = "TrainSet/SIQA-U-valid.jsonl"
    root = "TrainSet"
    output_dir = Path("checkpoints/qwen3-vl-2b-siqa-mixed-lora_frac_01")
    resume_dir = output_dir / "resume_state"
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
    num_workers = 0
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
    siqa_s_epoch_multiplier = 1.0 # Full sample: 16800
    siqa_u_epoch_multiplier = 0.1 # Full sample: 104021
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
        pin_memory=False,
        collate_fn=train_collator,
    )
    val_siqa_s_loader = DataLoader(
        val_siqa_s_dataset,
        batch_size=per_device_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=score_eval_collator,
    )
    val_siqa_u_loader = DataLoader(
        val_siqa_u_dataset,
        batch_size=per_device_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
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
        reset_cuda_memory()
        accelerator.wait_for_everyone()
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
                del outputs
                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_parameters, gradient_clip_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            reduced_total_loss = accelerator.gather(total_loss.detach().unsqueeze(0)).mean().item()
            reduced_lm_loss = accelerator.gather(lm_loss.detach().unsqueeze(0)).mean().item()
            reduced_score_loss = accelerator.gather(score_loss.detach().unsqueeze(0)).mean().item()
            del total_loss, lm_loss, score_loss

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
                        f"epoch_optimizer_step={epoch_optimizer_steps}/{optimizer_steps_per_epoch} ({epoch_progress:.1f}%) "
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
            clear_cuda_cache()
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
        reset_cuda_memory()

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
