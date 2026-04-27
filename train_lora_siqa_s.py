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
from torch.utils.data import DataLoader, Dataset
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
PERCEPTION_TASK_ID = 0
KNOWLEDGE_TASK_ID = 1
TEXT_LORA_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
VISION_LORA_SUFFIXES = ("qkv", "proj")
LORA_SUPPORTED_MODULE_TYPES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
RESUME_CONFIG_KEYS = (
    "model",
    "train_jsonl",
    "val_jsonl",
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
    # for module_name in resolved_targets["text"]:
    #     print(f"    {module_name}")
    print(f"  Vision ({len(resolved_targets['vision'])}):")
    # for module_name in resolved_targets["vision"]:
    #     print(f"    {module_name}")

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


def build_task_prompt(task: str) -> tuple[str, str]:
    if task == "perception":
        system_prompt = """You are an expert in scientific image analysis. Evaluate the given image on **Subjective Quality** only:

- Consider technical quality (sharpness, lighting, legibility) and aesthetic quality (visual appeal, layout balance, information density).
- Ignore scientific correctness.

Use exactly one of these five terms: [Bad, Poor, Fair, Good, Excellent]. Respond ONLY as:

Subjective: [Quality Word]
"""
        question = "How would you rate the subjective quality of this image?"
    elif task == "knowledge":
        system_prompt = """You are an expert in scientific image analysis. Evaluate the given image on **Objective Quality** only:

- Assess scientific rigor: completeness (e.g., scale bars, axis labels, units), correctness of data, and avoidance of redundancy.
- Ignore aesthetics or technical rendering.

Use exactly one of these five terms: [Bad, Poor, Fair, Good, Excellent]. Respond ONLY as:

Objective: [Quality Word]
"""
        question = "How would you rate the objective quality of this image?"
    else:
        raise ValueError(f"Unknown task: {task}")
    return system_prompt, question


def make_messages(image: Image.Image, task: str, assistant_text: str | None) -> list[dict[str, Any]]:
    system_prompt, question = build_task_prompt(task)
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


@dataclass
class SiqaScoreExample:
    image_path: str
    task: str
    target_score: float
    target_word: str
    target_text: str


class SiqaScoreDataset(Dataset):
    def __init__(self, jsonl_path: str, root: str) -> None:
        self.jsonl_path = jsonl_path
        self.root = root
        self.examples = self._load_examples()

    def _load_examples(self) -> list[SiqaScoreExample]:
        examples: list[SiqaScoreExample] = []
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
                    SiqaScoreExample(
                        image_path=image_path,
                        task="perception",
                        target_score=perception_score,
                        target_word=perception_word,
                        target_text=f"Subjective: {perception_word}",
                    )
                )
                examples.append(
                    SiqaScoreExample(
                        image_path=image_path,
                        task="knowledge",
                        target_score=knowledge_score,
                        target_word=knowledge_word,
                        target_text=f"Objective: {knowledge_word}",
                    )
                )
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> SiqaScoreExample:
        return self.examples[index]


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


class TrainCollator:
    def __init__(self, processor: Any, max_length: int, score_tokens: ScoreTokenHelper) -> None:
        self.processor = processor
        self.max_length = max_length
        self.score_tokens = score_tokens

    def __call__(self, batch: list[SiqaScoreExample]) -> dict[str, Any]:
        images: list[Image.Image] = []
        prompt_texts: list[str] = []
        full_texts: list[str] = []

        for example in batch:
            with Image.open(example.image_path) as image_file:
                image = image_file.convert("RGB").copy()
            images.append(image)
            prompt_messages = make_messages(image, example.task, assistant_text=None)
            full_messages = make_messages(image, example.task, assistant_text=example.target_text)
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

        for batch_index, example in enumerate(batch):
            prompt_len = int(prompt_lengths[batch_index].item())
            labels[batch_index, :prompt_len] = -100

            sequence_end = int(attention_mask[batch_index].sum().item())
            answer_ids = input_ids[batch_index, prompt_len:sequence_end].tolist()
            quality_offset = self._find_quality_position(answer_ids)
            if quality_offset is None:
                raise ValueError(
                    f"Could not locate quality token for task={example.task} target={example.target_text!r}"
                )
            quality_position = prompt_len + quality_offset
            # Keep the score token in the teacher-forced input, but supervise it only through the SIQA score path.
            labels[batch_index, quality_position] = -100
            quality_positions.append(quality_position)
            target_scores.append(example.target_score)

        full_inputs["labels"] = labels
        full_inputs["quality_positions"] = torch.tensor(quality_positions, dtype=torch.long)
        full_inputs["target_scores"] = torch.tensor(target_scores, dtype=torch.float32)
        return full_inputs

    def _find_quality_position(self, answer_ids: list[int]) -> int | None:
        for index, token_id in enumerate(answer_ids):
            if token_id in self.score_tokens.all_token_ids:
                return index
        return None


class EvalCollator:
    def __init__(self, processor: Any, max_length: int) -> None:
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch: list[SiqaScoreExample]) -> dict[str, Any]:
        images: list[Image.Image] = []
        prompt_texts: list[str] = []
        task_ids: list[int] = []
        target_scores: list[float] = []

        for example in batch:
            with Image.open(example.image_path) as image_file:
                image = image_file.convert("RGB").copy()
            images.append(image)
            prompt_messages = make_messages(image, example.task, assistant_text=None)
            prompt_texts.append(
                self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            )
            task_ids.append(PERCEPTION_TASK_ID if example.task == "perception" else KNOWLEDGE_TASK_ID)
            target_scores.append(example.target_score)

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


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def compute_hybrid_loss(
    outputs: Any,
    quality_positions: torch.Tensor,
    target_scores: torch.Tensor,
    score_tokens: ScoreTokenHelper,
    score_loss_weight: float,
    score_loss_type: str,
    score_huber_delta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = outputs.logits.size(0)
    batch_index = torch.arange(batch_size, device=outputs.logits.device)
    score_logit_positions = quality_positions - 1
    if torch.any(score_logit_positions < 0):
        raise ValueError("quality_positions must be at least 1 to align teacher-forced logits with target tokens")
    quality_logits = outputs.logits[batch_index, score_logit_positions]
    predicted_scores = score_tokens.expected_scores(quality_logits)
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
    total_loss = outputs.loss + (score_loss_weight * regression_loss)
    return total_loss, outputs.loss.detach(), regression_loss.detach()


def extract_predicted_scores(
    processor: Any,
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

        if math.isfinite(score_value):
            predicted_scores.append(score_value)
            continue

        predicted_scores.append(float("nan"))

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


def compute_siqa_s_metrics(perception_gt: list[float], perception_pred: list[float], knowledge_gt: list[float], knowledge_pred: list[float]) -> dict[str, float]:
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


def evaluate_model(
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
            desc="Validation",
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
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
            input_length = model_inputs["input_ids"].shape[1]
            generated_ids = outputs.sequences[:, input_length:]
            predicted_scores = extract_predicted_scores(processor, generated_ids, outputs.scores, score_tokens)
            task_id_list = task_ids.detach().cpu().tolist()
            target_score_list = target_scores.detach().cpu().tolist()
            predicted_score_list = predicted_scores

            for task_id, target_score, predicted_score in zip(task_id_list, target_score_list, predicted_score_list):
                if int(task_id) == PERCEPTION_TASK_ID:
                    perception_gt.append(float(target_score))
                    perception_pred.append(float(predicted_score))
                else:
                    knowledge_gt.append(float(target_score))
                    knowledge_pred.append(float(predicted_score))

    return compute_siqa_s_metrics(perception_gt, perception_pred, knowledge_gt, knowledge_pred)


def write_history(output_dir: Path, record: dict[str, Any]) -> None:
    history_path = output_dir / "history.jsonl"
    with open(history_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    model_name = "Qwen/Qwen3-VL-2B-Instruct"
    train_jsonl = "TrainSet/train_SIQA-S.jsonl"
    val_jsonl = "TrainSet/SIQA-S-valid.jsonl"
    root = "TrainSet"
    output_dir = Path("checkpoints/qwen3-vl-2b-siqa-s-lora_v2")
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

    train_dataset = SiqaScoreDataset(train_jsonl, root)
    val_dataset = SiqaScoreDataset(val_jsonl, root)

    train_collator = TrainCollator(processor, max_length, score_tokens)
    eval_collator = EvalCollator(eval_processor, max_length)

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
        collate_fn=eval_collator,
    )

    if accelerator.is_main_process:
        train_task_counts = {"perception": 0, "knowledge": 0}
        for example in train_dataset.examples:
            train_task_counts[example.task] += 1
        val_task_counts = {"perception": 0, "knowledge": 0}
        for example in val_dataset.examples:
            val_task_counts[example.task] += 1
        print(f"Train examples: {len(train_dataset)} {train_task_counts}")
        print(f"Val examples:   {len(val_dataset)} {val_task_counts}")

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
            model_inputs = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}

            with accelerator.accumulate(model):
                outputs = model(**model_inputs)
                total_loss, lm_loss, score_loss = compute_hybrid_loss(
                    outputs=outputs,
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
        if accelerator.is_main_process:
            # Run validation on the full, unsharded loader from rank 0 only. This avoids
            # cross-rank generation stalls turning into NCCL timeouts during metric gathers.
            metrics = evaluate_model(
                model=raw_model,
                processor=eval_processor,
                dataloader=val_loader,
                device=accelerator.device,
                score_tokens=score_tokens,
                max_new_tokens=max_new_tokens,
                show_progress=True,
            )
        accelerator.wait_for_everyone()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        last_dir = output_dir / "last"
        if save_last_on_each_node:
            if accelerator.local_process_index == 0:
                save_checkpoint(raw_model, eval_processor, last_dir)
        elif accelerator.is_main_process:
            save_checkpoint(raw_model, eval_processor, last_dir)
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            current_metric = metrics["SIQA_S_Score"]
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
                "best_SIQA_S_Score": best_metric,
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
