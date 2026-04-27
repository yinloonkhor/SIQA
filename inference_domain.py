import json
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


IMAGE_PLACEHOLDER = "<image>"
QWEN_CHAT_PROMPT_FORMAT = "qwen_chat_v1"
SECTION_MARKER_REPLACEMENTS = {
    "<|content|>": "Content",
    "<|context|>": "Context",
    "<|outline|>": "Outline",
}


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_text_input(inline_value: str | None = None, file_path: Path | None = None) -> str:
    if inline_value is not None and file_path is not None:
        raise ValueError("Use only one of the inline text value or the file path.")
    if file_path is not None:
        return file_path.expanduser().read_text(encoding="utf-8").strip()
    return (inline_value or "").strip()


def normalize_role(raw_role: Any) -> str | None:
    if raw_role is None:
        return None

    role = str(raw_role).strip().lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant"}:
        return "assistant"
    if role == "system":
        return "system"
    return None


def resolve_adapter_dir(checkpoint: Path, adapter_subdir: str) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    if (checkpoint / "adapter_config.json").exists():
        return checkpoint

    candidate = checkpoint / adapter_subdir
    if (candidate / "adapter_config.json").exists():
        return candidate

    raise FileNotFoundError(
        f"Could not find adapter_config.json under {checkpoint} or {candidate}."
    )


def load_dataset_sample(jsonl_path: Path, image_root: Path, sample_index: int) -> dict[str, Any]:
    jsonl_path = jsonl_path.expanduser().resolve()
    image_root = image_root.expanduser().resolve()

    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for current_index, line in enumerate(handle):
            if current_index != sample_index:
                continue

            item = json.loads(line)
            conversations = item.get("conversations") or []
            if not conversations:
                raise ValueError(f"Sample {sample_index} in {jsonl_path} has no conversations.")

            prompt_turns: list[tuple[str, str]] = []
            for message in conversations[:-1]:
                role = normalize_role(message.get("from"))
                if role is None:
                    raise ValueError(f"Unsupported role {message.get('from')!r} in sample {sample_index}.")
                prompt_turns.append((role, str(message.get("value") or "")))

            image_paths: list[Path] = []
            for image_path in item.get("image") or []:
                resolved_path = Path(image_path)
                if not resolved_path.is_absolute():
                    resolved_path = image_root / resolved_path
                image_paths.append(resolved_path.resolve())

            return {
                "example_id": str(item.get("id") or ""),
                "task_type": str(item.get("task_type") or ""),
                "system_text": str(item.get("system_instruction") or "").strip(),
                "prompt_turns": prompt_turns,
                "target_text": str(conversations[-1].get("value") or "").strip(),
                "image_paths": image_paths,
            }

    raise IndexError(f"Sample index {sample_index} is out of range for {jsonl_path}.")


def configure_image_processor(processor: Any, min_pixels: int | None, max_pixels: int | None) -> None:
    if min_pixels is None or max_pixels is None:
        return

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None or not hasattr(image_processor, "size"):
        return

    image_processor.size = {
        "shortest_edge": int(min_pixels),
        "longest_edge": int(max_pixels),
    }


def resolve_model_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def load_trusted_rgb_image(image_path: Path) -> Image.Image:
    previous_max_pixels = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(image_path) as image_file:
                if image_file.mode == "P" and "transparency" in image_file.info:
                    image_file = image_file.convert("RGBA")
                return image_file.convert("RGB").copy()
    finally:
        Image.MAX_IMAGE_PIXELS = previous_max_pixels


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


def ensure_prompt_turns_have_image_placeholders(
    system_text: str,
    prompt_turns: list[tuple[str, str]],
    image_count: int,
) -> tuple[str, list[tuple[str, str]]]:
    normalized_turns = [(role, str(text)) for role, text in prompt_turns]
    placeholder_count = system_text.count(IMAGE_PLACEHOLDER) + sum(
        text.count(IMAGE_PLACEHOLDER) for _, text in normalized_turns
    )

    if image_count <= 0:
        if placeholder_count > 0:
            raise ValueError(
                f"Found {placeholder_count} {IMAGE_PLACEHOLDER!r} placeholders but received no images."
            )
        return system_text, normalized_turns

    if placeholder_count == 0:
        prefix = "\n".join([IMAGE_PLACEHOLDER] * image_count)
        for index, (role, text) in enumerate(normalized_turns):
            if role != "user":
                continue
            normalized_turns[index] = (role, f"{prefix}\n{text}".strip())
            return system_text, normalized_turns

        normalized_turns.insert(0, ("user", prefix))
        return system_text, normalized_turns

    if placeholder_count != image_count:
        raise ValueError(
            f"Found {placeholder_count} {IMAGE_PLACEHOLDER!r} placeholders but received {image_count} image(s)."
        )

    return system_text, normalized_turns


def build_qwen_chat_messages(
    system_text: str,
    prompt_turns: list[tuple[str, str]],
    image_count: int,
) -> list[dict[str, Any]]:
    raw_messages: list[tuple[str, str]] = []
    if system_text.strip():
        raw_messages.append(("system", system_text))

    for role, text in prompt_turns:
        if raw_messages and raw_messages[-1][0] == role:
            raw_messages[-1] = (role, _join_message_text(raw_messages[-1][1], text))
            continue
        raw_messages.append((role, str(text)))

    image_cursor = 0
    qwen_messages: list[dict[str, Any]] = []
    for role, text in raw_messages:
        content, image_cursor = _inline_text_to_qwen_content(
            text,
            image_count=image_count,
            image_cursor=image_cursor,
        )
        if not content:
            continue
        qwen_messages.append({"role": role, "content": content})

    if image_cursor != image_count:
        raise ValueError(
            f"Expected {image_count} images but only used {image_cursor} placeholders while building chat messages."
        )

    return qwen_messages


def build_generation_prompt_from_turns(
    processor: Any,
    system_text: str,
    prompt_turns: list[tuple[str, str]],
    image_count: int,
) -> tuple[str, list[dict[str, Any]] | None]:
    if not hasattr(processor, "apply_chat_template"):
        raise ValueError("Loaded processor does not expose apply_chat_template required for native Qwen chat formatting.")

    qwen_messages = build_qwen_chat_messages(system_text, prompt_turns, image_count=image_count)
    prompt_text = processor.apply_chat_template(
        qwen_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt_text, qwen_messages


def load_generator_from_checkpoint(adapter_dir: Path, device_map: str | None) -> tuple[Any, Any, dict[str, Any]]:
    adapter_config = load_json(adapter_dir / "adapter_config.json")
    train_config_path = adapter_dir.parent / "train_config.json"
    train_config = load_json(train_config_path) if train_config_path.exists() else {}

    base_model_name = adapter_config["base_model_name_or_path"]
    mixed_precision = train_config.get("mixed_precision", "no")
    min_pixels = train_config.get("image_min_pixels")
    max_pixels = train_config.get("image_max_pixels")

    processor = AutoProcessor.from_pretrained(adapter_dir, trust_remote_code=True)
    configure_image_processor(processor, min_pixels=min_pixels, max_pixels=max_pixels)

    load_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    if mixed_precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        load_kwargs["dtype"] = torch.bfloat16

    base_model = AutoModelForImageTextToText.from_pretrained(base_model_name, **load_kwargs)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    model.eval()
    return model, processor, train_config


def prepare_inputs(
    processor: Any,
    prompt_text: str,
    image_paths: list[Path],
    model_device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    resolved_image_paths = [str(path.expanduser().resolve()) for path in image_paths]
    loaded_images = [load_trusted_rgb_image(Path(path)) for path in resolved_image_paths]

    processor_kwargs: dict[str, Any] = {
        "text": [prompt_text],
        "return_tensors": "pt",
        "padding": False,
        "truncation": False,
    }
    if loaded_images:
        processor_kwargs["images"] = [loaded_images]

    inputs = processor(**processor_kwargs)
    tensor_inputs = {
        key: value.to(model_device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    return tensor_inputs, resolved_image_paths


def generate_text(
    model: Any,
    processor: Any,
    prompt_text: str,
    image_paths: list[Path],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, list[str]]:
    model_device = resolve_model_device(model)
    inputs, resolved_image_paths = prepare_inputs(
        processor=processor,
        prompt_text=prompt_text,
        image_paths=image_paths,
        model_device=model_device,
    )

    generation_config = deepcopy(model.generation_config)
    generation_config.max_new_tokens = int(max_new_tokens)
    generation_config.do_sample = float(temperature) > 0.0
    if generation_config.do_sample:
        generation_config.temperature = float(temperature)
        generation_config.top_p = float(top_p)
    else:
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, generation_config=generation_config)

    prompt_length = int(inputs["input_ids"].shape[1])
    new_token_ids = generated_ids[0, prompt_length:]
    response_text = processor.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
    return response_text, resolved_image_paths


def run_prompt_turns(
    model: Any,
    processor: Any,
    system_text: str,
    prompt_turns: list[tuple[str, str]],
    image_paths: list[Path],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    print_prompt: bool,
) -> dict[str, Any]:
    system_text, prompt_turns = ensure_prompt_turns_have_image_placeholders(
        system_text,
        prompt_turns,
        len(image_paths),
    )
    prompt_text, qwen_messages = build_generation_prompt_from_turns(
        processor=processor,
        system_text=system_text,
        prompt_turns=prompt_turns,
        image_count=len(image_paths),
    )

    if print_prompt:
        print("=== Prompt ===")
        print(prompt_text)
        print("==============")

    response_text, resolved_image_paths = generate_text(
        model=model,
        processor=processor,
        prompt_text=prompt_text,
        image_paths=image_paths,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    return {
        "system": system_text,
        "prompt_turns": [{"role": role, "text": text} for role, text in prompt_turns],
        "prompt_format": QWEN_CHAT_PROMPT_FORMAT,
        "messages": qwen_messages,
        "prompt_text": prompt_text,
        "images": resolved_image_paths,
        "response": response_text,
    }


def run_single_prompt(
    model: Any,
    processor: Any,
    system_text: str,
    user_text: str,
    image_paths: list[Path],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    print_prompt: bool,
) -> dict[str, Any]:
    return run_prompt_turns(
        model=model,
        processor=processor,
        system_text=system_text,
        prompt_turns=[("user", user_text)],
        image_paths=image_paths,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        print_prompt=print_prompt,
    )


def interactive_loop(
    model: Any,
    processor: Any,
    system_text: str,
    image_paths: list[Path],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    print_prompt: bool,
) -> None:
    print("Interactive mode. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            user_text = input("Human> ").strip()
        except EOFError:
            print()
            break

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        result = run_single_prompt(
            model=model,
            processor=processor,
            system_text=system_text,
            user_text=user_text,
            image_paths=image_paths,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            print_prompt=print_prompt,
        )
        print("\nAI>")
        print(result["response"])
        print()


def main() -> None:
    checkpoint = Path("checkpoints/qwen3-vl-2b-domain-lora")
    adapter_subdir = "best"
    use_validation_sample = True
    sample_jsonl = Path("M-Paper/sft/3tasks_val.jsonl")
    sample_image_root = Path("M-Paper")
    sample_index = 2
    print_reference_answer = True
    image_paths: list[Path] = [
        # Path("path/to/image.png"),
    ]
    prompt = None
    prompt_file = None
    system_text = ""
    system_file = None
    max_new_tokens = 512
    temperature = 0.0
    top_p = 1.0
    device_map = None
    print_prompt = True
    output_json = None

    adapter_dir = resolve_adapter_dir(checkpoint, adapter_subdir)
    prompt_turns: list[tuple[str, str]] | None = None
    reference_answer = None

    if use_validation_sample:
        sample = load_dataset_sample(
            jsonl_path=sample_jsonl,
            image_root=sample_image_root,
            sample_index=sample_index,
        )
        system_text = sample["system_text"]
        prompt_turns = sample["prompt_turns"]
        image_paths = sample["image_paths"]
        reference_answer = sample["target_text"]
        print(
            f"Loaded sample index={sample_index} id={sample['example_id']} "
            f"task={sample['task_type']} from {sample_jsonl}"
        )
    else:
        system_text = resolve_text_input(system_text, system_file)

    user_text = resolve_text_input(prompt, prompt_file)

    model, processor, train_config = load_generator_from_checkpoint(
        adapter_dir=adapter_dir,
        device_map=device_map,
    )
    if device_map is None and torch.cuda.is_available():
        model = model.to("cuda")

    print(f"Loaded adapter: {adapter_dir}")
    print(f"Base model: {load_json(adapter_dir / 'adapter_config.json')['base_model_name_or_path']}")
    print(
        "Image processor limits: "
        f"min_pixels={train_config.get('image_min_pixels')} "
        f"max_pixels={train_config.get('image_max_pixels')}"
    )
    print(f"Prompt format: {QWEN_CHAT_PROMPT_FORMAT}")

    if prompt_turns is not None:
        result = run_prompt_turns(
            model=model,
            processor=processor,
            system_text=system_text,
            prompt_turns=prompt_turns,
            image_paths=image_paths,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            print_prompt=print_prompt,
        )
        print(result["response"])

        if print_reference_answer and reference_answer:
            print("\n=== Reference Answer ===")
            print(reference_answer)

        if output_json is not None:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            payload = dict(result)
            payload["reference_answer"] = reference_answer
            with open(output_json, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
            print(f"Saved response JSON to {output_json}")
        return

    if user_text:
        result = run_single_prompt(
            model=model,
            processor=processor,
            system_text=system_text,
            user_text=user_text,
            image_paths=image_paths,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            print_prompt=print_prompt,
        )
        print(result["response"])

        if output_json is not None:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2, ensure_ascii=False)
            print(f"Saved response JSON to {output_json}")
        return

    interactive_loop(
        model=model,
        processor=processor,
        system_text=system_text,
        image_paths=image_paths,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        print_prompt=print_prompt,
    )


if __name__ == "__main__":
    main()
