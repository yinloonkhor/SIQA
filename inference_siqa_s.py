import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from BaseModel import ScoreModel


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


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


def load_score_model_from_checkpoint(checkpoint_dir: Path, device_map: str | None = None) -> ScoreModel:
    adapter_config = load_json(checkpoint_dir / "adapter_config.json")
    train_config_path = checkpoint_dir.parent / "train_config.json"
    train_config = load_json(train_config_path) if train_config_path.exists() else {}

    base_model_name = adapter_config["base_model_name_or_path"]
    mixed_precision = train_config.get("mixed_precision", "no")
    min_pixels = train_config.get("image_min_pixels")
    max_pixels = train_config.get("image_max_pixels")

    processor = AutoProcessor.from_pretrained(checkpoint_dir, trust_remote_code=True)
    configure_image_processor(processor, min_pixels=min_pixels, max_pixels=max_pixels)

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
    }
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    if mixed_precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        load_kwargs["torch_dtype"] = torch.bfloat16

    base_model = AutoModelForImageTextToText.from_pretrained(base_model_name, **load_kwargs)
    base_model = PeftModel.from_pretrained(base_model, checkpoint_dir)

    model = ScoreModel(base_model, processor)
    model.score_head = model.score_head.to(resolve_model_device(base_model))
    model.eval()
    return model


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_submission(
    scorer: ScoreModel,
    dataset: list[dict[str, Any]],
    root: Path,
    team: str,
    method: str,
) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []

    for index, item in enumerate(tqdm(dataset, desc="SIQA-S Test Inference"), start=1):
        image_path = root / item["image_path"]
        perception = scorer.predict_score(str(image_path), "perception")
        knowledge = scorer.predict_score(str(image_path), "knowledge")
        predictions.append(
            {
                "id": index,
                "perception": round(float(perception), 4),
                "knowledge": round(float(knowledge), 4),
            }
        )

    return {
        "team": team,
        "method": method,
        "track": "S",
        "predictions": predictions,
    }
def main() -> None:
    checkpoint_dir = Path("checkpoints/qwen3-vl-2b-siqa-s-lora_lr_1e4/best")
    input_path = Path("TrainSet/SIQA-S-test.jsonl")
    root = Path("TrainSet")
    output_path = Path("submissions/DoubleY_data_SIQA-S.json")
    team = "DoubleY"
    method = "data"
    device_map = "auto" if torch.cuda.is_available() else None

    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    dataset = load_jsonl(input_path)
    scorer = load_score_model_from_checkpoint(checkpoint_dir, device_map=device_map)
    submission = build_submission(
        scorer=scorer,
        dataset=dataset,
        root=root,
        team=team,
        method=method,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(submission, handle, indent=2, ensure_ascii=False)

    print(f"Saved SIQA-S submission to {output_path}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Input: {input_path}")
    print(f"Team: {team}")
    print(f"Method: {method}")
    print(f"Items: {len(submission['predictions'])}")


if __name__ == "__main__":
    main()
