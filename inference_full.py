import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from BaseModel import ScoreModel, UnderstandModel


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


def load_models_from_checkpoint(
    checkpoint_dir: Path,
    device_map: str | None = None,
) -> tuple[ScoreModel, UnderstandModel]:
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

    scorer = ScoreModel(base_model, processor)
    scorer.score_head = scorer.score_head.to(resolve_model_device(base_model))
    scorer.eval()

    understander = UnderstandModel(base_model, processor)
    understander.eval()
    return scorer, understander


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_siqa_s_submission(
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


def build_siqa_u_submission(
    understander: UnderstandModel,
    dataset: list[dict[str, Any]],
    root: Path,
    team: str,
    method: str,
) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []

    for index, item in enumerate(tqdm(dataset, desc="SIQA-U Test Inference"), start=1):
        image_rel_path = item.get("image_path") or item["image"]
        image_path = root / image_rel_path
        answer = understander.predict_answer(
            str(image_path),
            question=item["question"],
            option=item["option"],
        )
        answer = answer[0].strip().upper() if answer else ""
        predictions.append(
            {
                "id": index,
                "type": item["type"],
                "precision": answer,
            }
        )

    return {
        "team": team,
        "method": method,
        "track": "U",
        "predictions": predictions,
    }


def main() -> None:
    checkpoint_dir = Path("checkpoints/qwen3-vl-2b-siqa-mixed-lora_frac_05/best")
    input_siqa_s_path = Path("TrainSet/SIQA-S-test.jsonl")
    input_siqa_u_path = Path("TrainSet/SIQA-U-test.jsonl")
    output_siqa_s_path = Path("submissions/DoubleY_data_SIQA-S.json")
    output_siqa_u_path = Path("submissions/DoubleY_data_SIQA-U.json")
    team = "DoubleY"
    method = "data"
    device_map = "auto" if torch.cuda.is_available() else None

    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    train_config_path = checkpoint_dir.parent / "train_config.json"
    train_config = load_json(train_config_path) if train_config_path.exists() else {}
    root = Path(train_config.get("root", "TrainSet"))

    siqa_s_dataset = load_jsonl(input_siqa_s_path)
    siqa_u_dataset = load_jsonl(input_siqa_u_path)
    scorer, understander = load_models_from_checkpoint(checkpoint_dir, device_map=device_map)
    siqa_s_submission = build_siqa_s_submission(
        scorer=scorer,
        dataset=siqa_s_dataset,
        root=root,
        team=team,
        method=method,
    )
    siqa_u_submission = build_siqa_u_submission(
        understander=understander,
        dataset=siqa_u_dataset,
        root=root,
        team=team,
        method=method,
    )

    output_siqa_s_path.parent.mkdir(parents=True, exist_ok=True)
    output_siqa_u_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_siqa_s_path, "w", encoding="utf-8") as handle:
        json.dump(siqa_s_submission, handle, indent=2, ensure_ascii=False)
    with open(output_siqa_u_path, "w", encoding="utf-8") as handle:
        json.dump(siqa_u_submission, handle, indent=2, ensure_ascii=False)

    print(f"Saved SIQA-S submission to {output_siqa_s_path}")
    print(f"Saved SIQA-U submission to {output_siqa_u_path}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"SIQA-S input: {input_siqa_s_path}")
    print(f"SIQA-U input: {input_siqa_u_path}")
    print(f"Team: {team}")
    print(f"Method: {method}")
    print(f"SIQA-S items: {len(siqa_s_submission['predictions'])}")
    print(f"SIQA-U items: {len(siqa_u_submission['predictions'])}")


if __name__ == "__main__":
    main()
