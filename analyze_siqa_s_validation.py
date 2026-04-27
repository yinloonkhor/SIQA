import csv
from pathlib import Path

import torch
from tqdm.auto import tqdm

from inference_siqa_s import load_jsonl, load_score_model_from_checkpoint


def build_error_rows(dataset: list[dict], scorer, root: Path) -> list[dict]:
    rows: list[dict] = []

    for sample_index, item in enumerate(tqdm(dataset, desc="SIQA-S Validation Diagnostics"), start=1):
        image_path = root / item["image_path"]
        perception_target = float(item["perception_raing"])
        knowledge_target = float(item["knowledge_rating"])

        perception_pred = float(scorer.predict_score(str(image_path), "perception"))
        knowledge_pred = float(scorer.predict_score(str(image_path), "knowledge"))

        rows.append(
            {
                "sample_index": sample_index,
                "pid": item.get("pid", ""),
                "task": "perception",
                "image_path": item["image_path"],
                "target_score": perception_target,
                "predicted_score": perception_pred,
                "error": perception_pred - perception_target,
                "abs_error": abs(perception_pred - perception_target),
                "squared_error": (perception_pred - perception_target) ** 2,
            }
        )
        rows.append(
            {
                "sample_index": sample_index,
                "pid": item.get("pid", ""),
                "task": "knowledge",
                "image_path": item["image_path"],
                "target_score": knowledge_target,
                "predicted_score": knowledge_pred,
                "error": knowledge_pred - knowledge_target,
                "abs_error": abs(knowledge_pred - knowledge_target),
                "squared_error": (knowledge_pred - knowledge_target) ** 2,
            }
        )

    rows.sort(key=lambda row: (row["abs_error"], row["squared_error"]), reverse=True)
    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    fieldnames = [
        "sample_index",
        "pid",
        "task",
        "image_path",
        "target_score",
        "predicted_score",
        "error",
        "abs_error",
        "squared_error",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    checkpoint_dir = Path("checkpoints/qwen3-vl-2b-siqa-s-lora_lr_1e4/best")
    input_path = Path("TrainSet/SIQA-S-valid.jsonl")
    root = Path("TrainSet")
    output_path = Path("analysis/siqa_s_validation_errors.csv")
    device_map = "auto" if torch.cuda.is_available() else None

    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    dataset = load_jsonl(input_path)
    scorer = load_score_model_from_checkpoint(checkpoint_dir, device_map=device_map)
    rows = build_error_rows(dataset, scorer, root)
    write_csv(rows, output_path)

    print(f"Saved validation diagnostics to {output_path}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Input: {input_path}")
    print(f"Rows: {len(rows)}")
    if rows:
        print("Worst sample:")
        print(rows[0])


if __name__ == "__main__":
    main()
