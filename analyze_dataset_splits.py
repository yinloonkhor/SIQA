"""Report the actual training/validation/test pool sizes for both stages.

Stage 1 — `train_domain_lora.py` on M-Paper applies a 3-layer filter:
  L1  filter_jsonl_by_image_health      (drops by image health + pixel cap + aspect)
  L2  MPaperSftDataset._inspect_item    (drops by conversation/structure;
                                         require_images=True drops `no_images` rows)
  L3  per-`task_type` subsample         (max(1, ceil(N · data_fraction)) per task)

  The script sweeps L3 over data_fraction ∈ {0.1, 0.4, 1.0} so reviewers can see
  the effective training pool at every commonly used setting. The val pool is
  fixed (data_fraction=1.0, apply_fraction=False) regardless of the train knob.

Stage 2 — `train_lora_full_from_domain.py` on SIQA has NO row-level filter.
  `image_max_pixels` is a runtime image-resize cap, never a sample drop. The
  per-epoch sample count is governed by FractionalFamilySampler:
    SIQA-S: 16,800 examples/epoch (8,400 raw rows × 2 perception/knowledge tasks)
    SIQA-U: max(1, ceil(N_type · siqa_u_epoch_multiplier)) per question type,
            summed over {yes-or-no, what, how}.

  The script sweeps siqa_u_epoch_multiplier ∈ {0.1, 0.5, 1.0} and reports
  per-type quotas + grand totals.

Validation/test sections report the fixed eval pools used by
`evaluate_siqa_s_model` / `evaluate_siqa_u_model` and the inference scripts.

Run: python analyze_dataset_splits.py
"""

from __future__ import annotations

import ast
import json
import math
from collections import Counter
from pathlib import PurePosixPath, Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
SFT_ROOT = REPO_ROOT / "M-Paper" / "sft"
TRAINER_PATH = REPO_ROOT / "train_domain_lora.py"

IMAGE_PLACEHOLDER = "<image>"
ROLE_ALIASES = {
    "assistant": "assistant", "bot": "assistant", "gpt": "assistant",
    "human": "user", "model": "assistant", "system": "system", "user": "user",
}


def load_known_broken_paths(trainer_source: Path) -> frozenset[str]:
    """Pull KNOWN_BROKEN_IMAGE_PATHS literal out of train_domain_lora.py via AST.

    Avoids importing the trainer module (which pulls in torch, accelerate, peft).
    """
    tree = ast.parse(trainer_source.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "KNOWN_BROKEN_IMAGE_PATHS":
                    value = node.value
                    # Source is `frozenset({...})` — unwrap the Call to its set literal arg.
                    if isinstance(value, ast.Call) and value.args:
                        return frozenset(ast.literal_eval(value.args[0]))
                    return frozenset(ast.literal_eval(value))
    return frozenset()


KNOWN_BROKEN_IMAGE_PATHS = load_known_broken_paths(TRAINER_PATH)


def normalize_role(raw_role: Any) -> str | None:
    if not isinstance(raw_role, str):
        return None
    return ROLE_ALIASES.get(raw_role.strip().lower())


def normalize_image_path_key(image_path: str) -> str:
    return str(PurePosixPath(image_path.replace("\\", "/")))


def coarse_bucket(task_type: str) -> str:
    """Match the per-task file partition: cap, analysis, outline."""
    if "outline_to_analysis" in task_type or "analysis" in task_type:
        return "analysis"
    if "cap" in task_type:
        return "cap"
    if "outline" in task_type:
        return "outline"
    return "other"


def count_nonblank(path: Path) -> int:
    n = 0
    with open(path, "rb") as handle:
        for line in handle:
            if line.strip():
                n += 1
    return n


def rollup(path: Path) -> tuple[int, Counter, Counter]:
    """Read a JSONL and return (total_rows, coarse_buckets, fine_task_types)."""
    coarse: Counter = Counter()
    fine: Counter = Counter()
    total = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tt = obj.get("task_type", "<missing>")
            fine[tt] += 1
            coarse[coarse_bucket(tt)] += 1
            total += 1
    return total, coarse, fine


def inspect_item(item: dict[str, Any], require_images: bool = True) -> tuple[bool, str, str]:
    """Mirror MPaperSftDataset._inspect_item from train_domain_lora.py:598-642.

    Returns (is_valid, task_type, drop_reason).
    """
    task_type = str(item.get("task_type") or "unknown_task")
    image_paths = item.get("image") or []

    if require_images and not image_paths:
        return False, task_type, "no_images"
    if not isinstance(image_paths, list) or any(
        not isinstance(p, str) or not p for p in image_paths
    ):
        return False, task_type, "invalid_image_paths"
    if any(normalize_image_path_key(p) in KNOWN_BROKEN_IMAGE_PATHS for p in image_paths):
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
    if require_images and prompt_placeholder_count != len(image_paths):
        return False, task_type, "image_placeholder_mismatch"

    return True, task_type, ""


def apply_layer_2(path: Path, require_images: bool) -> tuple[int, Counter, Counter, Counter]:
    """Apply MPaperSftDataset._inspect_item-equivalent checks. Returns
    (kept_total, kept_coarse, kept_fine, drop_reasons)."""
    kept = 0
    coarse: Counter = Counter()
    fine: Counter = Counter()
    drops: Counter = Counter()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                drops["empty_line"] += 1
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                drops["invalid_json"] += 1
                continue
            ok, tt, reason = inspect_item(item, require_images=require_images)
            if not ok:
                drops[reason] += 1
                continue
            kept += 1
            fine[tt] += 1
            coarse[coarse_bucket(tt)] += 1
    return kept, coarse, fine, drops


def apply_layer_3(fine_counts: Counter, data_fraction: float = 0.4) -> tuple[int, Counter]:
    """Mirror _select_offsets: ceil(N * fraction) per task_type, min 1.

    See train_domain_lora.py:660: sample_count = max(1, ceil(len*frac)).
    """
    eff: Counter = Counter()
    for tt, n in fine_counts.items():
        if data_fraction >= 1.0:
            eff[tt] = n
        else:
            eff[tt] = min(n, max(1, int(math.ceil(n * data_fraction))))
    by_bucket: Counter = Counter()
    for tt, n in eff.items():
        by_bucket[coarse_bucket(tt)] += n
    return sum(eff.values()), by_bucket


LABEL_WIDTH = 32


def fmt_row(label: str, total: int, coarse: Counter, retention: str = "") -> str:
    return (
        f"{label:<{LABEL_WIDTH}} "
        f"{total:>10,} "
        f"{coarse['cap']:>9,} "
        f"{coarse['analysis']:>9,} "
        f"{coarse['outline']:>9,} "
        f"{retention:>11}"
    )


def fmt_drop_reasons(drops: Counter) -> str:
    return ", ".join(f"{reason}={count:,}" for reason, count in drops.most_common())


def section(title: str) -> None:
    bar = "=" * 80
    print(f"\n{bar}\n  {title}\n{bar}")


def main() -> None:
    section("Stage 1 — M-Paper (train_domain_lora.py)")
    print(f"  sft root:                  {SFT_ROOT}")
    print(f"  known broken image paths:  {len(KNOWN_BROKEN_IMAGE_PATHS)} entries\n")

    header = (
        f'{"stage / layer":<{LABEL_WIDTH}} {"total":>10} {"cap":>9} '
        f'{"analysis":>9} {"outline":>9} {"vs prev":>11}'
    )
    print(header)
    print("-" * len(header))

    # ----- TRAIN split -----
    raw_train = SFT_ROOT / "3tasks_train.jsonl"
    f702_train = SFT_ROOT / "3tasks_train.filtered_maxpixels_702464.jsonl"
    f1m_train = SFT_ROOT / "3tasks_train.filtered_maxpixels_1003520.jsonl"

    n0, c0, _ = rollup(raw_train)
    print(fmt_row("TRAIN  L0 raw", n0, c0, "100.00%"))

    n1, c1, fine1 = rollup(f702_train)
    print(fmt_row("       L1 filter ≤702,464 px", n1, c1, f"{100*n1/n0:.2f}%"))

    n2, c2, fine2, drops2 = apply_layer_2(f702_train, require_images=True)
    print(fmt_row("       L2 dataset re-validate", n2, c2, f"{100*n2/n1:.2f}%"))
    if drops2:
        print(f"         └─ drop reasons: {fmt_drop_reasons(drops2)}")

    # L3 sweeps over data_fraction. Val is hard-coded data_fraction=1.0 with
    # apply_fraction=False (train_domain_lora.py:1199-1202), so the val pool
    # stays fixed at the L2 count regardless of the train fraction.
    for fraction in (0.1, 0.4, 1.0):
        n3, c3 = apply_layer_3(fine2, data_fraction=fraction)
        label = f"       L3 data_fraction={fraction:.1f}"
        row = fmt_row(label, n3, c3, f"{100*n3/n2:.2f}%")
        if fraction == 0.4:
            row += "  (default)"
        print(row)

    # Alternative pixel cap (≤1,003,520 px) — same filter, different threshold.
    n1b, c1b, _ = rollup(f1m_train)
    print(fmt_row("  alt: L1 filter ≤1,003,520 px", n1b, c1b, f"{100*n1b/n0:.2f}%"))

    # ----- VAL split -----
    print()
    raw_val = SFT_ROOT / "3tasks_val.jsonl"
    f702_val = SFT_ROOT / "3tasks_val.filtered_maxpixels_702464.jsonl"
    f1m_val = SFT_ROOT / "3tasks_val.filtered_maxpixels_1003520.jsonl"

    nv0, cv0, _ = rollup(raw_val)
    print(fmt_row("VAL    L0 raw", nv0, cv0, "100.00%"))

    nv1, cv1, fine_v1 = rollup(f702_val)
    print(fmt_row("       L1 filter ≤702,464 px", nv1, cv1, f"{100*nv1/nv0:.2f}%"))

    # train_domain_lora.py:1201 passes require_images=True for the val dataset too.
    nv2, cv2, fine_v2, drops_v2 = apply_layer_2(f702_val, require_images=True)
    print(fmt_row("       L2 dataset re-validate", nv2, cv2, f"{100*nv2/nv1:.2f}%"))
    if drops_v2:
        print(f"         └─ drop reasons: {fmt_drop_reasons(drops_v2)}")
    print(
        f"         └─ val pool fixed at {nv2:,} for any train data_fraction "
        "(apply_fraction=False)"
    )

    nv1b, cv1b, _ = rollup(f1m_val)
    print(fmt_row("  alt: L1 filter ≤1,003,520 px", nv1b, cv1b, f"{100*nv1b/nv0:.2f}%"))

    # ----- TEST split (no filtered variant on disk; trainer never reads it) -----
    print()
    raw_test = SFT_ROOT / "3tasks_test.jsonl"
    nt0, ct0, _ = rollup(raw_test)
    print(fmt_row("TEST   L0 raw", nt0, ct0, "100.00%"))
    print("         └─ test split is held out; trainer never reads it")

    # ----- Sanity: per-task files = bucketed rollup of raw splits -----
    print("\n  Sanity check (per-task jsonl files = bucketed rollup of combined splits):")
    sanity_header = f"    {'file':<26} {'train':>10} {'val':>8} {'test':>8}"
    print(sanity_header)
    print("    " + "-" * (len(sanity_header) - 4))
    for tag in ("cap", "analysis", "outline"):
        cells = []
        for split in ("train", "val", "test"):
            p = SFT_ROOT / f"{tag}_{split}.jsonl"
            cells.append(f"{count_nonblank(p):,}" if p.exists() else "-")
        print(
            f"    {tag + '_*.jsonl':<26} "
            f"{cells[0]:>10} {cells[1]:>8} {cells[2]:>8}"
        )

    # ----- Stage 2 (SIQA) — no row filter applied -----
    section("Stage 2 — SIQA (train_lora_full_from_domain.py, no row filter)")
    train_set = REPO_ROOT / "TrainSet"
    print("  Raw row counts (jsonl on disk):")
    raw_header = f"    {'file':<22} {'rows':>10}"
    print(raw_header)
    print("    " + "-" * (len(raw_header) - 4))
    for label, name in [
        ("train_SIQA-S",   "train_SIQA-S.jsonl"),
        ("SIQA-S-valid",   "SIQA-S-valid.jsonl"),
        ("SIQA-S-test",    "SIQA-S-test.jsonl"),
        ("train_SIQA-U",   "train_SIQA-U.jsonl"),
        ("SIQA-U-valid",   "SIQA-U-valid.jsonl"),
        ("SIQA-U-test",    "SIQA-U-test.jsonl"),
    ]:
        p = train_set / name
        if p.exists():
            print(f"    {label:<22} {count_nonblank(p):>10,}")

    siqa_s_train = train_set / "train_SIQA-S.jsonl"
    siqa_u_train = train_set / "train_SIQA-U.jsonl"
    if siqa_s_train.exists() and siqa_u_train.exists():
        siqa_s_rows = count_nonblank(siqa_s_train)
        # SiqaScoreDataset emits 2 examples/row (perception + knowledge),
        # see train_lora_full.py:360-377.
        siqa_s_examples = 2 * siqa_s_rows

        # SIQA-U has a per-`type` breakdown; FractionalFamilySampler scales each
        # type independently (train_lora_full.py:891-898), so we need the split.
        siqa_u_by_type: Counter = Counter()
        with open(siqa_u_train, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                siqa_u_by_type[str(obj["type"]).strip()] += 1
        siqa_u_total = sum(siqa_u_by_type.values())

        def per_epoch(family_size: int, multiplier: float) -> int:
            # Mirrors FractionalFamilySampler._resolve_target_count
            # (train_lora_full.py:903-909): max(1, ceil(N * m)), zero-shortcut at 0.
            if multiplier == 0.0:
                return 0
            return max(1, int(math.ceil(family_size * multiplier)))

        print("\n  Per-epoch sampling (FractionalFamilySampler):")
        siqa_s_mult = 1.0  # train_lora_full_from_domain.py:150
        siqa_s_per_epoch = per_epoch(siqa_s_examples, siqa_s_mult)
        print(
            f"    SIQA-S: {siqa_s_rows:,} raw rows × 2 (perception+knowledge) "
            f"= {siqa_s_examples:,} examples;"
        )
        print(
            f"            multiplier={siqa_s_mult:.1f} "
            f"→ {siqa_s_per_epoch:,} examples / epoch"
        )

        u_types_order = sorted(siqa_u_by_type)
        u_breakdown = ", ".join(f"{t}={siqa_u_by_type[t]:,}" for t in u_types_order)
        print(f"    SIQA-U: {siqa_u_total:,} raw rows ({u_breakdown})")

        u_header = (
            f'    {"multiplier":>10}  '
            + "  ".join(f"{t:>10}" for t in u_types_order)
            + f"  {'siqa_u_total':>13}  {'+ siqa_s':>10}  {'grand_total':>13}"
        )
        print()
        print(u_header)
        print("    " + "-" * (len(u_header) - 4))
        for u_mult in (0.1, 0.5, 1.0):
            per_type = {t: per_epoch(siqa_u_by_type[t], u_mult) for t in u_types_order}
            u_sum = sum(per_type.values())
            grand = u_sum + siqa_s_per_epoch
            row_cells = "  ".join(f"{per_type[t]:>10,}" for t in u_types_order)
            note = "  (default)" if u_mult == 0.5 else ""
            print(
                f"    {u_mult:>10.1f}  {row_cells}  {u_sum:>13,}  "
                f"{siqa_s_per_epoch:>10,}  {grand:>13,}{note}"
            )

    # ----- SIQA validation (and test) — no row filter, no multiplier -----
    print("\n  SIQA validation / test (per-epoch eval + inference, no multiplier):")
    eval_specs = [
        ("SIQA-S-valid",   "SIQA-S-valid.jsonl", "score"),
        ("SIQA-S-test",    "SIQA-S-test.jsonl",  "score"),
        ("SIQA-U-valid",   "SIQA-U-valid.jsonl", "understand"),
        ("SIQA-U-test",    "SIQA-U-test.jsonl",  "understand"),
    ]
    for label, name, family in eval_specs:
        path = train_set / name
        if not path.exists():
            continue
        if family == "score":
            n_rows = count_nonblank(path)
            # SiqaScoreDataset emits 2 examples per row (perception + knowledge);
            # train_lora_full_from_domain.py uses the same SiqaScoreDataset for val/test,
            # so the eval pool is also doubled. Per-task split is exactly 50/50.
            print(
                f"    {label:<14} {2 * n_rows:>5,} examples  "
                f"({n_rows:,} rows × 2 — perception + knowledge)"
            )
        else:
            by_type: Counter = Counter()
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    by_type[str(obj["type"]).strip()] += 1
            total = sum(by_type.values())
            type_str = ", ".join(f"{t}={by_type[t]:,}" for t in sorted(by_type))
            print(f"    {label:<14} {total:>5,} examples  ({type_str})")


if __name__ == "__main__":
    main()
