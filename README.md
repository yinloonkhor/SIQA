# SciQNet: Two-Stage Multimodal Adaptation for Scientific Image Quality Assessment

This repository contains the training, evaluation, and inference code for **SciQNet**, our entry to the [Scientific Image Quality Assessment Challenge (SIQA)](https://siqa-competition.github.io/). SciQNet is a Qwen3-VL-2B-based multimodal model trained in two stages: (1) domain adaptation on the `M-Paper` instruction-tuning corpus, then (2) mixed LoRA fine-tuning on the two SIQA benchmark tracks:

- `SIQA-S`: score prediction for `perception` and `knowledge`
- `SIQA-U`: multiple-choice understanding / VQA with answer letters `A-D`

## Environment

The code expects Python ≥ 3.12 with PyTorch, Transformers, Accelerate, PEFT, and SciPy.

```bash
uv sync
```

Dependencies are pinned in `pyproject.toml` and locked in `uv.lock`, so `uv sync` reproduces the same environment across machines.

Install PyTorch with the CUDA build matching your device — see the [PyTorch install guide](https://pytorch.org/get-started/locally/). `torch` and `torchvision` are intentionally left out of `pyproject.toml` because the right CUDA wheel varies by machine. Example:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Then activate the environment:

```bash
source ./.venv/bin/activate
```

Verified model family in the current scripts:

- `Qwen/Qwen3-VL-2B-Instruct`

The wrappers use `AutoModelForImageTextToText` and `AutoProcessor` with `trust_remote_code=True`, so any replacement model must support that path cleanly.

## Get The Datasets

This repo expects two local dataset folders under the repository root:

- `M-Paper/` for domain adaptation
- `TrainSet/` for SIQA-S and SIQA-U training / validation / test data

The easiest way to get both is to clone them from Hugging Face.

Install Git LFS first:

```bash
git lfs install
```

Then clone the datasets:

```bash
git clone https://huggingface.co/datasets/mPLUG/M-Paper
git clone https://huggingface.co/datasets/SIQA/TrainSet
```

Important:

- `M-Paper` uses Git LFS for large files; install `git-lfs` before cloning, or you will only get small pointer files instead of the real image assets.
- The defaults in [train_domain_lora.py](train_domain_lora.py) and [inference_domain.py](inference_domain.py) assume the cloned folder is named `M-Paper/` and sits at the repo root.
- The SIQA training and evaluation scripts assume the dataset is cloned into `TrainSet/` at the repo root.

## Expected Data Layout

The repo code assumes dataset folders live under the repository root:

```text
Evaluate-Pipeline/
├── TrainSet/
│   ├── train_SIQA-S.jsonl
│   ├── SIQA-S-valid.jsonl
│   ├── SIQA-S-test.jsonl
│   ├── train_SIQA-U.jsonl
│   ├── SIQA-U-valid.jsonl
│   ├── SIQA-U-test.jsonl
│   └── ... image files referenced by image_path
├── M-Paper/
│   └── sft/
│       ├── 3tasks_train.jsonl
│       └── 3tasks_val.jsonl
└── checkpoints/
```

These data folders are not part of the tracked code in this repo and are usually kept local.

## Dataset Formats

### SIQA-S JSONL

The SIQA-S loaders in this repo currently expect this exact schema:

```json
{
  "image_path": "relative/path/to/image.png",
  "perception_raing": 4.2,
  "knowledge_rating": 3.8
}
```

Note the spelling: `perception_raing`. The code uses that key as-is.

### SIQA-U JSONL

```json
{
  "image_path": "relative/path/to/image.png",
  "question": "What does the red arrow indicate?",
  "option": "A. ... B. ... C. ... D. ...",
  "answer": "A",
  "type": "what"
}
```

Valid `type` values in the current code are:

- `yes-or-no`
- `what`
- `how`

### M-Paper SFT JSONL

The domain-adaptation pipeline expects instruction data shaped roughly like:

```json
{
  "task_type": "some_task_name",
  "image": ["relative/or/absolute/image1.png"],
  "conversations": [
    {"from": "human", "value": "<image>\nPrompt text"},
    {"from": "gpt", "value": "Target answer"}
  ]
}
```

Current validation rules in [train_domain_lora.py](train_domain_lora.py) require:

- a final assistant turn in `conversations`
- non-empty assistant target text
- the number of `<image>` placeholders in the prompt to match the number of images

## Workflow

### 1. Train a domain adapter on M-Paper

Edit the config block at the top of `main()` in [train_domain_lora.py](train_domain_lora.py), then run:

```bash
python train_domain_lora.py
```

Default paths in the file:

- train data: `M-Paper/sft/3tasks_train.jsonl`
- val data: `M-Paper/sft/3tasks_val.jsonl`
- image root: `M-Paper`
- output: `checkpoints/qwen3-vl-2b-domain-lora`

### 2. Train on SIQA-S

There are two variants in the repo:

- [train_lora_siqa_s.py](train_lora_siqa_s.py): LoRA for `SIQA-S` starting from the base model
- [train_lora_siqa_s_from_domain.py](train_lora_siqa_s_from_domain.py): LoRA for `SIQA-S` initialized from a domain adapter checkpoint

Important — only the `*_from_domain.py` variants can continue from a domain-adapter checkpoint:

- To continue `SIQA-S` training from a domain adapter, use [train_lora_siqa_s_from_domain.py](train_lora_siqa_s_from_domain.py).
- To continue mixed `SIQA-S + SIQA-U` training from a domain adapter, use [train_lora_full_from_domain.py](train_lora_full_from_domain.py).
- [train_lora_siqa_s.py](train_lora_siqa_s.py) and [train_lora_full.py](train_lora_full.py) always start from the base model and ignore any domain adapter.

After editing the config values in the relevant script, run for example:

```bash
python train_lora_siqa_s.py
```

Default SIQA-S paths used across the training scripts:

- train data: `TrainSet/train_SIQA-S.jsonl`
- val data: `TrainSet/SIQA-S-valid.jsonl`
- image root: `TrainSet`

### 3. Train a mixed SIQA-U + SIQA-S model

[train_lora_full.py](train_lora_full.py) mixes the scoring and understanding tasks in one LoRA run, starting from the base model. For mixed training initialized from the `M-Paper` domain adapter, use [train_lora_full_from_domain.py](train_lora_full_from_domain.py).

Default paths:

- `TrainSet/train_SIQA-S.jsonl`
- `TrainSet/train_SIQA-U.jsonl`
- `TrainSet/SIQA-S-valid.jsonl`
- `TrainSet/SIQA-U-valid.jsonl`

Run after updating the config block if needed:

```bash
python train_lora_full.py
```

### 4. Evaluate a model on SIQA-U / SIQA-S

[eval_pipeline.py](eval_pipeline.py) loads the validation sets, runs inference, and writes:

- `results/<model_name>/SIQA-U.json`
- `results/<model_name>/SIQA-S.json`
- `results/<model_name>/results.json`

Run:

```bash
python eval_pipeline.py
```

Important: the current `__main__` block in [eval_pipeline.py](eval_pipeline.py) overrides CLI arguments with hardcoded local test values:

- `args.input_SIQA_U = "TrainSet/SIQA-U-valid.jsonl"`
- `args.input_SIQA_S = "TrainSet/SIQA-S-valid.jsonl"`
- `args.root = "TrainSet/"`
- `args.model = "Salesforce/blip2-opt-2.7b"`
- `args.SIQA_U = True`
- `args.SIQA_S = True`

To use different inputs or a different model, edit that block, or refactor the script back to a pure CLI entry point.

### 5. Create an SIQA-S submission file

[inference_siqa_s.py](inference_siqa_s.py) loads a trained SIQA-S checkpoint and writes a submission JSON.

Default values in the script:

- checkpoint: `checkpoints/qwen3-vl-2b-siqa-s-lora_lr_1e4/best`
- input: `TrainSet/SIQA-S-test.jsonl`
- image root: `TrainSet`
- output: `submissions/DoubleY_data_SIQA-S.json`

Run:

```bash
python inference_siqa_s.py
```

### 6. Create SIQA-S and SIQA-U submission files from a mixed checkpoint

[inference_full.py](inference_full.py) loads a mixed SIQA-S + SIQA-U checkpoint once, wraps it as both a `ScoreModel` and an `UnderstandModel`, and writes both submission JSONs.

Default values in the script:

- checkpoint: `checkpoints/qwen3-vl-2b-siqa-mixed-lora_frac_05/best`
- SIQA-S input: `TrainSet/SIQA-S-test.jsonl`
- SIQA-U input: `TrainSet/SIQA-U-test.jsonl`
- image root: `TrainSet`
- SIQA-S output: `submissions/DoubleY_data_SIQA-S.json`
- SIQA-U output: `submissions/DoubleY_data_SIQA-U.json`

Run:

```bash
python inference_full.py
```

### 7. Inspect domain-adapter generations

[inference_domain.py](inference_domain.py) can:

- load a sample from `M-Paper/sft/3tasks_val.jsonl`
- print the model response
- optionally print the reference answer
- optionally save the interaction as JSON

Run:

```bash
python inference_domain.py
```

### 8. Analyze SIQA-S validation errors

[analyze_siqa_s_validation.py](analyze_siqa_s_validation.py) re-runs SIQA-S validation inference and writes sorted error rows to:

- `analysis/siqa_s_validation_errors.csv`

Run:

```bash
python analyze_siqa_s_validation.py
```

### 9. Inspect training/validation/test pool sizes

[analyze_dataset_splits.py](analyze_dataset_splits.py) prints, for both training stages, the exact row counts feeding the model:

- Stage 1 (M-Paper, [train_domain_lora.py](train_domain_lora.py)): row counts after each layer of the filtering pipeline (image-health filter, dataset re-validation, per-`task_type` subsample), swept over `data_fraction ∈ {0.1, 0.4, 1.0}`. The val pool is reported separately and shown to be invariant to the train fraction.
- Stage 2 (SIQA, [train_lora_full_from_domain.py](train_lora_full_from_domain.py)): raw row counts plus the per-epoch sample counts produced by `FractionalFamilySampler`, with `siqa_u_epoch_multiplier ∈ {0.1, 0.5, 1.0}` and the per-question-type breakdown (`yes-or-no` / `what` / `how`).
- Validation and test pools for both SIQA-S (×2 perception/knowledge expansion) and SIQA-U (per-type breakdown).

The script is read-only: it parses on-disk JSONL files without loading the model or running any training. No CLI args.

Run:

```bash
python analyze_dataset_splits.py
```

## Outputs

Typical outputs produced by the scripts:

- `checkpoints/.../best`
- `checkpoints/.../last`
- `checkpoints/.../resume_state`
- `results/<model_name>/results.json`
- `submissions/*.json`
- `analysis/*.csv`

## Notes

- Most scripts are currently configured by editing Python variables inside `main()` rather than by passing stable command-line flags.
- [BaseModel.py](BaseModel.py) is a shared module that defines `ScoreSoftHead`, `ScoreModel`, and `UnderstandModel`; the training, evaluation, and inference scripts import from it.
- `eval_pipeline.py` computes SIQA-U accuracy by question type and computes SIQA-S from SRCC and PLCC over predicted `perception` and `knowledge` scores.
- `data/tran_OpenAI.py` still exists in the repo, but it is legacy preprocessing code and is not used by the current training, evaluation, or inference scripts.

## Acknowledgements

The datasets used in this work are obtained from:

- [SIQA/TrainSet](https://huggingface.co/datasets/SIQA/TrainSet) — official training, validation, and test data for `SIQA-S` and `SIQA-U`.
- [mPLUG/M-Paper](https://huggingface.co/datasets/mPLUG/M-Paper) — instruction-tuning data used in the domain-adaptation stage.

We thank their authors and maintainers for making these resources publicly available.
