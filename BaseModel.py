import torch
import torch.nn as nn
from PIL import Image
from typing import Literal, Union
from transformers import AutoProcessor, AutoModelForImageTextToText


class ScoreSoftHead(nn.Module):
    def __init__(self, processor, score_token_ids):
        super().__init__()
        self.processor = processor
        self.score_token_ids = score_token_ids

        self.score_words = ["Bad", "Poor", "Fair", "Good", "Excellent"]
        self.score_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)

        self.class_token_ids = []
        all_token_ids = []
        token_id_to_class = {}

        for class_idx, word in enumerate(self.score_words):
            tids = score_token_ids.get(word, [])
            if not tids:
                raise ValueError(f"Word '{word}' has no valid token!")
            self.class_token_ids.append(tids)
            for tid in tids:
                all_token_ids.append(tid)
                token_id_to_class[tid] = class_idx

        self.register_buffer("all_token_ids", torch.tensor(all_token_ids, dtype=torch.long))
        self.register_buffer("score_values_buf", self.score_values)

        max_vocab = processor.tokenizer.vocab_size
        token_to_class_map = torch.full((max_vocab,), -1, dtype=torch.long)
        for tid, cls in token_id_to_class.items():
            token_to_class_map[tid] = cls
        self.register_buffer("token_to_class_map", token_to_class_map)

    def forward(self, logits, shift_token_ids, task=None):
        device = logits.device
        batch_size = logits.size(0)

        score_pred = torch.full((batch_size,), float('nan'), device=device, dtype=torch.float32)
        mask = torch.isin(shift_token_ids, self.all_token_ids)
        mask = (mask.cumsum(dim=1) <= 2) & mask

        # all is score, no need to mask
        # task_mask = torch.tensor(
        #     [t in ("score_sub", "score_obj") for t in task],
        #     dtype=torch.bool, device=device
        # ).unsqueeze(1).expand_as(mask)
        # final_mask = mask & task_mask

        final_mask = mask
        if final_mask.sum().item() == 0:
            return score_pred

        logits_pred = logits[final_mask]
        token_pred = shift_token_ids[final_mask]
        batch_indices = final_mask.nonzero()[:, 0]

        num_classes = len(self.score_words)
        class_logits = torch.full((logits_pred.size(0), num_classes), -1e9, device=device)
        for class_idx, tids in enumerate(self.class_token_ids):
            if tids:
                tids_tensor = torch.tensor(tids, device=device)
                class_logits[:, class_idx] = logits_pred[:, tids_tensor].max(dim=1).values

        probs = torch.softmax(class_logits, dim=-1)
        pred_scores = (probs * self.score_values_buf.to(device)).sum(dim=-1)

        for i in range(batch_indices.size(0)):
            bid = batch_indices[i].item()
            score_pred[bid] = pred_scores[i]

        return score_pred


class ScoreModel(nn.Module):
    TERMS_STR = "Bad, Poor, Fair, Good, Excellent"

    def __init__(self, base_model, processor):
        super().__init__()
        self.base_model = base_model
        self.processor = processor

        # Build score_token_id map
        score_words = ["Bad", "Poor", "Fair", "Good", "Excellent"]
        score_token_id = {}
        tokenizer = processor.tokenizer
        for word in score_words:
            candidates = []
            for form in [" " + word, word]:
                token_ids = tokenizer.encode(form, add_special_tokens=False)
                if len(token_ids) == 1 and token_ids[0] != tokenizer.unk_token_id:
                    tid = token_ids[0]
                    if tid not in candidates:
                        candidates.append(tid)
            if candidates:
                score_token_id[word] = candidates
            else:
                raise RuntimeError(f"Cannot find single-token representation for '{word}'")

        self.score_head = ScoreSoftHead(processor, score_token_id)

    @classmethod
    def from_pretrained(cls, model_name_or_path, device=None, **kwargs):
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        base_model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            device_map=device if device else "auto",
            trust_remote_code=True,
            **kwargs
        )
        model = cls(base_model, processor)
        # 获取 base_model 所在设备（处理 multi-GPU 或 CPU 情况）
        if hasattr(base_model, "device"):
            device = base_model.device
        else:
            # 对于 device_map="auto" 的情况，取第一个参数的设备
            device = next(base_model.parameters()).device

        # 将 score_head 移动到相同设备
        model.score_head = model.score_head.to(device)
        model.eval()
        return model

    def _build_prompt(self, task: str) -> tuple[str, str]:
        if task == "perception":
            instruct = f"""You are an expert in scientific image analysis. Evaluate the given image on **Subjective Quality** only:

- Consider technical quality (sharpness, lighting, legibility) and aesthetic quality (visual appeal, layout balance, information density).
- Ignore scientific correctness.

Use exactly one of these five terms: [{self.TERMS_STR}]. Respond ONLY as:

Subjective: [Quality Word]
"""
            question = "How would you rate the subjective quality of this image?"
        elif task == "knowledge":
            instruct = f"""You are an expert in scientific image analysis. Evaluate the given image on **Objective Quality** only:

- Assess scientific rigor: completeness (e.g., scale bars, axis labels, units), correctness of data, and avoidance of redundancy.
- Ignore aesthetics or technical rendering.

Use exactly one of these five terms: [{self.TERMS_STR}]. Respond ONLY as:

Objective: [Quality Word]
"""
            question = "How would you rate the objective quality of this image?"
        else:
            raise ValueError("task must be 'perception' or 'knowledge'")
        return instruct, question

    def _preprocess(self, image_path: str, task: str):
        # Load image
        image = Image.open(image_path).convert("RGB")

        instruct, question = self._build_prompt(task)

        messages = [
            {"role": "system", "content": instruct},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image", "image": image}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[[image]],
            return_tensors="pt",
            padding=False,
            truncation=False,
            max_length=4096,
        )

        # Move to same device as model
        device = next(self.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        return inputs

    @torch.no_grad()
    def predict_score(self, image_path: Union[str, list], task: Literal["perception", "knowledge"]) -> float:
        """
        End-to-end inference: image path → predicted score (1~5)
        """
        if isinstance(image_path, list):
            image_path = image_path[0]  # support legacy format

        inputs = self._preprocess(image_path, task)

        # Generate
        outputs = self.base_model.generate(
            **inputs,
            max_new_tokens=20,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )

        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs.sequences[:, input_len:]  # [1, L]
        logits = torch.stack(outputs.scores, dim=1)  # [1, L, V]
        decoded_text = self.processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

        # Predict score
        pred_tensor = self.score_head(logits, generated_ids, task=[task])  # [1]
        pred_score = pred_tensor[0].item()

        if not (1.0 <= pred_score <= 5.0):
            print(
                f"⚠️ Warning: predicted score {pred_score:.2f} is out of [1,5] range. "
                f"Decoded output: {decoded_text!r}"
            )

        return pred_score


class UnderstandModel(nn.Module):

    def __init__(self, base_model, processor):
        super().__init__()
        self.base_model = base_model
        self.processor = processor
        self.eos_token_id = self.processor.tokenizer.eos_token_id
        self.pad_token_id = self.processor.tokenizer.pad_token_id

        vqa_words = ["A", "B", "C", "D"]
        vqa_token_id = {}

        tokenizer = processor.tokenizer
        for vqa in vqa_words:
            candidates = []
            token_ids = tokenizer.encode(vqa, add_special_tokens=False)
            if len(token_ids) == 1 and token_ids[0] != tokenizer.unk_token_id:
                tid = token_ids[0]
                if tid not in candidates:
                    candidates.append(tid)
            if candidates:
                vqa_token_id[vqa] = candidates  # e.g., "Good": [1289, 4234]
        self.vqa_token_id = vqa_token_id

    @classmethod
    def from_pretrained(cls, model_name_or_path, device=None, **kwargs):
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        base_model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            device_map=device if device else "auto",
            trust_remote_code=True,
            **kwargs
        )
        model = cls(base_model, processor)
        # 获取 base_model 所在设备（处理 multi-GPU 或 CPU 情况）
        if hasattr(base_model, "device"):
            device = base_model.device
        else:
            # 对于 device_map="auto" 的情况，取第一个参数的设备
            device = next(base_model.parameters()).device

        # 将 score_head 移动到相同设备
        model.eval()
        return model

    def _build_prompt(self, question: str, option: str) -> tuple[str, str]:
        instruct = "You are an expert in scientific image analysis. Your task is to answer visual question answering (VQA) questions based on the given image. Respond with ONLY a single uppercase letter: A, B, C, or D. Do not include any explanations, punctuation, spaces, or additional characters."
        question = f"""<image>Answer the following question based on the image.
Question: {question}
Choices: {option}
Respond with ONLY one uppercase letter: A, B, C, or D.
        """
        return instruct, question

    def _preprocess(self, image_path: str, question: str, option: str):
        # Load image
        image = Image.open(image_path).convert("RGB")

        instruct, question = self._build_prompt(question, option)

        messages = [
            {"role": "system", "content": instruct},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image", "image": image}
                ]
            }
        ]

        # Use HF chat template only if this processor defines one
        if getattr(self.processor, "chat_template", None):
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = f"{instruct}\nQuestion: {question}\nAnswer:"

        inputs = self.processor(
            text=[text],
            images=[[image]],
            return_tensors="pt",
            padding=False,
            truncation=False,
            max_length=4096,
        )

        # Move to same device as model
        device = next(self.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        return inputs

    @torch.no_grad()
    def predict_answer(self, image_path: Union[str, list], question: str, option: str):
        """
        End-to-end inference: image path → predicted score (1~5)
        """
        if isinstance(image_path, list):
            image_path = image_path[0]  # support legacy format

        inputs = self._preprocess(image_path, question, option)

        # Generate
        outputs = self.base_model.generate(
            **inputs,
            max_new_tokens=20,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )

        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs.sequences[:, input_len:]  # [1, L]
        answer = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return answer
