from scipy.stats import spearmanr, pearsonr
from BaseModel import ScoreModel, UnderstandModel
import argparse
import json
import os

def evaluate_SIQA_U(predicted_data):
    correct = {"yes-or-no": 0, "what": 0, "how": 0}
    total = {"yes-or-no": 0, "what": 0, "how": 0}

    for item in predicted_data:
        q_type = item["type"]
        if q_type not in total:
            continue  # skip unknown types
        gt = item["answer"].strip().upper()
        pred = item[args.model_name].strip().upper()

        total[q_type] += 1
        if gt == pred:
            correct[q_type] += 1

    acc = {}
    for t in total:
        acc[t] = correct[t] / total[t] if total[t] > 0 else 0.0

    score_u = 0.2 * acc.get("yes-or-no", 0) + 0.3 * acc.get("what", 0) + 0.5 * acc.get("how", 0)
    return {
        "ACC_yes-or-no": acc.get("yes-or-no", 0),
        "ACC_what": acc.get("what", 0),
        "ACC_how": acc.get("how", 0),
        "SIQA_U_Score": score_u
    }


def evaluate_SIQA_S(predicted_data, model_name):
    gt_perception = []
    pred_perception = []
    gt_knowledge = []
    pred_knowledge = []

    for item in predicted_data:
        # Ground truth
        gt_p = item.get("perception_raing")
        gt_k = item.get("knowledge_rating")
        # Prediction
        pred_dict = item.get(model_name, {})
        pred_p = pred_dict.get("perception")
        pred_k = pred_dict.get("knowledge")

        if all(v is not None for v in [gt_p, gt_k, pred_p, pred_k]):
            gt_perception.append(gt_p)
            pred_perception.append(pred_p)
            gt_knowledge.append(gt_k)
            pred_knowledge.append(pred_k)

    if len(gt_perception) == 0:
        return {"SIQA_S_Score": 0.0, "Perceptual": 0.0, "Factual": 0.0}

    import math
    perception_pair = [(g, p) for g, p in zip(gt_perception, pred_perception) if math.isfinite(p)]
    knowledge_pair = [(g, k) for g, k in zip(gt_knowledge, pred_knowledge) if math.isfinite(k)]

    pred_perception = [p for _, p in perception_pair]
    gt_perception = [g for g, _ in perception_pair]
    pred_knowledge = [k for _, k in knowledge_pair]
    gt_knowledge = [g for g, _ in knowledge_pair]

    # Perceptual
    srcc_p, _ = spearmanr(gt_perception, pred_perception)
    plcc_p, _ = pearsonr(gt_perception, pred_perception)
    score_p = max((srcc_p + plcc_p) / 2, 0) * 100

    # Knowledge
    srcc_k, _ = spearmanr(gt_knowledge, pred_knowledge)
    plcc_k, _ = pearsonr(gt_knowledge, pred_knowledge)
    score_k = max((srcc_k + plcc_k) / 2, 0) * 100

    final_score_s = (score_p + score_k) / 2

    return {
        "Perceptual_SRCC": float(srcc_p),
        "Perceptual_PLCC": float(plcc_p),
        "Perceptual_Score": score_p,
        "Knowledge_SRCC": float(srcc_k),
        "Knowledge_PLCC": float(plcc_k),
        "Factual_Score": score_k,
        "SIQA_S_Score": final_score_s
    }


def main(args):
    SIQA_U = args.SIQA_U
    SIQA_S = args.SIQA_S
    model_name = args.model_name
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, model_name), exist_ok=True)

    with open(args.input_SIQA_U) as f:
        SIQA_U_DATASET = [json.loads(line) for line in f if line.strip()]

    with open(args.input_SIQA_S) as f:
        SIQA_S_DATASET = [json.loads(line) for line in f if line.strip()]

    Predict_SIQA_U = []
    u_output = os.path.join(output_dir, model_name, "SIQA-U.json")
    if SIQA_U and not os.path.exists(u_output):
        Understander = UnderstandModel.from_pretrained(args.model)
        for item in SIQA_U_DATASET:
            image_path = item["image_path"]
            image_path = os.path.join(args.root, image_path)
            question = item["question"]
            option = item["option"]
            answer = Understander.predict_answer(image_path, question, option)
            new_item = item.copy()
            new_item[model_name] = answer[0]  # Assuming answer is a list and we take the first one
            Predict_SIQA_U.append(new_item)

        # Save intermediate U predictions
        with open(u_output, "w", encoding="utf-8") as f:
            json.dump(Predict_SIQA_U, f, indent=2, ensure_ascii=False)
        print(f"✅ SIQA-U predictions saved to {u_output}")
    elif SIQA_U:
        with open(u_output, "r", encoding="utf-8") as f:
            Predict_SIQA_U = json.load(f)
        print(f"✅ SIQA-U predictions loaded from {u_output}")

    Predict_SIQA_S = []
    s_output = os.path.join(output_dir, model_name, "SIQA-S.json")
    if SIQA_S and not os.path.exists(s_output):
        Scorer = ScoreModel.from_pretrained(args.model)
        for item in SIQA_S_DATASET:
            image_path = item["image_path"]
            image_path = os.path.join(args.root, image_path)
            perception = Scorer.predict_score(image_path, "perception")
            knowledge = Scorer.predict_score(image_path, "knowledge")
            new_item = item.copy()
            new_item[model_name] = {}
            new_item[model_name]["perception"] = perception
            new_item[model_name]["knowledge"] = knowledge
            Predict_SIQA_S.append(new_item)
        # Save intermediate S predictions
        with open(s_output, "w", encoding="utf-8") as f:
            json.dump(Predict_SIQA_S, f, indent=2, ensure_ascii=False)
        print(f"✅ SIQA-S predictions saved to {s_output}")
    elif SIQA_S:
        with open(s_output, "r", encoding="utf-8") as f:
            Predict_SIQA_S = json.load(f)
        print(f"✅ SIQA-S predictions loaded from {s_output}")


    # Evaluation
    results = {}

    if args.SIQA_U and Predict_SIQA_U:
        u_metrics = evaluate_SIQA_U(Predict_SIQA_U)
        results.update({"SIQA_U": u_metrics})
        print("\n📊 SIQA-U Results:")
        for k, v in u_metrics.items():
            print(f"  {k}: {v:.4f}")

    if args.SIQA_S and Predict_SIQA_S:
        s_metrics = evaluate_SIQA_S(Predict_SIQA_S, model_name)
        results.update({"SIQA_S": s_metrics})
        print("\n📊 SIQA-S Results:")
        print(f"  Perceptual Score: {s_metrics['Perceptual_Score']:.2f}")
        print(f"  Factual Score:    {s_metrics['Factual_Score']:.2f}")
        print(f"  SIQA-S Final:     {s_metrics['SIQA_S_Score']:.2f}")

    # Save final results
    result_output = os.path.join(output_dir, model_name, "results.json")
    with open(result_output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Final evaluation results saved to {result_output}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the SIQA results.")
    parser.add_argument("--input", type=str, default="outputs/", help="Path to Evaluate")
    parser.add_argument("--output", type=str, default="Predict/", help="Output Path")
    parser.add_argument("--root", type=str, default="images_root/", help="Root of the images")
    parser.add_argument("--model", type=str, default="Qwen2.5-VL-7B-Instruct", help="your LLM model path")
    parser.add_argument("--model_name", type=str, default="The_Best_IQA", help="Your Fashion Model Name")
    parser.add_argument("--SIQA_U", action="store_true", help="Evaluate the SIQA-U results.")
    parser.add_argument("--SIQA_S", action="store_true", help="Evaluate the SIQA-S results.")
    args = parser.parse_args()

    # For testing, you can directly set the arguments here instead of using command line
    args.input_SIQA_U = "TrainSet/SIQA-U-valid.jsonl"
    args.input_SIQA_S = "TrainSet/SIQA-S-valid.jsonl"
    args.output = "results/"
    args.root = "TrainSet/"
    args.model = "Salesforce/blip2-opt-2.7b" # "Qwen/Qwen3-VL-2B-Instruct", "OpenGVLab/InternVL3_5-2B-HF"
    args.model_name = args.model.split("/")[-1]
    args.SIQA_U = True
    args.SIQA_S = True

    main(args)
