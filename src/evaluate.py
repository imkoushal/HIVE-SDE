"""
evaluate.py — Evaluation harness.

Automated metrics for intent classification and escalation, plus an
LLM-as-a-judge rubric for reply quality. Includes human agreement
verification framework.
"""

import sys

# Fix Windows console encoding for emoji/unicode
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


import json
import csv
import os
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

from llm_client import LLMClient, get_client
from golden_set import load_golden_set, INTENT_TAXONOMY

# ─── Configuration ──────────────────────────────────────────────────────────

EVAL_DIR = Path(__file__).resolve().parent.parent / "evaluation"
RESULTS_DIR = EVAL_DIR / "results"

# ─── Intent Metrics ────────────────────────────────────────────────────────

def _safe_str(val) -> str:
    """Coerce a value to string — handles lists, dicts, etc. from LLM output."""
    if isinstance(val, str):
        return val.strip().lower()
    if isinstance(val, list):
        return str(val[0]).strip().lower() if val else "other"
    if isinstance(val, dict):
        # Try common keys
        for k in ["category", "intent", "value"]:
            if k in val:
                return _safe_str(val[k])
        return "other"
    return str(val).strip().lower()


def evaluate_intents(predictions: list[dict], golden: list[dict]) -> dict:
    """
    Evaluate intent classification against golden set.
    
    Returns dict with accuracy, macro F1, per-class metrics, and confusion matrix.
    """
    y_true = [_safe_str(g["intent"]) for g in golden]
    y_pred = [_safe_str(p["intent"]) for p in predictions]

    # Ensure all labels are valid
    valid_intents = set(INTENT_TAXONOMY.keys())
    y_true = [y if y in valid_intents else "other" for y in y_true]
    y_pred = [y if y in valid_intents else "other" for y in y_pred]

    labels = sorted(valid_intents)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    report = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Distribution info
    true_dist = Counter(y_true)
    pred_dist = Counter(y_pred)

    results = {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": {
            label: {
                "precision": round(report[label]["precision"], 4),
                "recall": round(report[label]["recall"], 4),
                "f1": round(report[label]["f1-score"], 4),
                "support": report[label]["support"],
            }
            for label in labels if label in report
        },
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "true_distribution": dict(true_dist),
        "pred_distribution": dict(pred_dist),
    }

    return results


# ─── Escalation Metrics ────────────────────────────────────────────────────

def evaluate_escalation(predictions: list[dict], golden: list[dict]) -> dict:
    """
    Evaluate escalation decisions against golden set.
    
    Returns dict with precision, recall, F1 for escalation.
    """
    y_true = [1 if _safe_str(g["escalate"]) in ("yes", "1", "true") else 0 for g in golden]
    y_pred = [1 if _safe_str(p["escalate"]) in ("yes", "1", "true") else 0 for p in predictions]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    # Breakdown
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    # Escalation rate
    true_rate = sum(y_true) / len(y_true) if y_true else 0
    pred_rate = sum(y_pred) / len(y_pred) if y_pred else 0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "true_escalation_rate": round(true_rate, 4),
        "predicted_escalation_rate": round(pred_rate, 4),
    }


# ─── LLM-as-a-Judge (Reply Quality) ───────────────────────────────────────

JUDGE_RUBRIC = """You are an expert evaluator of customer support responses. 
Rate the following AI-generated reply on three dimensions using a 1-5 scale.

CUSTOMER QUERY: "{query}"
AI-GENERATED REPLY: "{reply}"
BRAND'S ACTUAL HISTORICAL REPLY: "{actual_reply}"

Rate on these dimensions:

1. **HELPFULNESS** (1-5): Does the reply address the customer's issue? Does it provide actionable next steps?
   - 1: Completely unhelpful, ignores the issue
   - 3: Partially helpful, addresses the issue vaguely
   - 5: Excellent, directly addresses the issue with clear next steps

2. **TONE** (1-5): Does the reply match a professional, empathetic support agent's voice?
   - 1: Rude, robotic, or inappropriate
   - 3: Neutral, acceptable but not warm
   - 5: Warm, empathetic, professional — matches Apple Support's style

3. **GROUNDEDNESS** (1-5): Is the reply factually grounded? Does it avoid making up solutions?
   - 1: Fabricates information or makes false promises
   - 3: Mostly accurate but somewhat vague
   - 5: Fully grounded, doesn't hallucinate, suggests appropriate actions

Respond with ONLY this JSON:
{{"helpfulness": <int>, "tone": <int>, "groundedness": <int>, "overall": <float>, "reasoning": "<brief explanation>"}}

The "overall" score should be the average of the three scores.
"""


def judge_reply_quality(
    query: str,
    generated_reply: str,
    actual_reply: str,
    llm: LLMClient,
) -> dict:
    """Score a single reply using the LLM judge."""
    prompt = JUDGE_RUBRIC.format(
        query=query,
        reply=generated_reply,
        actual_reply=actual_reply,
    )
    try:
        result = llm.generate_json(prompt)
        # Validate scores
        for key in ["helpfulness", "tone", "groundedness"]:
            result[key] = max(1, min(5, int(result.get(key, 3))))
        result["overall"] = round(
            np.mean([result["helpfulness"], result["tone"], result["groundedness"]]), 2
        )
        return result
    except Exception as e:
        print(f"[eval] Judge error: {e}")
        return {
            "helpfulness": 3, "tone": 3, "groundedness": 3,
            "overall": 3.0, "reasoning": f"Error: {e}"
        }


def evaluate_reply_quality(
    predictions: list[dict],
    golden: list[dict],
    llm: LLMClient = None,
    max_n: int = 50,
) -> dict:
    """
    Evaluate generated replies using LLM-as-judge on a subsample.
    
    Args:
        max_n: Maximum number of examples to judge (saves API calls).
    
    Returns aggregate scores and per-example scores.
    """
    import random
    random.seed(42)

    llm = llm or get_client()
    per_example = []

    # Subsample if too many examples (saves API calls on rate-limited free tiers)
    indices = list(range(len(predictions)))
    if len(indices) > max_n:
        indices = random.sample(indices, max_n)
        print(f"[eval] Subsampling {max_n}/{len(predictions)} examples for LLM-as-judge")

    for idx in tqdm(indices, desc="Judging replies"):
        pred = predictions[idx]
        gold = golden[idx]
        scores = judge_reply_quality(
            query=pred["query"],
            generated_reply=pred["reply"],
            actual_reply=gold.get("brand_response_actual", ""),
            llm=llm,
        )
        scores["query"] = pred["query"]
        scores["generated_reply"] = pred["reply"]
        scores["index"] = idx
        per_example.append(scores)

    # Aggregate
    helpfulness = [s["helpfulness"] for s in per_example]
    tone = [s["tone"] for s in per_example]
    groundedness = [s["groundedness"] for s in per_example]
    overall = [s["overall"] for s in per_example]

    return {
        "aggregate": {
            "helpfulness": {"mean": round(np.mean(helpfulness), 2), "std": round(np.std(helpfulness), 2)},
            "tone": {"mean": round(np.mean(tone), 2), "std": round(np.std(tone), 2)},
            "groundedness": {"mean": round(np.mean(groundedness), 2), "std": round(np.std(groundedness), 2)},
            "overall": {"mean": round(np.mean(overall), 2), "std": round(np.std(overall), 2)},
            "n_judged": len(per_example),
        },
        "per_example": per_example,
    }


# ─── Human Agreement ──────────────────────────────────────────────────────

def generate_human_scoring_sheet(predictions: list[dict], golden: list[dict], n: int = 50):
    """
    Generate a CSV for human scoring of a subsample.
    The human fills in their scores, then we compare with LLM judge.
    """
    import random
    random.seed(42)

    indices = list(range(len(predictions)))
    if len(indices) > n:
        indices = random.sample(indices, n)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "human_scoring_sheet.csv"

    fieldnames = [
        "index", "customer_query", "generated_reply", "actual_reply",
        "human_helpfulness", "human_tone", "human_groundedness", "human_notes",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in indices:
            writer.writerow({
                "index": idx,
                "customer_query": predictions[idx]["query"],
                "generated_reply": predictions[idx]["reply"],
                "actual_reply": golden[idx].get("brand_response_actual", ""),
                "human_helpfulness": "",
                "human_tone": "",
                "human_groundedness": "",
                "human_notes": "",
            })

    print(f"[eval] Human scoring sheet saved to {out_path}")
    print(f"[eval] Please fill in the human_* columns with scores 1-5.")
    return out_path


def calculate_human_agreement(
    llm_scores: list[dict],
    human_scores_path: Path,
) -> dict:
    """
    Calculate agreement between LLM judge and human scores.
    Uses Pearson and Spearman correlation.
    """
    from scipy.stats import pearsonr, spearmanr

    # Load human scores
    human_data = {}
    with open(human_scores_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["index"])
            if row["human_helpfulness"]:
                human_data[idx] = {
                    "helpfulness": int(row["human_helpfulness"]),
                    "tone": int(row["human_tone"]),
                    "groundedness": int(row["human_groundedness"]),
                }

    if not human_data:
        return {"error": "No human scores found"}

    # Align LLM and human scores
    agreement = {}
    for dimension in ["helpfulness", "tone", "groundedness"]:
        llm_vals = []
        human_vals = []
        for idx, human in human_data.items():
            if idx < len(llm_scores):
                llm_vals.append(llm_scores[idx][dimension])
                human_vals.append(human[dimension])

        if len(llm_vals) >= 5:
            pearson_r, pearson_p = pearsonr(llm_vals, human_vals)
            spearman_r, spearman_p = spearmanr(llm_vals, human_vals)
            agreement[dimension] = {
                "pearson_r": round(pearson_r, 4),
                "pearson_p": round(pearson_p, 4),
                "spearman_r": round(spearman_r, 4),
                "spearman_p": round(spearman_p, 4),
                "n_samples": len(llm_vals),
            }
        else:
            agreement[dimension] = {"error": f"Not enough samples ({len(llm_vals)})"}

    return agreement


# ─── Failure Analysis ──────────────────────────────────────────────────────

def find_failures(
    predictions: list[dict],
    golden: list[dict],
    judge_scores: list[dict] = None,
    top_n: int = 5,
) -> list[dict]:
    """
    Find the worst failures across intent, escalation, and reply quality.
    """
    failures = []

    for i, (pred, gold) in enumerate(zip(predictions, golden)):
        error_score = 0
        error_types = []

        # Intent mismatch
        if pred["intent"].lower() != gold["intent"].lower():
            error_score += 2
            error_types.append(f"Intent: predicted '{pred['intent']}' vs actual '{gold['intent']}'")

        # Escalation mismatch
        pred_esc = pred["escalate"].lower() in ("yes", "1", "true")
        gold_esc = gold["escalate"].lower() in ("yes", "1", "true")
        if pred_esc != gold_esc:
            error_score += 2
            direction = "over-escalated" if pred_esc else "under-escalated"
            error_types.append(f"Escalation: {direction}")

        # Low reply quality (if judge scores available)
        if judge_scores and i < len(judge_scores):
            overall = judge_scores[i].get("overall", 3)
            if overall <= 2:
                error_score += (3 - overall)
                error_types.append(f"Reply quality: {overall}/5")

        if error_types:
            failures.append({
                "index": i,
                "query": pred["query"],
                "error_score": error_score,
                "error_types": error_types,
                "prediction": pred,
                "golden": gold,
                "judge_score": judge_scores[i] if judge_scores and i < len(judge_scores) else None,
            })

    # Sort by error severity
    failures.sort(key=lambda x: x["error_score"], reverse=True)
    return failures[:top_n]


# ─── Full Evaluation Pipeline ──────────────────────────────────────────────

def run_full_evaluation(
    agent,
    golden: list[dict],
    baselines: dict = None,
    llm_judge: LLMClient = None,
) -> dict:
    """
    Run the complete evaluation pipeline with checkpointing.
    
    Saves intermediate results after each system so a crash doesn't lose
    everything. Resumes from checkpoints if they exist.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Try to load existing checkpoint
    checkpoint_path = RESULTS_DIR / "checkpoint_results.json"
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"[eval] Resuming from checkpoint with keys: {list(results.keys())}")
    else:
        results = {}

    # Extract queries from golden set
    queries = [g["customer_query"] for g in golden]

    # ── Evaluate main agent ──
    if "pipeline" not in results:
        print("\n" + "=" * 60)
        print("  Evaluating: Main Pipeline (RAG Agent)")
        print("=" * 60)

        pred_path = RESULTS_DIR / "predictions_pipeline.json"
        predictions = agent.process_batch(queries, save_path=str(pred_path))

        results["pipeline"] = {
            "intent": evaluate_intents(predictions, golden),
            "escalation": evaluate_escalation(predictions, golden),
        }

        # Reply quality (LLM judge)
        if llm_judge:
            reply_eval = evaluate_reply_quality(predictions, golden, llm_judge)
            results["pipeline"]["reply_quality"] = reply_eval["aggregate"]
            results["pipeline"]["reply_scores_per_example"] = reply_eval["per_example"]

            # Generate human scoring sheet
            generate_human_scoring_sheet(predictions, golden)

        # Save checkpoint
        _save_checkpoint(results, checkpoint_path)
        print("[eval] [OK] Pipeline evaluation checkpointed.")
    else:
        print("[eval] Pipeline evaluation already checkpointed — skipping.")
        # Load predictions for failure analysis later
        pred_path = RESULTS_DIR / "predictions_pipeline.json"
        if pred_path.exists():
            with open(pred_path, "r", encoding="utf-8") as f:
                predictions = json.load(f)
        else:
            predictions = []

    # ── Evaluate baselines ──
    if baselines:
        for name, baseline in baselines.items():
            if name in results:
                print(f"[eval] {name} already checkpointed — skipping.")
                continue

            print(f"\n  Evaluating: {name}")
            print("-" * 40)
            
            bl_pred_path = RESULTS_DIR / f"predictions_{name}.json"
            
            # For trivial baseline (no API calls), just run directly
            if name == "trivial_baseline":
                baseline_preds = [baseline.process_query(q) for q in tqdm(queries, desc=f"Running {name}")]
            else:
                # For LLM baselines, use resumable saving
                baseline_preds = []
                if bl_pred_path.exists():
                    with open(bl_pred_path, "r", encoding="utf-8") as f:
                        baseline_preds = json.load(f)
                    print(f"[eval] Resuming {name} from {len(baseline_preds)}/{len(queries)}")
                
                start_idx = len(baseline_preds)
                for i, q in enumerate(tqdm(queries[start_idx:], desc=f"Running {name}", initial=start_idx, total=len(queries))):
                    baseline_preds.append(baseline.process_query(q))
                    if (start_idx + i + 1) % 5 == 0:
                        with open(bl_pred_path, "w", encoding="utf-8") as f:
                            json.dump(baseline_preds, f, indent=2, default=str)
                
                # Final save
                with open(bl_pred_path, "w", encoding="utf-8") as f:
                    json.dump(baseline_preds, f, indent=2, default=str)

            results[name] = {
                "intent": evaluate_intents(baseline_preds, golden),
                "escalation": evaluate_escalation(baseline_preds, golden),
            }

            # Only run LLM judge on simple_baseline (skip trivial to save API calls)
            if llm_judge and name == "simple_baseline":
                bl_reply = evaluate_reply_quality(baseline_preds, golden, llm_judge, max_n=30)
                results[name]["reply_quality"] = bl_reply["aggregate"]

            # Save checkpoint after each baseline
            _save_checkpoint(results, checkpoint_path)
            print(f"[eval] [OK] {name} checkpointed.")

    # ── Failure analysis ──
    if "failure_analysis" not in results and predictions:
        judge_scores = results.get("pipeline", {}).get("reply_scores_per_example", [])
        failures = find_failures(predictions, golden, judge_scores)
        results["failure_analysis"] = [
            {
                "query": f["query"],
                "error_types": f["error_types"],
                "error_score": f["error_score"],
                "prediction_intent": f["prediction"]["intent"],
                "golden_intent": f["golden"]["intent"],
            }
            for f in failures
        ]

    # ── Save final results ──
    out_path = RESULTS_DIR / "evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        serializable = json.loads(json.dumps(results, default=str))
        json.dump(serializable, f, indent=2)
    print(f"\n[eval] Results saved to {out_path}")

    # ── Print summary ──
    print_results_summary(results)

    return results


def _save_checkpoint(results: dict, path: Path):
    """Save intermediate results as a checkpoint."""
    with open(path, "w", encoding="utf-8") as f:
        serializable = json.loads(json.dumps(results, default=str))
        json.dump(serializable, f, indent=2)



def print_results_summary(results: dict):
    """Print a formatted summary of evaluation results."""
    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS SUMMARY")
    print("=" * 70)

    for system_name in ["pipeline", "trivial_baseline", "simple_baseline"]:
        if system_name not in results:
            continue

        r = results[system_name]
        label = {
            "pipeline": "[RAG] RAG Pipeline",
            "trivial_baseline": "[BL1] Trivial Baseline",
            "simple_baseline": "[BL2] Simple Baseline",
        }.get(system_name, system_name)

        print(f"\n  {label}")
        print("-" * 50)

        if "intent" in r:
            print(f"    Intent Accuracy:     {r['intent']['accuracy']:.1%}")
            print(f"    Intent Macro F1:     {r['intent']['macro_f1']:.1%}")

        if "escalation" in r:
            print(f"    Escalation F1:       {r['escalation']['f1']:.1%}")
            print(f"    Escalation Prec:     {r['escalation']['precision']:.1%}")
            print(f"    Escalation Recall:   {r['escalation']['recall']:.1%}")

        if "reply_quality" in r:
            rq = r["reply_quality"]
            print(f"    Reply Helpfulness:   {rq['helpfulness']['mean']:.2f}/5")
            print(f"    Reply Tone:          {rq['tone']['mean']:.2f}/5")
            print(f"    Reply Groundedness:  {rq['groundedness']['mean']:.2f}/5")
            print(f"    Reply Overall:       {rq['overall']['mean']:.2f}/5")

    print("\n" + "=" * 70)

    if "failure_analysis" in results:
        print("\n  TOP FAILURES:")
        for i, f in enumerate(results["failure_analysis"], 1):
            print(f"\n    {i}. \"{f['query'][:80]}...\"")
            for et in f["error_types"]:
                print(f"       ⚠ {et}")

    print()
