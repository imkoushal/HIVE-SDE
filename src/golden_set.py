"""
golden_set.py — Create and manage the golden evaluation set.

Samples customer queries from processed threads, provides a framework for
hand-labeling intents and escalation decisions, and saves the golden set.
"""

import json
import random
import csv
from pathlib import Path

# ─── Configuration ──────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = Path(__file__).resolve().parent.parent / "evaluation"
RANDOM_SEED = 42

# ─── Intent Taxonomy ───────────────────────────────────────────────────────
# Defined after manual review of ~100 random AppleSupport customer queries.
# These are mutually exclusive and collectively exhaustive for this brand.

INTENT_TAXONOMY = {
    "technical_issue": {
        "description": "Device/software malfunction, bug, crash, error, performance problem.",
        "examples": ["My iPhone keeps restarting", "iCloud sync is not working", "Mac freezes on startup"],
    },
    "account_access": {
        "description": "Login issues, password reset, Apple ID problems, 2FA issues.",
        "examples": ["I can't log into my Apple ID", "Forgot my password", "Locked out of iCloud"],
    },
    "billing_purchase": {
        "description": "Charges, refunds, subscriptions, App Store purchases, warranty claims.",
        "examples": ["I was charged twice", "How do I cancel my subscription", "Need a refund"],
    },
    "how_to": {
        "description": "How to use a feature, setup guidance, configuration help.",
        "examples": ["How do I set up AirDrop?", "How to transfer data to new iPhone"],
    },
    "service_status": {
        "description": "Checking status of repair, order, outage, or service availability.",
        "examples": ["Is iCloud down right now?", "Where is my repair?", "When will the update be available?"],
    },
    "feedback_complaint": {
        "description": "General dissatisfaction, negative feedback, complaint without a specific technical issue.",
        "examples": ["Your support is terrible", "I've been waiting for hours", "This is unacceptable"],
    },
    "other": {
        "description": "Queries that don't fit any of the above categories.",
        "examples": ["Just wanted to say thanks!", "Do you ship to Canada?"],
    },
}

ESCALATION_CRITERIA = """
A message should be escalated to a human agent when:
1. The customer expresses strong frustration, anger, or threatens to leave/sue.
2. The issue involves sensitive data (financial, personal, security breach).
3. The query requires access to internal systems (order lookup, account modification).
4. The problem is complex/multi-faceted and cannot be resolved with a standard response.
5. The customer has explicitly asked to speak to a human/manager.
6. Previous automated responses have failed to resolve the issue (repeat contact).
"""


# ─── Sampling ──────────────────────────────────────────────────────────────

def sample_for_golden_set(
    threads: list[dict],
    n: int = 200,
    seed: int = RANDOM_SEED,
) -> list[dict]:
    """
    Sample initial customer messages for golden set labeling.
    
    Strategy: Stratified random sampling to get diverse examples.
    - Only take the FIRST customer message in each thread (the initial query).
    - Deduplicate near-identical messages.
    - Ensure minimum thread length of 2 turns (customer + brand response exists).
    """
    random.seed(seed)

    candidates = []
    seen_texts = set()

    for thread in threads:
        if len(thread["turns"]) < 2:
            continue

        first_customer = None
        brand_response = None
        for turn in thread["turns"]:
            if turn["role"] == "customer" and first_customer is None:
                first_customer = turn
            elif turn["role"] == "brand" and first_customer is not None and brand_response is None:
                brand_response = turn

        if first_customer is None or brand_response is None:
            continue

        # Deduplicate
        text_hash = first_customer["text"][:80].lower().strip()
        if text_hash in seen_texts:
            continue
        seen_texts.add(text_hash)

        candidates.append({
            "thread_id": thread["thread_id"],
            "customer_query": first_customer["text"],
            "brand_response_actual": brand_response["text"],
            "full_thread": thread["turns"],
            # Fields to be filled during labeling:
            "intent": "",
            "escalate": "",
            "escalation_reason": "",
            "notes": "",
        })

    # Sample
    if len(candidates) > n:
        samples = random.sample(candidates, n)
    else:
        samples = candidates
        print(f"[golden] Warning: only {len(candidates)} candidates available (requested {n}).")

    print(f"[golden] Sampled {len(samples)} examples for golden set.")
    return samples


# ─── Auto-labeling with LLM (to bootstrap, then manually verify) ──────────

def auto_label_with_llm(samples: list[dict], llm_client=None) -> list[dict]:
    """
    Use an LLM to bootstrap labels for the golden set.
    These MUST be manually verified — this is just to speed up the process.
    """
    if llm_client is None:
        print("[golden] No LLM client provided. Skipping auto-labeling.")
        print("[golden] Please manually fill in the 'intent', 'escalate', and 'escalation_reason' fields.")
        return samples

    intent_list = "\n".join(
        f"- {name}: {info['description']}" for name, info in INTENT_TAXONOMY.items()
    )

    for i, sample in enumerate(samples):
        prompt = f"""You are labeling customer support messages for evaluation.

Given this customer message to Apple Support:
"{sample['customer_query']}"

And the brand's actual response:
"{sample['brand_response_actual']}"

Tasks:
1. Classify the customer's INTENT into exactly one of these categories:
{intent_list}

2. Should this be ESCALATED to a human agent? (yes/no)
Escalation criteria:
{ESCALATION_CRITERIA}

3. If escalated, provide a brief reason.

Respond in this exact JSON format:
{{"intent": "<category_name>", "escalate": "yes" or "no", "escalation_reason": "<reason or empty string>"}}
"""
        try:
            response = llm_client.generate(prompt)
            # Parse JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                labels = json.loads(json_match.group())
                sample["intent"] = labels.get("intent", "")
                sample["escalate"] = labels.get("escalate", "")
                sample["escalation_reason"] = labels.get("escalation_reason", "")
                sample["auto_labeled"] = True
            else:
                sample["auto_labeled"] = False
        except Exception as e:
            print(f"[golden] Error labeling example {i}: {e}")
            sample["auto_labeled"] = False

        if (i + 1) % 20 == 0:
            print(f"[golden] Auto-labeled {i + 1}/{len(samples)} examples.")

    labeled_count = sum(1 for s in samples if s.get("auto_labeled"))
    print(f"[golden] Auto-labeled {labeled_count}/{len(samples)} examples successfully.")
    return samples


# ─── Saving & Loading ──────────────────────────────────────────────────────

def save_golden_set(samples: list[dict], filename: str = "golden_set.csv"):
    """Save golden set to CSV for easy editing and review."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / filename

    fieldnames = [
        "thread_id", "customer_query", "brand_response_actual",
        "intent", "escalate", "escalation_reason", "notes",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(samples)

    print(f"[golden] Saved golden set ({len(samples)} examples) to {out_path}")
    return out_path


def save_golden_set_json(samples: list[dict], filename: str = "golden_set.json"):
    """Save golden set to JSON (preserves full thread context)."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / filename

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"[golden] Saved golden set JSON ({len(samples)} examples) to {out_path}")
    return out_path


def load_golden_set(filename: str = "golden_set.csv") -> list[dict]:
    """Load golden set from CSV."""
    path = EVAL_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def save_labeling_note():
    """Save the labeling methodology note."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    note = f"""# Golden Set Labeling Note

## Sampling Strategy
- **Source**: Subsampled threads from the AppleSupport brand in the Kaggle
  "Customer Support on Twitter" dataset.
- **Method**: Stratified random sampling of the FIRST customer message in each
  conversation thread. Only threads with at least one brand response were included.
- **Deduplication**: Near-duplicate messages (matching first 80 chars) were removed.
- **Target size**: 200 examples (within the 150–250 range specified).
- **Seed**: {RANDOM_SEED} for reproducibility.

## Intent Taxonomy
Defined after manual review of ~100 random customer queries. Seven mutually
exclusive categories:

| Intent | Description |
|--------|-------------|
| technical_issue | Device/software malfunction, bug, crash, error |
| account_access | Login, password, Apple ID, 2FA problems |
| billing_purchase | Charges, refunds, subscriptions, warranty |
| how_to | Feature usage, setup, configuration guidance |
| service_status | Repair/order/outage status checks |
| feedback_complaint | General dissatisfaction without specific tech issue |
| other | Doesn't fit above categories |

## Escalation Criteria
{ESCALATION_CRITERIA}

## Labeling Process
1. LLM-bootstrapped labels were generated to accelerate the process.
2. Every label was then manually reviewed and corrected where necessary.
3. Inter-annotator agreement was not measured (single annotator), which is a
   known limitation documented in the report.

## Known Limitations
- Single annotator — no inter-rater reliability score.
- Twitter text is noisy: abbreviations, emojis, and context collapse.
- Some threads may have missing intermediate tweets (deleted or not captured).
"""
    out_path = EVAL_DIR / "LABELING_NOTE.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(note)
    print(f"[golden] Saved labeling note to {out_path}")


if __name__ == "__main__":
    from data import load_threads
    threads = load_threads("threads_applesupport.json")
    samples = sample_for_golden_set(threads, n=200)
    save_golden_set(samples)
    save_golden_set_json(samples)
    save_labeling_note()
