"""
data.py — Data loading, cleaning, thread reconstruction, and subsampling.

Loads the Kaggle 'Customer Support on Twitter' dataset, filters for a chosen
brand, reconstructs multi-turn conversation threads, cleans text, and produces
a workable subsample for the pipeline and golden-set creation.
"""

import os
import re
import sys
import json
import random
import hashlib
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

# Fix Windows console encoding for emoji/unicode
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ─── Configuration ──────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
BRAND = "AppleSupport"  # Default brand — change as needed
RANDOM_SEED = 42

# ─── Downloading ────────────────────────────────────────────────────────────

def download_dataset() -> Path:
    """Download the Kaggle dataset using kagglehub and return the path."""
    try:
        import kagglehub
        path = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
        print(f"[data] Dataset downloaded to: {path}")
        return Path(path)
    except Exception as e:
        print(f"[data] Error downloading dataset via kagglehub: {e}")
        raise RuntimeError(
            "Automatic Kaggle download failed. Please download 'twcs.csv' manually from "
            "https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter "
            "and place it in the 'data/raw/' directory."
        ) from e



# ─── Loading ────────────────────────────────────────────────────────────────

def _pick_best_csv(candidates: list[Path]) -> Path:
    """Prefer the full dataset (twcs.csv) over sample.csv."""
    # Look for twcs.csv first
    for c in candidates:
        if "twcs" in c.name.lower():
            return c
    # Fall back to the largest CSV
    return max(candidates, key=lambda c: c.stat().st_size)

def load_raw_data(dataset_path: Path = None) -> pd.DataFrame:
    """
    Load the raw CSV. If dataset_path is None, look for it in RAW_DIR
    or attempt download.
    """
    if dataset_path is None:
        # Check common locations
        candidates = list(RAW_DIR.glob("*.csv"))
        if not candidates:
            # Try the data dir itself
            candidates = list(DATA_DIR.glob("**/*.csv"))
        if not candidates:
            print("[data] CSV not found locally — downloading via kagglehub...")
            dataset_path = download_dataset()
            candidates = list(Path(dataset_path).glob("**/*.csv"))
        if not candidates:
            raise FileNotFoundError(
                "Could not find dataset CSV. Place it in data/raw/ or ensure kagglehub works."
            )
        # Prefer twcs.csv (full dataset) over sample.csv
        csv_path = _pick_best_csv(candidates)
    else:
        csv_files = list(Path(dataset_path).glob("**/*.csv"))
        csv_path = _pick_best_csv(csv_files) if csv_files else dataset_path

    print(f"[data] Loading CSV from {csv_path} ...")
    df = pd.read_csv(csv_path, dtype=str)
    print(f"[data] Loaded {len(df):,} rows.")
    return df


# ─── Cleaning ───────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Light cleaning: collapse whitespace, remove zero-width chars."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\u200b", "").replace("\u200d", "")
    text = re.sub(r"https?://\S+", "[URL]", text)      # mask URLs
    text = re.sub(r"@\w+", "", text)                     # remove @mentions
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_english_heuristic(text: str) -> bool:
    """Quick heuristic: mostly ASCII printable → probably English."""
    if not text:
        return False
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return (ascii_count / len(text)) > 0.8


# ─── Thread Reconstruction ─────────────────────────────────────────────────

def build_threads(df: pd.DataFrame, brand: str = BRAND) -> list[dict]:
    """
    Reconstruct conversation threads for a specific brand.

    Each thread is a dict:
      {
        "thread_id": str,
        "brand": str,
        "turns": [
            {"role": "customer"|"brand", "author_id": str, "text": str, "tweet_id": str},
            ...
        ]
      }
    """
    # Identify the brand's author_id(s) by matching outbound tweets
    # Dataset columns: tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id
    print(f"[data] Columns: {list(df.columns)}")

    # Inbound = True means customer message; False = brand response
    if "inbound" in df.columns:
        df["is_inbound"] = df["inbound"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        df["is_inbound"] = True  # fallback

    # Build a lookup: tweet_id → row
    df["tweet_id"] = df["tweet_id"].astype(str)
    tweet_lookup = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Indexing tweets"):
        tweet_lookup[row["tweet_id"]] = row

    # Find brand outbound tweets (is_inbound == False)
    # Then filter those belonging to the chosen brand
    # The dataset doesn't have a "brand" column, but we can infer:
    # Brand tweets are outbound (inbound==False). We identify the brand by author_id.
    outbound = df[~df["is_inbound"]]
    brand_author_ids = set()

    # Heuristic: look for author_ids whose tweets frequently mention the brand name or
    # who have the most outbound tweets (brand accounts tweet a lot)
    # Actually, the dataset often stores the brand name in the author_id for outbound tweets.
    for aid in outbound["author_id"].unique():
        aid_str = str(aid).lower()
        if brand.lower() in aid_str or aid_str in brand.lower():
            brand_author_ids.add(aid)

    if not brand_author_ids:
        # Fallback: find the most prolific outbound author
        top_authors = outbound["author_id"].value_counts().head(20)
        print(f"[data] Top outbound authors:\n{top_authors}")
        # Ask user to pick, or just use top one
        brand_author_ids = {top_authors.index[0]}

    print(f"[data] Brand author_id(s) for '{brand}': {brand_author_ids}")

    # Now reconstruct threads by following reply chains
    # in_response_to_tweet_id links a reply to its parent
    if "in_response_to_tweet_id" in df.columns:
        df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].astype(str)
    if "response_tweet_id" in df.columns:
        df["response_tweet_id"] = df["response_tweet_id"].astype(str)

    # Build parent->children map
    children_map = defaultdict(list)
    for _, row in df.iterrows():
        parent_id = str(row.get("in_response_to_tweet_id", ""))
        if parent_id and parent_id != "nan" and parent_id != "":
            children_map[parent_id].append(row["tweet_id"])

    # Find thread roots: inbound tweets that are NOT replies to anything in the dataset
    # OR whose in_response_to_tweet_id is nan/missing
    inbound_df = df[df["is_inbound"]]
    roots = []
    for _, row in inbound_df.iterrows():
        parent = str(row.get("in_response_to_tweet_id", ""))
        if parent in ("", "nan", "None", "NaN"):
            roots.append(row["tweet_id"])

    print(f"[data] Found {len(roots):,} potential thread roots.")

    # Walk each root to build threads, keeping only threads that involve our brand
    threads = []
    for root_id in tqdm(roots, desc="Building threads"):
        turns = []
        queue = [root_id]
        visited = set()
        involves_brand = False

        while queue:
            tid = queue.pop(0)
            if tid in visited:
                continue
            visited.add(tid)

            row = tweet_lookup.get(tid)
            if row is None:
                continue

            author = str(row.get("author_id", ""))
            is_brand = author in brand_author_ids
            if is_brand:
                involves_brand = True

            text = clean_text(str(row.get("text", "")))
            if not text or not is_english_heuristic(text):
                continue

            turns.append({
                "role": "brand" if is_brand else "customer",
                "author_id": author,
                "text": text,
                "tweet_id": tid,
            })

            # Follow replies
            for child_id in children_map.get(tid, []):
                queue.append(child_id)

            # Also follow response_tweet_id if present
            resp_ids = str(row.get("response_tweet_id", ""))
            if resp_ids and resp_ids != "nan":
                for rid in resp_ids.split(","):
                    rid = rid.strip()
                    if rid:
                        queue.append(rid)

        if involves_brand and len(turns) >= 2:
            thread_id = hashlib.md5(root_id.encode()).hexdigest()[:12]
            threads.append({
                "thread_id": thread_id,
                "brand": brand,
                "turns": turns,
            })

    print(f"[data] Built {len(threads):,} threads involving {brand}.")
    return threads


# ─── Subsampling ────────────────────────────────────────────────────────────

def subsample_threads(threads: list[dict], n: int = 5000, seed: int = RANDOM_SEED) -> list[dict]:
    """Randomly subsample threads."""
    random.seed(seed)
    if len(threads) <= n:
        return threads
    return random.sample(threads, n)


# ─── Persistence ────────────────────────────────────────────────────────────

def save_threads(threads: list[dict], filename: str = "threads.json"):
    """Save threads to JSON."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(threads, f, indent=2, ensure_ascii=False)
    print(f"[data] Saved {len(threads)} threads to {out_path}")
    return out_path


def load_threads(filename: str = "threads.json") -> list[dict]:
    """Load threads from JSON."""
    path = PROCESSED_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Summary Stats ──────────────────────────────────────────────────────────

def print_summary(threads: list[dict]):
    """Print summary statistics about the thread dataset."""
    n = len(threads)
    turn_counts = [len(t["turns"]) for t in threads]
    customer_msgs = [t for thread in threads for t in thread["turns"] if t["role"] == "customer"]
    brand_msgs = [t for thread in threads for t in thread["turns"] if t["role"] == "brand"]

    print("\n" + "=" * 60)
    print(f"  Dataset Summary for brand: {threads[0]['brand'] if threads else 'N/A'}")
    print("=" * 60)
    print(f"  Total threads:         {n:,}")
    print(f"  Total customer msgs:   {len(customer_msgs):,}")
    print(f"  Total brand msgs:      {len(brand_msgs):,}")
    print(f"  Avg turns per thread:  {np.mean(turn_counts):.1f}")
    print(f"  Median turns:          {np.median(turn_counts):.0f}")
    print(f"  Max turns:             {max(turn_counts)}")
    print(f"  Min turns:             {min(turn_counts)}")

    # Sample a random thread
    sample = random.choice(threads)
    print(f"\n  --- Sample Thread (id: {sample['thread_id']}) ---")
    for turn in sample["turns"][:6]:
        role_label = "[Customer]" if turn["role"] == "customer" else "[Brand]  "
        # Sanitize text for safe printing (remove emoji/non-ASCII)
        safe_text = turn['text'][:120].encode('ascii', errors='replace').decode('ascii')
        print(f"    {role_label}: {safe_text}...")
    print("=" * 60 + "\n")


# ─── Main ───────────────────────────────────────────────────────────────────

def prepare_data(brand: str = BRAND, subsample_n: int = 5000) -> list[dict]:
    """Full data preparation pipeline."""
    # Check if already processed
    processed_file = PROCESSED_DIR / f"threads_{brand.lower()}.json"
    if processed_file.exists():
        print(f"[data] Loading pre-processed threads from {processed_file}")
        threads = load_threads(processed_file.name)
        print_summary(threads)
        return threads

    # Load, filter, build, subsample
    df = load_raw_data()
    threads = build_threads(df, brand=brand)
    threads = subsample_threads(threads, n=subsample_n)
    save_threads(threads, filename=f"threads_{brand.lower()}.json")
    print_summary(threads)
    return threads


if __name__ == "__main__":
    threads = prepare_data()
