"""
agent.py — The AI Support Agent pipeline.

Three components:
1. Intent Classifier (few-shot prompting)
2. Reply Drafter (RAG: retrieve similar past threads → generate grounded reply)
3. Escalation Engine (structured decision with reason)
"""

import json
import re
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from llm_client import LLMClient, get_client
from golden_set import INTENT_TAXONOMY, ESCALATION_CRITERIA

# ─── Configuration ──────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"

# ─── Embedding & Vector Store ──────────────────────────────────────────────

class VectorStore:
    """Simple FAISS-based vector store for RAG retrieval."""

    def __init__(self, index_path: Path = None):
        self.index_path = index_path or INDEX_DIR
        self.index = None
        self.documents = []
        self.embedder = None

    def _get_embedder(self):
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            print("[agent] Loaded embedding model: all-MiniLM-L6-v2")
        return self.embedder

    def build_index(self, threads: list[dict]):
        """
        Build a FAISS index from historical conversation threads.
        
        Each document is the first customer message in a thread, with the
        full thread stored as metadata for retrieval.
        """
        import faiss

        embedder = self._get_embedder()
        texts = []
        self.documents = []

        for thread in tqdm(threads, desc="Preparing documents"):
            # Extract the first customer message as the query text
            customer_msg = None
            brand_response = None
            for turn in thread["turns"]:
                if turn["role"] == "customer" and customer_msg is None:
                    customer_msg = turn["text"]
                elif turn["role"] == "brand" and customer_msg is not None and brand_response is None:
                    brand_response = turn["text"]

            if customer_msg and brand_response:
                texts.append(customer_msg)
                self.documents.append({
                    "customer_query": customer_msg,
                    "brand_response": brand_response,
                    "full_thread": thread["turns"][:6],  # Keep first 6 turns max
                    "thread_id": thread["thread_id"],
                })

        print(f"[agent] Embedding {len(texts)} documents...")
        embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=64)
        embeddings = np.array(embeddings, dtype="float32")

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Build index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product = cosine after normalization
        self.index.add(embeddings)

        print(f"[agent] FAISS index built: {self.index.ntotal} vectors, dim={dim}")

        # Save
        self._save()

    def _save(self):
        """Persist index and documents."""
        import faiss
        self.index_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path / "faiss.index"))
        with open(self.index_path / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)
        print(f"[agent] Index saved to {self.index_path}")

    def load(self):
        """Load persisted index."""
        import faiss
        self.index = faiss.read_index(str(self.index_path / "faiss.index"))
        with open(self.index_path / "documents.pkl", "rb") as f:
            self.documents = pickle.load(f)
        print(f"[agent] Index loaded: {self.index.ntotal} vectors")

    def search(self, query: str, k: int = 3) -> list[dict]:
        """Search for similar documents."""
        import faiss
        embedder = self._get_embedder()
        q_emb = embedder.encode([query]).astype("float32")
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                doc = self.documents[idx].copy()
                doc["similarity_score"] = float(score)
                results.append(doc)
        return results


# ─── Intent Classifier ─────────────────────────────────────────────────────

def classify_intent(query: str, llm: LLMClient) -> dict:
    """
    Classify a customer query into one of the defined intents using few-shot prompting.
    
    Returns: {"intent": str, "confidence": str}
    """
    intent_descriptions = "\n".join(
        f"- **{name}**: {info['description']} (e.g., {', '.join(info['examples'][:2])})"
        for name, info in INTENT_TAXONOMY.items()
    )

    prompt = f"""You are an intent classifier for Apple customer support on Twitter.

Classify the following customer message into EXACTLY ONE of these intents:

{intent_descriptions}

Customer message: "{query}"

Respond with ONLY this JSON (no other text):
{{"intent": "<intent_name>", "confidence": "high" or "medium" or "low"}}
"""
    try:
        result = llm.generate_json(prompt)
        # Validate intent is in taxonomy
        if result.get("intent") not in INTENT_TAXONOMY:
            result["intent"] = "other"
        return result
    except Exception as e:
        print(f"[agent] Intent classification error: {e}")
        return {"intent": "other", "confidence": "low"}


# ─── Reply Drafter (RAG) ───────────────────────────────────────────────────

def draft_reply(
    query: str,
    intent: str,
    similar_threads: list[dict],
    llm: LLMClient,
) -> str:
    """
    Draft a reply grounded in similar historical interactions.
    
    Uses RAG: retrieves similar past threads and asks the LLM to generate
    a response in the brand's voice and style.
    """
    # Build context from retrieved threads
    context_parts = []
    for i, thread in enumerate(similar_threads[:3], 1):
        context_parts.append(
            f"Example {i} (similarity: {thread.get('similarity_score', 0):.2f}):\n"
            f"  Customer: {thread['customer_query']}\n"
            f"  Brand response: {thread['brand_response']}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""You are a customer support agent for Apple on Twitter. Your job is to draft a helpful reply.

STYLE RULES:
- Match Apple Support's Twitter tone: friendly, professional, concise.
- Use 280 characters or less when possible (Twitter limit).
- Be empathetic but action-oriented.
- If you need more info, ask a specific question.
- Don't make up technical solutions — ground your response in the examples below.
- Don't use hashtags or excessive emojis.

DETECTED INTENT: {intent}

CUSTOMER MESSAGE:
"{query}"

SIMILAR PAST INTERACTIONS (use these as reference for how Apple Support responds):
{context}

Draft your reply below. Write ONLY the reply text, nothing else.
"""
    try:
        reply = llm.generate(prompt)
        # Clean up
        reply = reply.strip().strip('"').strip("'")
        # Truncate to ~280 chars if too long
        if len(reply) > 300:
            reply = reply[:280].rsplit(" ", 1)[0] + "..."
        return reply
    except Exception as e:
        print(f"[agent] Reply drafting error: {e}")
        return "We'd like to help! Please DM us with more details so we can look into this for you."


# ─── Escalation Engine ─────────────────────────────────────────────────────

def decide_escalation(query: str, intent: str, llm: LLMClient) -> dict:
    """
    Decide whether to escalate to a human agent.
    
    Returns: {"escalate": "yes"|"no", "reason": str}
    """
    prompt = f"""You are an escalation triage system for Apple customer support.

Analyze the following customer message and decide if it should be handled automatically or escalated to a human agent.

ESCALATION CRITERIA:
{ESCALATION_CRITERIA}

DETECTED INTENT: {intent}

CUSTOMER MESSAGE:
"{query}"

Consider:
- Emotional intensity (frustration, anger, threats)
- Complexity (multi-step, requires system access)
- Sensitivity (financial, security, privacy)
- Explicit request for human help

Respond with ONLY this JSON:
{{"escalate": "yes" or "no", "reason": "<brief reason for your decision>"}}
"""
    try:
        result = llm.generate_json(prompt)
        result["escalate"] = str(result.get("escalate", "no")).lower()
        if result["escalate"] not in ("yes", "no"):
            result["escalate"] = "no"
        return result
    except Exception as e:
        print(f"[agent] Escalation decision error: {e}")
        return {"escalate": "yes", "reason": "Error in classification — defaulting to escalation."}


# ─── Full Pipeline ──────────────────────────────────────────────────────────

class SupportAgent:
    """Complete AI support agent pipeline."""

    def __init__(self, llm: LLMClient = None, vector_store: VectorStore = None):
        self.llm = llm or get_client()
        self.vector_store = vector_store or VectorStore()

    def setup(self, threads: list[dict] = None):
        """Initialize the agent: build or load the vector index."""
        if (INDEX_DIR / "faiss.index").exists():
            self.vector_store.load()
        elif threads is not None:
            self.vector_store.build_index(threads)
        else:
            raise ValueError("No index found and no threads provided. Run setup with threads first.")

    def process_query(self, query: str) -> dict:
        """
        Process a single customer query through the full pipeline.
        
        Returns:
            {
                "query": str,
                "intent": str,
                "intent_confidence": str,
                "reply": str,
                "escalate": str,
                "escalation_reason": str,
                "similar_threads": list,
            }
        """
        # Step 1: Classify intent
        intent_result = classify_intent(query, self.llm)

        # Step 2: Retrieve similar threads
        similar = self.vector_store.search(query, k=3)

        # Step 3: Draft reply
        reply = draft_reply(query, intent_result["intent"], similar, self.llm)

        # Step 4: Escalation decision
        escalation = decide_escalation(query, intent_result["intent"], self.llm)

        return {
            "query": query,
            "intent": intent_result["intent"],
            "intent_confidence": intent_result.get("confidence", "unknown"),
            "reply": reply,
            "escalate": escalation["escalate"],
            "escalation_reason": escalation.get("reason", ""),
            "similar_threads": [
                {"query": s["customer_query"], "response": s["brand_response"], "score": s["similarity_score"]}
                for s in similar
            ],
        }

    def process_batch(self, queries: list[str], show_progress: bool = True, save_path: str = None) -> list[dict]:
        """Process a batch of queries with optional per-query checkpointing."""
        import json as _json
        from pathlib import Path as _Path

        # Resume from checkpoint if it exists
        results = []
        if save_path:
            sp = _Path(save_path)
            if sp.exists():
                with open(sp, "r", encoding="utf-8") as f:
                    results = _json.load(f)
                print(f"[agent] Resuming batch from checkpoint: {len(results)}/{len(queries)} done")

        start_idx = len(results)
        if start_idx >= len(queries):
            print(f"[agent] Batch already complete ({len(results)} results)")
            return results

        remaining = queries[start_idx:]
        iterator = tqdm(remaining, desc="Processing queries", initial=start_idx, total=len(queries)) if show_progress else remaining

        for query in iterator:
            result = self.process_query(query)
            results.append(result)

            # Save checkpoint every 5 queries
            if save_path and len(results) % 5 == 0:
                sp = _Path(save_path)
                sp.parent.mkdir(parents=True, exist_ok=True)
                with open(sp, "w", encoding="utf-8") as f:
                    _json.dump(results, f, indent=2, default=str)

        # Final save
        if save_path:
            sp = _Path(save_path)
            sp.parent.mkdir(parents=True, exist_ok=True)
            with open(sp, "w", encoding="utf-8") as f:
                _json.dump(results, f, indent=2, default=str)

        return results


# ─── Baselines ──────────────────────────────────────────────────────────────

class TrivialBaseline:
    """
    Trivial baseline: always predicts the most common intent,
    always escalates, uses a canned response.
    """

    def __init__(self, most_common_intent: str = "technical_issue"):
        self.most_common_intent = most_common_intent
        self.canned_reply = (
            "We'd like to help! Please send us a DM with more details "
            "so we can assist you further."
        )

    def process_query(self, query: str) -> dict:
        return {
            "query": query,
            "intent": self.most_common_intent,
            "intent_confidence": "N/A",
            "reply": self.canned_reply,
            "escalate": "yes",
            "escalation_reason": "Trivial baseline always escalates.",
            "similar_threads": [],
        }


class SimpleBaseline:
    """
    Simple baseline: zero-shot LLM classification (no RAG, no few-shot examples).
    """

    def __init__(self, llm: LLMClient = None):
        self.llm = llm or get_client()

    def process_query(self, query: str) -> dict:
        prompt = f"""You are an Apple customer support agent on Twitter.

A customer sent: "{query}"

1. What is their intent? Choose from: technical_issue, account_access, billing_purchase, how_to, service_status, feedback_complaint, other
2. Draft a brief, helpful reply (under 280 characters).
3. Should this be escalated to a human? (yes/no, with reason)

Respond in JSON:
{{"intent": "...", "reply": "...", "escalate": "yes/no", "escalation_reason": "..."}}
"""
        try:
            result = self.llm.generate_json(prompt)
            return {
                "query": query,
                "intent": result.get("intent", "other"),
                "intent_confidence": "N/A",
                "reply": result.get("reply", "Please DM us for help."),
                "escalate": str(result.get("escalate", "no")).lower(),
                "escalation_reason": result.get("escalation_reason", ""),
                "similar_threads": [],
            }
        except Exception:
            return TrivialBaseline().process_query(query)


if __name__ == "__main__":
    from data import prepare_data
    threads = prepare_data()
    agent = SupportAgent()
    agent.setup(threads)

    # Test with a sample query
    result = agent.process_query("My iPhone 14 keeps crashing when I open the camera app")
    print(json.dumps(result, indent=2))
