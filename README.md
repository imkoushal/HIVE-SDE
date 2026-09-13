# Hiver SDE Intern — AI Customer Support Agent

An AI support agent for **AppleSupport** on Twitter that classifies intents, drafts grounded replies via RAG, and makes escalation decisions — with a rigorous evaluation framework to prove it works.

## Headline Results

| Metric | RAG Pipeline | Simple Baseline | Trivial Baseline |
|--------|:-----------:|:---------------:|:----------------:|
| Intent Macro F1 | **50.6%** | 41.0% | 11.9% |
| Escalation F1 | **66.7%** | 43.8% | 48.5% |
| Reply Quality (1–5) | 4.44 | 4.51 | — |

> The RAG pipeline wins on the metrics that matter (Macro F1 across all intents, escalation safety) while maintaining competitive reply quality. See [REPORT.md](REPORT.md) for full analysis.

## Quick Start (Reproduce results in < 15 minutes)

```bash
# 1. Clone & install
git clone https://github.com/imkoushal/HIVE-SDE.git && cd HIVE-SDE
py -m pip install -r requirements.txt

# 2. Add your API key (one of the three options)
cp .env.example .env
# Edit .env — add GROQ_API_KEY (free), OPENAI_API_KEY, or GEMINI_API_KEY

# 3. Run evaluation + report (data & golden set are pre-built)
py src/main.py --step evaluate    # ~50 min with Groq free tier
py src/main.py --step report      # Generates REPORT.md & DECISION_LOG.md
```

### Run individual steps

```bash
py src/main.py --step data       # Download + prepare data (re-downloads from Kaggle)
py src/main.py --step golden     # Recreate golden evaluation set (requires LLM)
py src/main.py --step evaluate   # Run agent + baselines + LLM judge
py src/main.py --step report     # Generate report & decision log from saved results
py src/main.py --step all        # Full pipeline end-to-end
```

> **Note**: The `data/`, `evaluation/golden_set.*`, and `evaluation/results/` are committed so reviewers can skip straight to `--step report` to regenerate the report from saved results.

## Project Structure

```
hive-sde/
├── src/
│   ├── main.py           # Entry point — orchestrates all phases
│   ├── data.py           # Data loading, thread reconstruction, subsampling
│   ├── golden_set.py     # Golden set sampling, taxonomy, LLM labeling
│   ├── agent.py          # AI agent: intent classifier, RAG reply drafter, escalation
│   ├── llm_client.py     # Unified LLM client (Groq / OpenAI / Gemini)
│   └── evaluate.py       # Evaluation harness, LLM judge, human agreement
├── data/
│   ├── raw/              # Raw Kaggle CSV (auto-downloaded)
│   ├── processed/        # Cleaned threads JSON (3.9 MB, 5k threads)
│   └── index/            # FAISS vector index + document store
├── evaluation/
│   ├── golden_set.csv    # 200 hand-labelled examples
│   ├── golden_set.json   # Full version with thread context
│   ├── LABELING_NOTE.md  # Sampling & labeling methodology
│   └── results/          # Evaluation metrics, predictions, human scoring sheet
├── REPORT.md             # Full evaluation report (5 sections)
├── DECISION_LOG.md       # 15 non-obvious decisions with rationale
├── requirements.txt
├── .env.example
└── README.md             # This file
```

## Architecture

```
Customer Query
       │
       ├──► Intent Classifier (few-shot LLM) ──► intent label (7 classes)
       │
       ├──► Vector Search (FAISS + MiniLM) ──► top-3 similar past threads
       │         │
       │         └──► Reply Drafter (RAG LLM) ──► grounded reply (≤280 chars)
       │
       └──► Escalation Engine (LLM) ──► escalate yes/no + reason
```

**Three LLM calls per query**, each independent and auditable:
1. **Intent classification**: Few-shot prompting with 7-class taxonomy
2. **Reply drafting**: RAG-grounded reply using top-3 retrieved past interactions
3. **Escalation decision**: Rule-aware LLM decides if human handoff is needed

## Evaluation Framework

- **Golden Set**: 200 LLM-bootstrapped, manually-verified examples (intent + escalation + notes)
- **Metrics**: Accuracy, Macro F1, per-class Precision/Recall/F1, Confusion Matrix
- **LLM-as-Judge**: 3-dimension rubric (Helpfulness, Tone, Groundedness) on 1–5 scale
- **Human Agreement**: 50-example scoring sheet for Pearson/Spearman correlation
- **Baselines**:
  - *Trivial*: Majority class (`technical_issue`) + canned reply + always escalate
  - *Simple*: Zero-shot LLM, no RAG, single prompt

## Requirements

- Python 3.10+
- An API key (one of):
  - **Groq** (free tier, `allam-2-7b`) — slowest but free
  - **OpenAI** (`gpt-4o-mini`) — fastest, paid
  - **Google Gemini** (`gemini-3.5-flash-lite`) — generous free tier
- ~2 GB disk for data + embeddings
- ~50 min for full evaluation (Groq free tier), ~15 min (OpenAI)

## Dataset

- **Primary**: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (Kaggle, thoughtvector)
- **Brand**: AppleSupport (subsampled to 5,000 threads from ~100k+)
- **Class distribution**: 71.5% technical_issue, 13.5% feedback_complaint, 7.5% how_to, 5.5% other/billing/service/account
