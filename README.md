# Hiver SDE Intern — AI Customer Support Agent & Evaluation Framework

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![Build Status](https://img.shields.io/badge/status-submission--ready-success)
![LLM Support](https://img.shields.io/badge/LLM-Groq%20%7C%20OpenAI%20%7C%20Gemini-orange)

An end-to-end, production-ready AI Customer Support Agent built for **AppleSupport** on Twitter. The system automates intent classification, drafts RAG-grounded replies, makes risk-aware escalation decisions, and includes a comprehensive evaluation framework with baseline benchmarks.

---

## 🏆 Headline Results

| Metric | RAG Pipeline | Simple Baseline | Trivial Baseline | Target / Benchmark |
|--------|:-----------:|:---------------:|:----------------:|:------------------:|
| **Intent Macro F1** | **50.6%** | 41.0% | 11.9% | Multi-class balance (+9.6% over baseline) |
| **Escalation F1** | **66.7%** | 43.8% | 48.5% | Risk-aware escalation (+22.9% over baseline) |
| **Escalation Recall** | **92.2%** | 60.9% | 100.0% | High recall for safety (catches critical issues) |
| **Reply Quality (1–5)** | **4.44** | 4.51 | — | LLM-as-Judge 3-dimension rubric |

> **Key Takeaway**: The RAG pipeline significantly outperforms baselines on intent classification (+9.6% F1) and escalation safety (+22.9% F1) while maintaining high grounded reply quality (4.44/5.0). For a deep dive into the methodology and failure analysis, see [REPORT.md](REPORT.md) and [DECISION_LOG.md](DECISION_LOG.md).

---

## 🚀 Quick Start (Reproduce Results in < 5 Minutes)

### 1. Clone & Install
```bash
git clone https://github.com/imkoushal/HIVE-SDE.git
cd HIVE-SDE
py -m pip install -r requirements.txt
```

### 2. Configure API Key
Copy the example environment file and add an API key (Groq, OpenAI, or Gemini):
```bash
cp .env.example .env
```
*Edit `.env` and set your key (e.g. `GROQ_API_KEY=gsk_...`, `OPENAI_API_KEY=sk-...`, or `GEMINI_API_KEY=AIza...`).*

### 3. Run Pipeline Steps
All processed data, golden dataset, and evaluation results are pre-built and committed in the repository so reviewers can immediately generate reports or re-run evaluations:

```bash
# Option A: Instant Report Generation (No API key needed)
py src/main.py --step report

# Option B: Run Full Evaluation Pipeline (~15-50 min depending on provider)
py src/main.py --step evaluate

# Option C: Full End-to-End Execution (Data preparation -> Golden set -> Evaluate -> Report)
py src/main.py --step all
```

---

## 🧩 Submission Deliverables & File Index

| Deliverable | File Path | Description |
|-------------|-----------|-------------|
| **Main Orchestration** | [`src/main.py`](file:///c:/Users/admin/Desktop/hive%20sde/src/main.py) | CLI entrypoint managing execution phases (`--step data\|golden\|evaluate\|report\|all`). |
| **Data Engine** | [`src/data.py`](file:///c:/Users/admin/Desktop/hive%20sde/src/data.py) | Reconstructs 5,000 AppleSupport conversation threads from Kaggle raw CSV. |
| **Golden Set Engine** | [`src/golden_set.py`](file:///c:/Users/admin/Desktop/hive%20sde/src/golden_set.py) | Bootstraps 200 representative evaluation queries across 7 intent taxonomy classes. |
| **AI Support Agent** | [`src/agent.py`](file:///c:/Users/admin/Desktop/hive%20sde/src/agent.py) | 3-stage modular agent: Intent Classifier, FAISS RAG Reply Drafter, & Escalation Engine. |
| **Unified LLM Client** | [`src/llm_client.py`](file:///c:/Users/admin/Desktop/hive%20sde/src/llm_client.py) | Multi-provider client supporting Groq, OpenAI, and Google Gemini. |
| **Evaluation Harness** | [`src/evaluate.py`](file:///c:/Users/admin/Desktop/hive%20sde/src/evaluate.py) | Calculates Macro F1, confusion matrix, LLM-as-Judge scores, & human agreement correlation. |
| **Evaluation Report** | [`REPORT.md`](file:///c:/Users/admin/Desktop/hive%20sde/REPORT.md) | Comprehensive 5-part report covering problem framing, benchmarks, failure analysis, & future roadmap. |
| **Decision Log** | [`DECISION_LOG.md`](file:///c:/Users/admin/Desktop/hive%20sde/DECISION_LOG.md) | Architectural decision record detailing 15 key technical trade-offs. |

---

## 📐 Architecture & Workflow

```
                         Customer Query (Twitter DM / Tweet)
                                         │
       ┌─────────────────────────────────┼─────────────────────────────────┐
       ▼                                 ▼                                 ▼
[Intent Classifier]           [FAISS Vector Search]            [Escalation Engine]
Few-Shot LLM (7 Classes)     Sentence-Transformers Embeddings   Rule-Aware LLM Evaluator
       │                      Top-3 Similar Threads                │
       ▼                                 │                         ▼
 Intent Label                            ▼                   Escalate: Yes/No
  (e.g., how_to)               [RAG Reply Drafter]           + Reason / Handoff Tag
                               Grounding LLM Prompt
                                         │
                                         ▼
                            Grounded Reply (≤280 Chars)
```

---

## 🔬 Evaluation & Benchmarking Methodology

1. **Golden Evaluation Dataset**:
   - 200 hand-verified queries selected via stratified sampling across 7 intent categories.
   - Ground-truth labels for intent, escalation decision, and human evaluation context.
   - Located in [`evaluation/golden_set.csv`](file:///c:/Users/admin/Desktop/hive%20sde/evaluation/golden_set.csv) and [`evaluation/golden_set.json`](file:///c:/Users/admin/Desktop/hive%20sde/evaluation/golden_set.json).

2. **Baselines Included for Comparison**:
   - **Trivial Baseline**: Majority-class prediction (`technical_issue`), fixed template reply, always escalate.
   - **Simple Baseline**: Zero-shot single LLM prompt without RAG retrieval or explicit safety rules.
   - **RAG Pipeline (Our Agent)**: Multi-stage pipeline combining vector search, few-shot classification, and dedicated escalation assessment.

3. **Evaluation Metrics**:
   - **Classification & Escalation**: Precision, Recall, Macro F1, Accuracy, and Confusion Matrices.
   - **Reply Quality (LLM-as-Judge)**: 3-axis assessment on a 1–5 scale evaluating *Helpfulness*, *Tone*, and *Groundedness*.
   - **Human Validation**: 50-sample human scoring sheet ([`evaluation/results/human_scoring_sheet.csv`](file:///c:/Users/admin/Desktop/hive%20sde/evaluation/results/human_scoring_sheet.csv)) validating LLM-as-Judge alignment.

---

## ⚙️ Configuration & Environment

Supported model providers in `.env`:

| Provider | Recommended Model | Required `.env` Variable | Notes |
|----------|-------------------|--------------------------|-------|
| **Groq** *(Default)* | `allam-2-7b` | `GROQ_API_KEY` | Free API tier available |
| **OpenAI** | `gpt-4o-mini` | `OPENAI_API_KEY` | High speed & accuracy |
| **Google Gemini** | `gemini-3.5-flash-lite` | `GEMINI_API_KEY` | Generous free quota |

---

## 💻 Tech Stack

- **Language**: Python 3.10+
- **Vector Search & Embeddings**: FAISS (`faiss-cpu`), `sentence-transformers` (`all-MiniLM-L6-v2`)
- **LLM Integrations**: `groq`, `openai`, `google-genai`
- **Data Processing & Stats**: `pandas`, `scikit-learn`, `numpy`, `scipy`

---

## 📄 License & Attribution

- **Dataset**: [Customer Support on Twitter Dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) by ThoughtVector on Kaggle.
- Developed as part of the **Hiver SDE Intern Assignment**.
