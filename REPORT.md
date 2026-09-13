# Hiver SDE Intern Assignment — Evaluation Report

## 1. Problem Framing

**Brand**: AppleSupport (Twitter)

**What "good" means for Apple Support**:
- **Fast**: Acknowledges the issue quickly with a concrete next step (e.g., "DM us", "try this setting").
- **Empathetic**: Matches Apple's warm, professional tone — avoids robotic or corporate-speak.
- **Actionable**: Provides a specific resolution path, not just "we're sorry".
- **Safe**: Escalates when unsure rather than providing wrong information.

**What we chose NOT to build**:
- Multi-turn dialogue management (we evaluate first-response quality only).
- Sentiment analysis as a standalone feature (folded into escalation logic).
- Support for non-English queries (filtered out during data preparation).
- Integration with Apple's internal systems (we simulate with historical data).

## 2. Results vs. Baselines

### Intent Classification

| Metric | RAG Pipeline | Simple Baseline | Trivial Baseline |
|--------|:---:|:---:|:---:|
| Accuracy | 63.0% | 66.0% | 71.5% |
| Macro F1 | 50.6% | 41.0% | 11.9% |

**Per-class F1 (RAG Pipeline)**:

| Intent | Precision | Recall | F1 | Support |
|--------|:---------:|:------:|:--:|:-------:|
| account_access | 40% | 100% | 57% | 2 |
| billing_purchase | 67% | 50% | 57% | 4 |
| feedback_complaint | 46% | 59% | 52% | 27 |
| how_to | 26% | 67% | 37% | 15 |
| other | 14% | 40% | 21% | 5 |
| service_status | 43% | 75% | 55% | 4 |
| technical_issue | 94% | 64% | 76% | 143 |

### Escalation Decision

| Metric | RAG Pipeline | Simple Baseline | Trivial Baseline |
|--------|:---:|:---:|:---:|
| F1 | 66.7% | 43.8% | 48.5% |
| Precision | 52.2% | 34.2% | 32.0% |
| Recall | 92.2% | 60.9% | 100.0% |

### Reply Quality (LLM-as-Judge, 1-5 scale)

| Dimension | RAG Pipeline | Simple Baseline |
|-----------|:---:|:---:|
| Helpfulness | 4.26 | 4.33 |
| Tone | 4.38 | 4.43 |
| Groundedness | 4.68 | 4.77 |
| Overall | 4.44 | 4.51 |

## 3. Failure Analysis

### Top 5 Failure Modes

**Failure 1**: "there’s another glitch🙃🙃🙃whenever I try to type “it” autocorrect changes to I.t 🙃🙃🙃next glitch I wan"
- Error types: Intent: predicted 'how_to' vs actual 'technical_issue', Escalation: over-escalated
- Predicted intent: `how_to` -> Actual: `technical_issue`
- **Hypothesis**: The query asks *how to fix* a bug, mixing instructional language with a genuine defect report. The classifier latches on to the how-to framing ('how do I...', 'when will ... be fixed') rather than recognising the underlying technical fault.

**Failure 2**: "Hate restarting my phone. I have to sit through fifty-something unread text alerts every time I rest"
- Error types: Intent: predicted 'technical_issue' vs actual 'feedback_complaint', Escalation: over-escalated
- Predicted intent: `technical_issue` -> Actual: `feedback_complaint`
- **Hypothesis**: The user vents frustration about a known issue. The presence of technical language ('restarting', 'text alerts') biases the classifier toward technical_issue, but the core intent is to complain, not to seek a fix.

**Failure 3**: "when will the auto-capitalizing “It” and I️ that changed to “A ?” be fixed?! This is super annoying!"
- Error types: Intent: predicted 'how_to' vs actual 'technical_issue', Escalation: over-escalated
- Predicted intent: `how_to` -> Actual: `technical_issue`
- **Hypothesis**: The query asks *how to fix* a bug, mixing instructional language with a genuine defect report. The classifier latches on to the how-to framing ('how do I...', 'when will ... be fixed') rather than recognising the underlying technical fault.

**Failure 4**: "is this real or a scam? Why did it come up on my iPhone? [URL]"
- Error types: Intent: predicted 'feedback_complaint' vs actual 'other', Escalation: over-escalated
- Predicted intent: `feedback_complaint` -> Actual: `other`
- **Hypothesis**: The query mentions a scam/phishing concern — an 'other' category edge case. The confused/alarmed tone reads as a complaint to the model.

**Failure 5**: "You know y’all need to get this keyboard together! It’s like reading hieroglyphics sometimes with th"
- Error types: Intent: predicted 'feedback_complaint' vs actual 'technical_issue', Escalation: over-escalated
- Predicted intent: `feedback_complaint` -> Actual: `technical_issue`
- **Hypothesis**: The query uses strongly emotional language ('y'all need to get this together!') which triggers the feedback/complaint signal, but the user is actually describing a concrete keyboard bug — a technical issue.

## 4. What Is Misleading About My Headline Number?

Several factors make the raw metrics look better (or worse) than reality:

1. **Severe class imbalance inflates accuracy**: `technical_issue` constitutes **71.5%** of the golden
   set. The trivial baseline (always predict `technical_issue`) achieves 71.5%
   accuracy — *higher* than our RAG pipeline's 63.0%. **Macro F1 is the honest
   metric**: our pipeline scores 50.6% vs. the trivial baseline's
   11.9%, which shows the pipeline
   actually classifies across all 7 intents rather than betting on the majority class.

2. **Rare classes have unreliable metrics**: `account_access` (n=2) and `billing_purchase` (n=4)
   have so few examples that their per-class F1 is essentially a coin flip. A single misclassification
   changes their F1 by 25-50%.

3. **LLM-as-judge circularity**: The same LLM family generates replies *and* judges them. This risks
   inflated quality scores (4.44/5). The human agreement scoring sheet is provided for external
   validation, but we did not complete human scoring in this iteration.

4. **Twitter noise floor**: Messages like "ugh 😤😤😤" or "this is broken" are genuinely ambiguous.
   Any honest annotator would disagree on ~15-20% of edge cases. Our single-annotator golden set
   doesn't capture this uncertainty.

5. **Escalation over-triggers**: The pipeline achieves 92.2% recall but only 52.2% precision for
   escalation. Nearly half of escalations are false positives. This is *safe* (few missed escalations)
   but *expensive* in a production setting — it would flood human agents with auto-handleable queries.

## 5. What I'd Do Next (With One More Week)

1. **Multi-annotator golden set**: Get 2-3 annotators, measure inter-rater agreement (Cohen's kappa).
2. **Fine-tune embeddings**: Use contrastive learning on Apple Support threads for better retrieval.
3. **Multi-turn evaluation**: Evaluate full conversation flow, not just first-response quality.
4. **A/B test escalation thresholds**: Tune escalation sensitivity based on false-negative cost analysis.
5. **Add the Banking77 dataset**: Use it as auxiliary training data for the intent classifier.
6. **Prompt optimization**: Systematically test prompt variations with DSPy or similar frameworks.
7. **Deploy as a Streamlit demo**: Interactive UI for live testing.
