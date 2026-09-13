# Golden Set Labeling Note

## Sampling Strategy
- **Source**: Subsampled threads from the AppleSupport brand in the Kaggle
  "Customer Support on Twitter" dataset.
- **Method**: Stratified random sampling of the FIRST customer message in each
  conversation thread. Only threads with at least one brand response were included.
- **Deduplication**: Near-duplicate messages (matching first 80 chars) were removed.
- **Target size**: 200 examples (within the 150–250 range specified).
- **Seed**: 42 for reproducibility.

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

A message should be escalated to a human agent when:
1. The customer expresses strong frustration, anger, or threatens to leave/sue.
2. The issue involves sensitive data (financial, personal, security breach).
3. The query requires access to internal systems (order lookup, account modification).
4. The problem is complex/multi-faceted and cannot be resolved with a standard response.
5. The customer has explicitly asked to speak to a human/manager.
6. Previous automated responses have failed to resolve the issue (repeat contact).


## Labeling Process
1. LLM-bootstrapped labels were generated to accelerate the process.
2. Every label was then manually reviewed and corrected where necessary.
3. Inter-annotator agreement was not measured (single annotator), which is a
   known limitation documented in the report.

## Known Limitations
- Single annotator — no inter-rater reliability score.
- Twitter text is noisy: abbreviations, emojis, and context collapse.
- Some threads may have missing intermediate tweets (deleted or not captured).
