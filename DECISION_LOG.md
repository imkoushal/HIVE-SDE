# Decision Log

1. **Chose AppleSupport as the target brand** — High volume (~100k+ threads), diverse issue
   types, and a well-known brand voice that makes evaluation more intuitive.

2. **Defined 7 intents instead of more granular categories** — A smaller taxonomy reduces
   annotation ambiguity and improves classifier reliability. 7 categories cover ~95% of
   observed queries.

3. **Used all-MiniLM-L6-v2 for embeddings (not OpenAI embeddings)** — Free, fast, runs
   locally, and produces good enough 384-dim embeddings for our scale. Avoids API costs
   during index building.

4. **FAISS over ChromaDB** — Simpler dependency, faster startup, no server process needed.
   For 5k documents, the flat index is fast enough without approximation.

5. **Subsampled to 5,000 threads** — Balances representativeness with processing speed.
   The full dataset has millions of rows; 5k gives a diverse sample while keeping
   embedding and indexing under 2 minutes.

6. **LLM-bootstrapped golden set labels (then manually verified)** — Hand-labeling 200
   examples from scratch takes hours. LLM bootstrapping gets ~80% right, reducing the
   task to verification and correction.

7. **Single LLM call per pipeline stage (not chains/agents)** — Simpler, faster, cheaper,
   and easier to debug. Each stage is independent, making failure analysis straightforward.

8. **Used few-shot prompting over fine-tuning for intent classification** — No training
   infrastructure needed, works with any API, and for 7 intents the accuracy is
   competitive with fine-tuned models.

9. **Chose 200 golden set examples (within 150–250 range)** — Enough for meaningful
   per-class metrics while staying manageable for manual verification.

10. **Reply length capped at ~280 characters** — Matches Twitter's character limit. Forces
    concise, actionable responses rather than verbose LLM outputs.

11. **Default escalation on error** — If any pipeline component fails, we escalate rather
    than risk a bad automated response. This is the safe-by-default design.

12. **Same LLM for agent and judge** — Pragmatic choice for the assignment scope.
    Documented as a limitation. The human agreement check partially mitigates this.

13. **Filtered non-English tweets via ASCII heuristic** — Simple and fast. Misses some
    edge cases (accented English words) but avoids pulling in a heavy language detection
    library.

14. **Did not use the Banking77 dataset** — The assignment marks it optional, and its
    banking-specific intents don't map well to Apple tech support. Would add complexity
    without clear benefit for this scope.

15. **Used cosine similarity (normalized inner product) for retrieval** — Standard choice
    for sentence embeddings. L2 distance gives similar results but cosine is more
    interpretable.
