# Retrieval evaluation sets

Two sets of 40 labelled queries for evaluating scheme search:

| File | Purpose |
|---|---|
| `retrieval_dev.jsonl` | Comparing setups and choosing settings (e.g. the hybrid weight) |
| `retrieval_test.jsonl` | Held out; used only to report final numbers |

The two sets share no target schemes. Each line is a `RetrievalQuery`
(`src/scheme_assistant/models.py`):

| Field | Meaning |
|---|---|
| `query` | What the user types |
| `user_state` | Where the user lives (`null` if unstated); used by the state filter |
| `style` | `persona`, `hinglish`, `hindi` or `keyword` |
| `relevant` | Scheme ids that should be retrieved |

## Construction

**Dev set (q001–q040).** Sampled with seed 42: 5 well-known central schemes, 10 random
central schemes, and 1 random scheme from each of 25 random states/UTs. 6 schemes
aimed at organisations were dropped by hand and replaced with 6 Hindi, Hinglish and
keyword queries about well-known schemes.

**Test set (q041–q080).** Sampled with seed 2026 from schemes not used in the dev set,
keeping only those whose `beneficiary_type` includes Individual or Family. The sample
was 4 well-known central schemes, 10 random central schemes, and 1 random scheme from
each of 25 random states/UTs. 4 schemes still aimed at organisations were dropped and
replaced with 5 Hindi and keyword queries.

For both sets, an LLM (Claude) read each scheme and wrote a query in the voice of a
user describing their situation. Most queries avoid the scheme's name and wording.

## Scoring

Each query was written from one target scheme. `relevant` adds a second scheme only
where it clearly also fits. Report **Hit@k** and **MRR**:

```bash
uv run python -m scheme_assistant.evaluate --split dev
uv run python -m scheme_assistant.evaluate --split test
```

Results are written to `results/retrieval_<split>.json`.

## Limitations

- 40 queries per set: one query equals 2.5 points, so small differences may be noise.
- Queries were written by an LLM rather than real users, and have not been reviewed by hand.
- Labels are incomplete. Other schemes can also fit a query (e.g. other states'
  SC pre-matric scholarships), and retrieving them first counts as a miss.

## Results

Search setup: hybrid 70/30 + state filter, top 10. Full numbers are in `results/`.

### Search (`retrieval_<split>.json`)

| Set | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|
| dev | 0.65 | 0.90 | 0.95 | 0.75 |
| test | 0.75 | 0.88 | 0.93 | 0.80 |

### Query rewrite (`rewrite_<split>_<model>.json`)

An LLM rewrites each question into English before searching
(`uv run python -m scheme_assistant.evaluate_rewrite --split test --model <model>`).
The prompt was tuned on dev only.

| Set | Search text | Model | Hit@1 | Hit@5 | Hit@10 | MRR | Mean time |
|---|---|---|---|---|---|---|---|
| dev | original | - | 0.65 | 0.90 | 0.95 | 0.75 | - |
| dev | rewrite | gemini-3.5-flash-lite | 0.68 | 0.95 | 0.97 | 0.77 | 1.0 s |
| dev | rewrite | gemini-3.8-flash | 0.60 | 0.90 | 0.95 | 0.74 | 4.4 s |
| test | original | - | 0.75 | 0.88 | 0.93 | 0.80 | - |
| test | rewrite | gemini-3.5-flash-lite | 0.68 | 0.85 | 0.90 | 0.75 | 1.0 s |
| test | rewrite | gemini-3.8-flash | 0.72 | 0.90 | 0.93 | 0.79 | 4.4 s |

Joining the original and the rewrite into one search text did not help either.

**Finding:** rewriting does not reliably improve search. It gains a little on dev and
loses a little on test, a difference of one or two queries. bge-m3 is multilingual and
already handles Hindi and Hinglish; one Hindi test query found at rank 1 was lost after
translation. Search therefore uses the original question.

The rewrite step is kept for what search cannot do: extracting the user's state and
facts about them for the eligibility check. Across all 80 queries, gemini-3.8-flash
returned no wrong state and gemini-3.5-flash-lite one (q024, a student living in Delhi
asking about their unnamed home state, where Delhi is arguably correct).
