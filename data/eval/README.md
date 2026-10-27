# Evaluation sets

Labelled data for evaluating scheme search and the eligibility check.

## Search queries

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

### Construction

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

### Scoring

Each query was written from one target scheme. `relevant` adds a second scheme only
where it clearly also fits. Report **Hit@k** and **MRR**:

```bash
uv run python -m scheme_assistant.evaluate --split dev
uv run python -m scheme_assistant.evaluate --split test
```

Results are written to `results/retrieval_<split>.json`.

### Limitations

- 40 queries per set: one query equals 2.5 points, so small differences may be noise.
- Queries were written by an LLM rather than real users, and have not been reviewed by hand.
- Labels are incomplete. Other schemes can also fit a query (e.g. other states'
  SC pre-matric scholarships), and retrieving them first counts as a miss.

## Eligibility cases

Two sets of 42 labelled cases for evaluating the eligibility check:

| File | Purpose |
|---|---|
| `eligibility_dev.jsonl` | Tuning the prompt |
| `eligibility_test.jsonl` | Held out; used only to report final numbers |

Each line is an `EligibilityCase` (`src/scheme_assistant/models.py`): a user description,
the user's state, one scheme, the correct verdict and a note explaining it.

### Construction

14 schemes were taken from each search set (dev cases use dev schemes, test cases use test
schemes), choosing schemes whose eligibility text has checkable conditions. For each
scheme, an LLM (Claude) read the eligibility text and wrote three descriptions:

| Verdict | Rule |
|---|---|
| `eligible` | The description covers every key condition and breaks none |
| `not_eligible` | The description breaks one condition (e.g. age 47 where the limit is 40) |
| `need_more_info` | Nothing is broken, but a key condition (age, income, caste, ...) is not stated |

Routine conditions (citizenship, documents, bank account, portal registration, not
already receiving the benefit) are assumed to be met unless stated otherwise. The same
rules are given to the model in its prompt.

### Limitations

- Each case turns on one clear fact, so the set is easy; near-perfect scores mean it
  cannot yet separate strong models. Harder cases (values on a boundary, several
  conditions at once, Hinglish descriptions) are needed.
- Labels were written by an LLM and have not been reviewed by hand.
- Labels follow the dataset text only, which may differ from the official rules.

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

### Eligibility check (`eligibility_<split>_<model>.json`)

`uv run python -m scheme_assistant.evaluate_eligibility --split test --model <model>`.
"False eligible" counts users wrongly told they qualify, the most harmful error.

| Set | Model | Accuracy | False eligible | Mean time |
|---|---|---|---|---|
| dev | gemini-3.5-flash-lite | 0.98 | 0 | 1.1 s |
| dev | gemini-3.8-flash | 1.00 | 0 | 4.0 s |
| test | gemini-3.5-flash-lite | 0.93 | 0 | 1.0 s |
| test | gemini-3.8-flash | 1.00 | 0 | 3.9 s |

All of Flash-Lite's errors were cautious: an eligible user was marked `need_more_info`
because the model asked about a routine condition (portal registration, a caste
certificate number, whether a family member already applied).
