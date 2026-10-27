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

Two sets of 66 labelled cases (42 basic + 24 hard) for evaluating the eligibility check
and the end-to-end assistant:

| File | Purpose |
|---|---|
| `eligibility_dev.jsonl` | Tuning the prompt |
| `eligibility_test.jsonl` | Held out; used only to report final numbers |

Each line is an `EligibilityCase` (`src/scheme_assistant/models.py`): a case kind, a
user description, the user's state, one scheme, the correct verdict and a note
explaining it.

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

These 42 `basic` cases each turn on one clearly stated fact. 24 harder cases per set were
added on the same schemes, 6 of each kind and 8 of each verdict:

| Kind | Example |
|---|---|
| `boundary` | Age exactly 40 where the limit is 40; "mid-fifties" where the limit is 55 |
| `indirect` | ₹800 a day for 26 days against a monthly limit; 5 hectares against a 10-acre limit |
| `hinglish` | "Mujhe bachpan se polio hai aur maine abhi LLB pass kiya hai" |
| `multi_condition` | An age relaxation for SC applicants; an exception that ignores family income |

### Limitations

- 66 cases per set: one case is 1.5 points.
- A few labels are judgement calls, e.g. e096 (NET result awaited: `need_more_info`,
  though `not_eligible` is also defensible) and e111 (income "around ₹3 lakh" against a
  ₹3 lakh limit).
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
"False eligible" counts users wrongly told they qualify, the most harmful error. "No reply"
counts cases where the model returned invalid JSON three times in a row (counted as wrong).
Cost is per check at paid-tier list prices (Gemma is free only). Claude models run at
effort `medium`; Gemini models at their defaults.

| Model | Dev | Test | Test, hard cases | False eligible (dev + test) | No reply | Time | Cost per check |
|---|---|---|---|---|---|---|---|
| claude-opus-5-5 | 0.98 | **0.98** | 0.96 | 1 | 0 | 3.4 s | $0.0071 |
| gemini-3.8-flash | 0.98 | **0.98** | 0.96 | 1 | 0 | 3.8 s | $0.0012 |
| claude-haiku-5-5 | 0.97 | 0.94 | 0.96 | **0** | 0 | 2.5 s | **$0.0002** |
| claude-sonnet-5-5 | 0.97 | 0.92 | 0.88 | 1 | 0 | 2.7 s | $0.0034 |
| gemma-4-26b-a4b-it | 0.91 | 0.92 | 0.88 | 6 | 0 | 3.4 s | free |
| gemini-3.5-flash-lite | 0.92 | 0.89 | 0.83 | 3 | 0 | **1.2 s** | $0.0003 |
| gemma-4-31b-it | 0.91 | 0.82 | 0.75 | 3 | 12 | 4.5 s | free |

Findings:

- **Opus and Gemini Flash are the most accurate** (98% on both sets). Flash costs about a
  sixth of Opus per check.
- **Haiku is the best value**: 94-97% at the lowest price, and the only model with no false
  "eligible" verdicts on either set.
- **The open Gemma models are less accurate and less reliable.** Gemma 31B is right 95% of
  the time when it replies, but 12 of its 132 replies were invalid JSON even after retries.
  Gemma 26B (4B active parameters) replies reliably but gave the most false "eligible"
  verdicts (6), and one dev run had very long replies (34 s per check on average).
- **Common errors**: e111 (income "around ₹3 lakh" against a ₹3 lakh limit) is marked
  eligible by most models. On e001 every Claude model asked whether the user pays income
  tax, an exclusion the prompt says to assume; this is cautious, not harmful.
- **Run-to-run variation**: re-running Flash-Lite on the same dev cases gave 0.97 and then
  0.92. Differences of a few points between models are within this noise.

### End to end (`pipeline_<split>_<model>.json`)

`uv run python -m scheme_assistant.evaluate_pipeline --split test --model <model>`.

Each case description goes through the full assistant (hybrid search with the given state,
then eligibility checks on the top 10). A case succeeds when the user gets the right
answer: for `eligible` and `need_more_info`, the scheme is found with that verdict; for
`not_eligible`, the scheme is not shown as eligible.

| Set | Model | Success | Scheme found | Verdict correct when found | False eligible | Time | Cost per question |
|---|---|---|---|---|---|---|---|
| dev | gemini-3.5-flash-lite | 0.83 | 0.85 | 0.96 | 0 | 3.2 s | ≈ $0.003 |
| dev | gemini-3.8-flash | 0.85 | 0.85 | 0.98 | 0 | 13.8 s | ≈ $0.012 |
| test | claude-haiku-5-5 | **0.94** | 0.94 | 0.98 | **0** | 8.0 s | $0.003 |
| test | claude-opus-5-5 | **0.94** | 0.94 | 0.98 | 1 | 12.0 s | $0.096 |
| test | gemini-3.8-flash | **0.94** | 0.94 | 0.98 | 1 | 15.2 s | ≈ $0.012 |
| test | gemini-3.5-flash-lite | 0.89 | 0.94 | 0.94 | 1 | 2.9 s | ≈ $0.003 |
| test | claude-sonnet-5-5 | 0.88 | 0.94 | 0.92 | 1 | 8.2 s | $0.039 |

Each question runs 10 eligibility checks (5 in parallel). Gemini costs (≈) are estimated
from the per-check cost measured in the eligibility runs; Claude costs are measured.

**Findings:** Haiku, Opus and Flash tie at 94%; Haiku does it at 1/30 of Opus's cost
with no false "eligible" verdicts, so it is the best default for the assistant. Search is
now the bottleneck. With Flash, every failure but one or two is a
scheme that search did not retrieve; when the scheme is found, the verdict is right 98% of
the time. Search fails mostly on short, vague descriptions ("I work as a security guard
and want a monthly pension when I am old") and on dev's Hinglish cases. Flash is about
5 times slower than Flash-Lite because each question needs 10 eligibility checks.
