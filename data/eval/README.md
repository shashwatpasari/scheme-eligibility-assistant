# Search test set

`retrieval_queries.jsonl` has 40 test questions for scheme search. Each line is one
`RetrievalQuery` (see `src/scheme_assistant/models.py`):

| Field | Meaning |
|---|---|
| `query` | What the user types |
| `user_state` | Where the user lives (`null` if they don't say). Used to test the state filter. |
| `style` | `persona` (describes themselves), `hinglish`, `hindi`, or `keyword` |
| `relevant` | Scheme ids the search should find |

## How it was made

1. **Picking schemes (random, seed 42):** 5 well-known central schemes, 10 random
   central schemes, and 1 random scheme from each of 25 random states/UTs.
2. **Dropped 6** schemes aimed at organisations, not people: `aogusysuaptl`, `jrcpc`,
   `facapabgist`, `cmagy`, `evgodrsd`, `isbadp`.
3. **Writing questions:** an LLM (Claude) read each scheme and wrote one question the way a
   user would describe themselves. Most questions avoid the scheme's name and wording, so
   search has to match meaning, not just titles.
4. **6 extra questions** in Hindi, Hinglish, and keyword style for well-known schemes.

## How to score search with it

Each question was written from one target scheme, so `relevant` lists that scheme, plus
a second scheme only where it clearly fits too (APY and PM-SYM both cover the pension
questions). Other schemes may also fit a question, so measure whether the relevant schemes
are found (**Hit@k**, **MRR**), not whether every result is relevant.

## Limitations

- Only 40 questions, so small differences between search methods may be noise.
- The questions were written by an LLM, not by real users, and haven't been reviewed by a person yet.
- Tests in `tests/test_eval_queries.py` check that every relevant scheme exists and fits
  the user's state.
