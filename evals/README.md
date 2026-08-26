# Evals

A benchmark for the RAG pipeline. It answers one question: **did that change
make the assistant better or worse?**

```bash
python assistant.py eval --retrieval-only     # ~0.2s, no LLM, fully deterministic
python assistant.py eval --no-judge           # end-to-end answers, no judge
python assistant.py eval                      # everything, incl. groundedness
python assistant.py eval --json > run.json    # machine-readable
python assistant.py eval --threshold 0.8      # exit 1 if a headline drops below 80%
```

## Why this exists

Every RAG knob — `chunk_size`, `chunk_overlap`, `rag_top_k`, the embedding
model, the prompt template — is a guess until it is measured. "Feels better
after I bumped top_k" is not a result; it is a memory of three questions you
happened to ask. Evals turn tuning from taste into arithmetic.

## Design

### The document is fictional on purpose

[`sample_docs/nimbusedge_handbook.md`](../sample_docs/nimbusedge_handbook.md)
describes a product that does not exist, made by a company that does not exist.
This is the single most important property of the whole suite.

If you evaluate RAG on a real document, a correct answer proves nothing —
llama3.2 may simply already know that HIPAA exists or that AES-256 is an
encryption standard, and you would score a working pipeline and a completely
broken one identically. With invented facts (a *26-month* warranty, certificate
`#HX7-2025-0114`, founders Priya Venkataraman and Tomas Reinholt), the only path
to a correct answer runs through retrieval.

### Two stages, measured separately

RAG fails in two ways that need opposite fixes:

| Failure | Symptom | Fix |
|---|---|---|
| **Retrieval** | The answer never made it into the prompt | chunking, `top_k`, embedding model |
| **Generation** | The right chunk *was* in the prompt; the model still blew it | prompt template, model |

A single accuracy number cannot distinguish them, so the runner scores each
stage independently and then cross-tabulates. `cross_tabulate()` in
[`runner.py`](runner.py) sorts every answerable case into `both_ok`,
`retrieval_failed`, `generation_failed`, or `both_failed`, and the report turns
that into an instruction about what to go change.

The `retrieval_failed` bucket — *wrong chunk retrieved, right answer given* — is
the most interesting one. It means the model answered from its own weights,
which on a fictional document usually means it guessed and got lucky.

### Absent questions are graded backwards

Three cases (`absent-price`, `absent-wifi`, `absent-ceo`) ask about facts that
are **not** in the handbook. For these, refusing is correct and answering is
failure. They exist because the anti-hallucination clamp in
`RAG_PROMPT_TEMPLATE` ("say you don't know based on the provided documents") is
a *claim*, and claims should be tested.

This is why the report never blends the two into one accuracy figure — a system
that answers everything scores 100% on answerable questions and 0% on absent
ones, and a system that refuses everything does the reverse. Both are broken.
The pair of numbers, plus `over-refusal rate`, tells you which.

### The metrics are boring on purpose

Everything in [`metrics.py`](metrics.py) is exact substring matching after
normalization (lowercase, collapse non-alphanumerics). No fuzzy ratios, no
embedding similarity between answer and reference.

A fuzzy scorer means a score change might come from the scorer rather than the
system, which defeats the point. The normalizer exists only to stop `26-month`
and `26 months` from counting as different, and its one real trap — `26 month`
matching inside `126 months` — is guarded by left-padding and has a test.

The metrics are themselves unit-tested in
[`tests/test_evals.py`](../tests/test_evals.py), including two tests that
validate *the dataset*: that every case has an expectation (a case with none
would pass unconditionally) and that every `must_retrieve` fragment actually
occurs in the source document (otherwise the case is unpassable and you would
chase a phantom retrieval bug).

### The judge is quarantined

Groundedness — "is every claim supported by the retrieved context?" — cannot be
a string function, so [`judge.py`](judge.py) asks the LLM. Three constraints
keep it honest: temperature 0, judging *support* rather than *truth*, and a
`None` verdict (excluded from the average) when the call fails, so an Ollama
hiccup never masquerades as a groundedness regression.

It is the same local llama3.2 doing the judging, which is a real accuracy
ceiling. The report labels the number **advisory** for that reason. Treat it as
a smoke alarm, not a measurement.

## Baseline, and what CI gates

Measured on the development machine (i5-1135G7, 8 GB, llama3.2:3b, top_k 4).
Recorded here so a threshold is derived from a number rather than chosen because
it looked round.

| stage | metric | baseline |
|---|---|---|
| retrieval | hit@4 | **100%** (12 answerable cases) |
| retrieval | MRR | **1.000** |
| retrieval | wall time | 0.3s, no model |
| generation | answerable accuracy | **67%** (8 of 12) |
| generation | refusal accuracy | **67%** (2 of 3 absent cases) |
| generation | over-refusal | 0% |
| generation | wall time | 279s, 18.5s per case |

**CI gates retrieval at 0.95**, and nothing else. One miss of twelve is 91.7%,
so a single regression fails the build. Not 1.0, because embedding scores are
floating point and a build agent is not this laptop; not 0.9, because that would
let a real miss through.

Generation is deliberately *not* gated. It needs Ollama and a 2 GB model, takes
four and a half minutes, and its score moves with sampling. A gate on a number
that wobbles teaches people to re-run CI until it goes green, which is worse
than not measuring it.

The generation figure is the honest headline of this project: **a third of
answerable questions are still wrong**, with retrieval scoring perfectly on
every one of them. The failures — `warranty-length`, `founders`,
`peak-power`, `warranty-paraphrase` — are all cases where the right chunk was
retrieved at rank 1 and the model still did not produce the fact. That is a
generation problem, and it is where the next work is.

### A run that measured nothing is never a pass

With Ollama down, every case returned `[engine error: ...]`, the harness scored
it 8% and printed *"retrieval OK, model fumbled — tune RAG_PROMPT_TEMPLATE or
try a larger model"*. Nothing was wrong with the prompt or the model. Engine
failures are now counted separately, reported as an outage, and fail the gate
outright regardless of the arithmetic.

## Metric reference

| Metric | Meaning | Read it as |
|---|---|---|
| `hit@k` | Fraction of cases where a retrieved chunk contained the gold fact | Retrieval ceiling — generation can never beat it |
| `MRR` | Mean of 1/(rank of first correct chunk) | Ranking quality. 1.0 = always first |
| `mean top score` | Average cosine similarity of the best chunk | Calibration. Useful for picking a relevance threshold |
| `answerable accuracy` | Correct answers on in-document questions | The headline |
| `refusal accuracy` | Correct refusals on absent questions | Hallucination resistance |
| `over-refusal rate` | Answerable questions wrongly refused | The cost of the anti-hallucination clamp |
| `groundedness` | Judge says the answer is entailed by the context | Advisory |

## Adding a case

Add an object to `cases` in [`dataset.json`](dataset.json):

```json
{
  "id": "unique-slug",
  "kind": "fact | paraphrase | multi-hop | exact-token | absent",
  "question": "What the user types",
  "must_retrieve": ["substring that must appear in one retrieved chunk"],
  "answer_any_of": ["accepted", "surface", "forms"],
  "also_expect": ["optional secondary fact"],
  "note": "why this case exists"
}
```

`must_retrieve` fragments are AND-ed and must all land in the **same** chunk —
a fact split across two chunks is a chunking failure and should score as a miss.
For `absent` cases, set `"expect_refusal": true` and omit the other two fields.

`tests/test_evals.py` will fail if your new case is malformed or unpassable, so
run `pytest tests/test_evals.py` after editing.

## Known limits

- **The corpus is 6 chunks.** With `top_k=4` you retrieve two-thirds of the
  document, so `hit@4` is close to unfalsifiable. `--top-k 1` is the honest
  ranking test. Real confidence needs a corpus of hundreds of chunks.
- **One document.** Cross-document retrieval — picking the right *file* before
  the right chunk — is untested.
- **Single-turn only.** The follow-up-question weakness (`chat_stream` embeds
  the raw last message, so "what about the second one?" retrieves noise) is a
  known gap with no case covering it.
- **The judge is a 3B model.** See above.
