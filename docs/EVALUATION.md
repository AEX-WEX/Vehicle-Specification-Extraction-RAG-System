# Evaluation

How the system is measured, what each number means, and how to keep it honest as the code or the manuals change.

## What is measured

Accuracy is measured on questions with a known answer; safety is measured on questions with no answer.

| Set | File | Size | Purpose |
|---|---|---|---|
| dev | `eval/ground_truth.json` | 43 specs | Used while building the parser. It can be over-fitted, so use it to catch regressions, not as a headline number. |
| held-out | `eval/heldout.json` | 25 specs | Never used for tuning. Short, natural queries. Labels were read from the manual tables by eye, not produced by the parser. **This is the number to quote.** |
| unanswerable | `eval/unanswerable.json` | 12 queries | Things that are not in the manual (spark plug torque, engine oil capacity...). The right behaviour is to return nothing. |

### Metrics

- **Exact-match accuracy**: the right component, the right value and the right unit. A torque row lists several units, so any listed (value, unit) pair counts.
- **Page accuracy (given a correct answer)**: the cited page is a page that really contains the row. Some rows appear in several sections, so a gold spec can list several valid pages.
- **Wrong value or unit**: the component was found but the number or unit was wrong. This is the dangerous failure for a technician, so it is tracked separately from "no answer".
- **No answer / false-answer rate**: no answer on a question that has one costs recall; an answer to a question that has none is a hallucination. The unanswerable set measures the second.
- **Answers per query**: how many candidate rows are returned. The same row name can occur in several sections (front vs rear suspension), so a query without section words returns each candidate with its own page.
- **Latency p50 / p95** and **extraction method mix** (table parser vs LLM vs regex).

## How an answer is protected

An LLM will happily produce a plausible answer to a question the manual never answers (an early run returned "Engine oil = 1.66 liters" for an engine-coolant question). Each answer therefore passes several independent checks, and failing any of them produces "not found":

1. **Deterministic first.** Torque and fill-capacity rows are parsed straight from the PDF text, so the number is copied and never generated.
2. **Retrieval sufficiency.** If no retrieved chunk covers at least 60% of the question's key words, the system abstains without calling the LLM.
3. **Answer only from the text.** The prompt forbids outside knowledge, allows an empty answer, and contains only made-up examples, so the model has nothing to copy from.
4. **Quoted evidence.** The LLM must quote the manual. The quote is checked verbatim against the retrieved chunks and must contain the value; paraphrased or invented quotes are dropped.
5. **Grounding.** The chunk behind the answer must contain both the component's words and the value, and cover at least 80% of the question's key words. The answer type must fit the question (a torque is not a capacity).
6. **Provenance.** Each result carries the page and chunk ID of its verified evidence.

These checks trade a little recall for safety: an answer the model cannot quote is dropped. They reduce hallucination; they do not prove its absence, which is why the unanswerable set and production monitoring exist.

## Running it

```bash
python -m eval.run_eval                                   # all sets, writes eval/report.md
python -m eval.run_eval --sets heldout unanswerable --min-exact 0.90 --max-false-answer 0.10
python -m eval.run_eval --predictions eval/predictions.json   # re-score saved output, no models needed
pytest tests/test_table_parser.py tests/test_eval_scoring.py  # unit tests, also run in CI
```

`--min-exact` and `--max-false-answer` make the command exit with status 1 when a threshold is missed, so it can gate a release. After any change to parsing, chunking or extraction, rebuild the index (`python main.py index <pdf> --force`) and re-run the evaluation.

## Reading the numbers

- A small test set gives a wide interval. With 25 held-out specs, a perfect score only supports "at least about 87%" at 95% confidence (Wilson lower bound); 25 + 43 = 68 specs support about 95%. Quote the count with the percentage.
- All sets come from one manual in one layout (Ford workshop-manual tables). Results do not transfer to a differently laid-out manual until it has its own labeled set.
- The unanswerable set was extended after one false answer was found, so treat its 0% as a regression test, not a promise.

## Before production use

1. **Label a new set for every new manual or manual family** (about 30 specs, with natural queries) and keep it out of development until it is time to measure.
2. **Log in production**: method used, number of answers, ungrounded answers dropped, latency, and user thumbs-up/down. `logs/extraction_stats.json` already records method, timing and counts.
3. **Monitor**: the abstention rate, the share of answers from the LLM path (these are the ones not copied from a table), and p95 latency. A drop in table-parser share usually means a new PDF layout.
4. **Feed corrections back**: every wrong answer reported by a user becomes a new labeled spec; re-run the gate before releasing.
5. **Human in the loop for safety-critical torques**: show the cited page and chunk next to every value so the technician can verify it.
