# Evaluation report

Document: data/sample-service-manual_1.pdf (2014 Ford F-150 Workshop Manual, 852 pp.)

## dev (43 labeled specs)

| Metric | Value |
|---|---|
| Exact-match accuracy (component + value + unit) | 100.0% |
| Page correct, given a correct answer | 100.0% |
| Wrong value or unit (component found) | 0.0% |
| No answer returned | 0.0% |
| Answers returned per query (avg) | 1.2 |
| Latency p50 / p95 | 0.02 s / 0.02 s |
| Extraction method | {'table_parser': 43} |

## heldout (26 labeled specs)

| Metric | Value |
|---|---|
| Exact-match accuracy (component + value + unit) | 100.0% |
| Page correct, given a correct answer | 100.0% |
| Wrong value or unit (component found) | 0.0% |
| No answer returned | 0.0% |
| Answers returned per query (avg) | 1.6 |
| Latency p50 / p95 | 0.02 s / 0.03 s |
| Extraction method | {'table_parser': 26} |

## unanswerable: questions not in the manual (12)

| Metric | Value |
|---|---|
| Correctly returned nothing | 100.0% |
| False answers | 0.0% |


