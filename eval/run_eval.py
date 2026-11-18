"""Evaluate the pipeline. Three sets, each answering a different question:

  dev           eval/ground_truth.json  43 labeled specs used while building the parser (can be over-fitted)
  heldout       eval/heldout.json       25 labeled specs never used for tuning - the number to quote
  unanswerable  eval/unanswerable.json  12 questions NOT answered in the manual; the system must return nothing

Usage (repo root, index built, Ollama running for capacity/prose questions):
    python -m eval.run_eval                                  # all sets
    python -m eval.run_eval --sets heldout --min-exact 0.9   # CI/release gate: exit code 1 if below threshold
    python -m eval.run_eval --predictions eval/predictions.json   # re-score saved output, no model needed
Writes eval/predictions.json and eval/report.md.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from eval.scoring import percentile, score_abstention, score_one, summarize

HERE = Path(__file__).parent
FILES = {"dev": "ground_truth.json", "heldout": "heldout.json", "unanswerable": "unanswerable.json"}


def run_queries(pipe, items):
    out = {}
    for key, query in items:
        t0 = time.time()
        res = pipe.query(query)
        out[key] = {"specifications": res.get("specifications", []), "method": res.get("extraction_method"),
                    "latency_s": round(time.time() - t0, 3)}
    return out


def stats(preds):
    lat = [p["latency_s"] for p in preds.values() if "latency_s" in p]
    methods = {}
    for p in preds.values():
        methods[p.get("method")] = methods.get(p.get("method"), 0) + 1
    return {"p50": percentile(lat, 0.5), "p95": percentile(lat, 0.95), "methods": methods,
            "avg_answers": sum(len(p["specifications"]) for p in preds.values()) / max(len(preds), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=list(FILES), choices=list(FILES))
    ap.add_argument("--predictions", help="score a saved predictions.json instead of running the pipeline")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default=str(HERE / "report.md"))
    ap.add_argument("--min-exact", type=float, help="fail (exit 1) if held-out exact-match accuracy is below this")
    ap.add_argument("--max-false-answer", type=float, help="fail if more than this share of unanswerable queries get an answer")
    a = ap.parse_args()

    data = {s: json.load(open(HERE / FILES[s], encoding="utf-8")) for s in a.sets}
    if a.predictions:
        allp = json.load(open(a.predictions, encoding="utf-8"))
    else:
        from src.pipeline import create_pipeline
        pipe = create_pipeline()
        pipe.load_index()
        allp = {}
        for s in a.sets:
            if s == "unanswerable":
                items = [(q, q) for q in data[s]["queries"]]
            else:
                items = [(g["id"], g["query"]) for g in data[s]["specs"]]
            allp[s] = run_queries(pipe, items[: a.limit] if a.limit else items)
        json.dump(allp, open(HERE / "predictions.json", "w", encoding="utf-8"), indent=1)

    L = ["# Evaluation report", "", "Document: data/sample-service-manual_1.pdf (2014 Ford F-150 Workshop Manual, 852 pp.)", ""]
    gate_fail = []
    summary = {}
    for s in a.sets:
        preds = allp.get(s, {})
        st = stats(preds)
        if s == "unanswerable":
            qs = data[s]["queries"][: a.limit] if a.limit else data[s]["queries"]
            m = score_abstention([preds.get(q, {}).get("specifications", []) for q in qs])
            summary[s] = m
            L += [f"## {s}: questions not in the manual ({m['n_queries']})", "", "| Metric | Value |", "|---|---|",
                  f"| Correctly returned nothing | {m['abstention_rate']:.1%} |",
                  f"| False answers | {m['false_answer_rate']:.1%} |", ""]
            for q in qs:
                if preds.get(q, {}).get("specifications"):
                    sp = preds[q]["specifications"][0]
                    L.append(f"- answered `{q}` with {sp['component']} = {sp['value']} {sp['unit']} (page {sp['page_number']})")
            if a.max_false_answer is not None and m["false_answer_rate"] > a.max_false_answer:
                gate_fail.append(f"false-answer rate {m['false_answer_rate']:.1%} > {a.max_false_answer:.0%}")
            L.append("")
            continue
        specs = data[s]["specs"][: a.limit] if a.limit else data[s]["specs"]
        rows = [score_one(preds.get(g["id"], {}).get("specifications", []), g) for g in specs]
        m = summarize(rows)
        summary[s] = m
        L += [f"## {s} ({m['n_specs']} labeled specs)", "", "| Metric | Value |", "|---|---|",
              f"| Exact-match accuracy (component + value + unit) | {m['exact_match_accuracy']:.1%} |",
              f"| Page correct, given a correct answer | {m['page_accuracy_given_correct']:.1%} |",
              f"| Wrong value or unit (component found) | {m['wrong_value_rate']:.1%} |",
              f"| No answer returned | {m['no_answer_rate']:.1%} |",
              f"| Answers returned per query (avg) | {st['avg_answers']:.1f} |",
              f"| Latency p50 / p95 | {st['p50']:.2f} s / {st['p95']:.2f} s |",
              f"| Extraction method | {st['methods']} |", ""]
        miss = [r for r in rows if not r["correct"]]
        if miss:
            L += ["Failures:", "", "| id | gold component | outcome |", "|---|---|---|"]
            for r in miss:
                why = "no answer" if r["no_answer"] else ("wrong value/unit" if r["wrong_value"] else "component not found")
                L.append(f"| {r['id']} | {r['component']} | {why} |")
            L.append("")
        if s == "heldout" and a.min_exact is not None and m["exact_match_accuracy"] < a.min_exact:
            gate_fail.append(f"held-out exact-match {m['exact_match_accuracy']:.1%} < {a.min_exact:.0%}")

    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    if gate_fail:
        print("\nGATE FAILED: " + "; ".join(gate_fail))
        sys.exit(1)


if __name__ == "__main__":
    main()
