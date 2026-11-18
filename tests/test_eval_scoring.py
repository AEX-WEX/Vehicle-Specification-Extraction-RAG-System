"""Unit tests for the evaluation scoring (no model or index needed)."""
import json
from pathlib import Path

from eval.scoring import component_match, norm_unit, percentile, score_abstention, score_one, summarize

GOLD = json.load(open(Path(__file__).parent.parent / "eval" / "ground_truth.json", encoding="utf-8"))["specs"]
SHOCK = next(g for g in GOLD if g["id"] == "T09")  # rear shock absorber nuts: 90 Nm / 66 lb-ft, page 128


def _p(c, v, u, page=128, t="Torque"):
    return {"component": c, "spec_type": t, "value": v, "unit": u, "page_number": page}


def test_units_normalised():
    assert norm_unit("ft-lbs") == "lb-ft" and norm_unit("N-m") == "Nm" and norm_unit("liters") == "L"


def test_component_match_is_tolerant_but_not_loose():
    assert component_match("Rear shock absorber nuts", "Shock absorber nuts")
    assert not component_match("Brake caliper guide pin bolts", "Shock absorber nuts")


def test_correct_answer_in_either_unit():
    assert score_one([_p("Shock absorber nuts", "90", "Nm")], SHOCK)["correct"]
    assert score_one([_p("Shock absorber nuts", "66", "lb-ft")], SHOCK)["correct"]


def test_known_repo_error_is_flagged():
    # results.csv in the repo reported 136 lb-in for the spring shackle nut; the manual says 185 Nm / 136 lb-ft.
    g = next(x for x in GOLD if x["component"] == "Spring shackle-to-frame nut")
    r = score_one([_p("Spring shackle-to-frame nut", "136", "lb-in", 259)], g)
    assert not r["correct"] and r["wrong_value"] and r["value_only"]


def test_page_checked_separately():
    r = score_one([_p("Shock absorber nuts", "90", "Nm", page=259)], SHOCK)
    assert r["correct"] and not r["page_ok"]


def test_no_answer_and_summary():
    rows = [score_one([], SHOCK), score_one([_p("Shock absorber nuts", "90", "Nm")], SHOCK)]
    s = summarize(rows)
    assert rows[0]["no_answer"] and s["exact_match_accuracy"] == 0.5 and s["no_answer_rate"] == 0.5


def test_ground_truth_sane():
    assert len(GOLD) >= 40 and len({g["id"] for g in GOLD}) == len(GOLD)
    assert all(g["accept"] and g["page_number"] for g in GOLD)


def test_page_list_accepted():
    g = {"id": "x", "query": "q", "component": "Skid plate bolts", "spec_type": "Torque", "page_number": [59, 234],
         "accept": [{"value": 48, "unit": "Nm"}]}
    assert score_one([_p("Skid plate bolts", "48", "Nm", page=234)], g)["page_ok"]
    assert not score_one([_p("Skid plate bolts", "48", "Nm", page=5)], g)["page_ok"]


def test_abstention_and_percentile():
    m = score_abstention([[], [], [{"component": "x"}], []])
    assert m["false_answer_rate"] == 0.25 and m["abstention_rate"] == 0.75
    assert percentile([1, 2, 3, 4, 100], 0.5) == 3 and percentile([], 0.5) == 0.0


def test_heldout_and_unanswerable_files():
    root = Path(__file__).parent.parent / "eval"
    held = json.load(open(root / "heldout.json", encoding="utf-8"))["specs"]
    dev_q = {g["query"] for g in GOLD}
    assert len(held) >= 20 and not ({h["query"] for h in held} & dev_q)
    assert len(json.load(open(root / "unanswerable.json", encoding="utf-8"))["queries"]) >= 10
