"""Pure-Python scoring for the vehicle-spec extraction benchmark (no ML dependencies)."""
import re
from typing import Dict, List, Optional

UNIT_ALIASES = {
    "nm": "Nm", "n-m": "Nm", "n.m": "Nm", "newton-meter": "Nm", "newton-metre": "Nm",
    "lb-ft": "lb-ft", "ft-lb": "lb-ft", "ft-lbs": "lb-ft", "lbf-ft": "lb-ft", "lb-ft.": "lb-ft",
    "lb-in": "lb-in", "in-lb": "lb-in", "in-lbs": "lb-in", "lbf-in": "lb-in",
    "l": "L", "liter": "L", "liters": "L", "litre": "L", "litres": "L", "lt": "L",
    "oz": "oz", "ounce": "oz", "ounces": "oz",
}
STOP = {"the", "of", "and", "a", "an", "to", "for", "bolt", "bolts", "nut", "nuts"}


def norm_unit(u: Optional[str]) -> str:
    u = (u or "").strip().lower().replace("·", "-").replace("_", "-")
    return UNIT_ALIASES.get(u, u)


def to_float(v) -> Optional[float]:
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        m = re.search(r"-?\d+(?:\.\d+)?", str(v or ""))
        return float(m.group(0)) if m else None


def _tokens(s: str, drop_fasteners: bool = False) -> set:
    t = set(re.findall(r"[a-z0-9]+", (s or "").lower()))
    return t - STOP if drop_fasteners else t - {"the", "of", "and", "a", "an", "to", "for"}


def component_match(pred: str, gold: str, threshold: float = 0.75) -> bool:
    """Token-set match: every gold token must be present in the prediction, or Jaccard >= threshold."""
    p, g = _tokens(pred), _tokens(gold)
    if not p or not g:
        return False
    if g <= p or p <= g and len(p) >= max(2, len(g) - 1):
        return True
    return len(p & g) / len(p | g) >= threshold


def value_unit_match(pred: Dict, accept: List[Dict], tol: float = 0.01) -> bool:
    pv, pu = to_float(pred.get("value")), norm_unit(pred.get("unit"))
    if pv is None:
        return False
    return any(abs(pv - float(a["value"])) <= tol and pu == norm_unit(a["unit"]) for a in accept)


def score_one(preds: List[Dict], gold: Dict) -> Dict:
    """Outcome for one gold spec: found / correct / page_ok / wrong_value / no_answer."""
    cands = [p for p in preds if component_match(p.get("component", ""), gold["component"])
             and (p.get("spec_type") or "").strip().lower() == gold["spec_type"].lower()]
    correct = [p for p in cands if value_unit_match(p, gold["accept"])]
    value_only = [p for p in cands if not correct and to_float(p.get("value")) is not None
                  and any(abs(to_float(p["value"]) - float(a["value"])) <= 0.01 for a in gold["accept"])]
    pages = gold["page_number"] if isinstance(gold["page_number"], list) else [gold["page_number"]]
    page_ok = any(p.get("page_number") in pages for p in correct)
    return {
        "id": gold["id"], "query": gold["query"], "component": gold["component"],
        "returned": len(preds), "found": bool(cands), "correct": bool(correct),
        "page_ok": page_ok, "value_only": bool(value_only),
        "wrong_value": bool(cands) and not correct,
        "no_answer": not preds,
    }


def summarize(rows: List[Dict]) -> Dict:
    n = len(rows) or 1
    c = sum(r["correct"] for r in rows)
    return {
        "n_specs": len(rows),
        "exact_match_accuracy": c / n,            # right component + value + unit
        "page_accuracy_given_correct": (sum(r["page_ok"] for r in rows) / c) if c else 0.0,
        "component_found_rate": sum(r["found"] for r in rows) / n,
        "wrong_value_rate": sum(r["wrong_value"] for r in rows) / n,   # found component, wrong number/unit
        "no_answer_rate": sum(r["no_answer"] for r in rows) / n,
        "value_only_rate": sum(r["value_only"] for r in rows) / n,     # right number, wrong/missing unit
    }


def score_abstention(preds_per_query: List[List[Dict]]) -> Dict:
    """For unanswerable questions the right behaviour is to return nothing."""
    n = len(preds_per_query) or 1
    answered = sum(1 for p in preds_per_query if p)
    return {"n_queries": len(preds_per_query), "false_answer_rate": answered / n, "abstention_rate": 1 - answered / n}


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]
