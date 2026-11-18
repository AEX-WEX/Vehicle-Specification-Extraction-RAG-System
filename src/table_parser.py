"""
Table-aware helpers for service-manual specification tables.

Pure Python (no ML dependencies):
  * rows_from_words      - rebuild table rows from PyMuPDF word boxes
  * find_section         - "SECTION 204-01A: Front Suspension ..." title of a page
  * split_table_blocks   - separate "Torque Specifications" tables from prose
  * parse_torque_rows    - "Shock absorber nuts  90  66  -" -> description + Nm / lb-ft / lb-in
  * query_rows           - rank parsed rows against a user query
  * attribute_source     - find which retrieved chunk a value really came from
"""
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

NM_TO_LBFT = 0.737562
NM_TO_LBIN = 8.85075

SECTION_RE = re.compile(r"SECTION\s*:?\s*(\d{3}-\d{2}[A-Z]?)\s*:\s*([^\n]+)")
TABLE_HEADINGS = ("Torque Specifications", "Material", "General Specifications")
BOILERPLATE_RE = re.compile(
    r"^\s*(Page \d+ (?:of|sur) \d+.*|file:///.*|.*repair4less.*|\d{4}-\d{2}-\d{2})\s*$",
    re.IGNORECASE | re.MULTILINE,
)
DASHES = {"—", "–", "-", "‑", "−"}
NUM_RE = re.compile(r"^\d+(?:\.\d+)?$")

STOP = {
    "torque", "torques", "specification", "specifications", "spec", "specs", "what", "is", "the", "of",
    "for", "value", "values", "tightening", "tighten", "to", "a", "an", "and", "in", "on", "vehicle",
    "ford", "f", "150", "manual", "capacity", "fill",
}


# ----------------------------------------------------------------------------- layout
def rows_from_words(words: List[tuple], y_tol: float = 3.0, gap: float = 12.0) -> str:
    """Rebuild text rows from PyMuPDF `page.get_text("words")` tuples.

    Cells of one table row share a vertical position, so they are joined on one line
    (two spaces between distant cells, one between close words). PyMuPDF's plain "text"
    mode puts every cell on its own line, which destroys the table structure.
    """
    ws = [w for w in words if str(w[4]).strip()]
    ws.sort(key=lambda w: ((w[1] + w[3]) / 2.0, w[0]))
    rows: List[List[tuple]] = []
    for w in ws:
        yc = (w[1] + w[3]) / 2.0
        if rows and abs(yc - rows[-1][0]) <= y_tol:
            rows[-1][1].append(w)
        else:
            rows.append([yc, [w]])
    lines = []
    for _, cells in rows:
        cells.sort(key=lambda w: w[0])
        out, prev = "", None
        for w in cells:
            if prev is not None:
                out += ("  " if w[0] - prev[2] > gap else " ")
            out += str(w[4])
            prev = w
        lines.append(out)
    return "\n".join(lines)


def strip_boilerplate(text: str) -> str:
    """Remove page footers/URLs but keep every number (table cells are numeric)."""
    return BOILERPLATE_RE.sub("", text)


# ----------------------------------------------------------------------------- sections
def find_section(text: str) -> Optional[str]:
    m = SECTION_RE.search(text or "")
    if not m:
        return None
    title = re.split(r"\s{2,}", m.group(2).strip())[0]
    title = re.sub(r"\s*\d{4}\s+F-?150 Workshop Manual.*$", "", title).strip()
    return f"SECTION {m.group(1)}: {title}"


def section_prefix(chunk_text: str) -> str:
    m = re.match(r"^\[([^\]]+)\]", chunk_text or "")
    return m.group(1) if m else ""


# ----------------------------------------------------------------------------- tables
def split_table_blocks(text: str) -> Tuple[str, List[str]]:
    """Return (prose_without_tables, [table blocks]); a block runs from its heading to the next heading/end."""
    heads = [m for m in re.finditer(r"^\s*(%s)\s*$" % "|".join(map(re.escape, TABLE_HEADINGS)), text,
                                    re.MULTILINE)]
    if not heads:
        return text, []
    blocks, prose, last = [], [], 0
    for i, m in enumerate(heads):
        prose.append(text[last:m.start()])
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        blocks.append(text[m.start():end].strip())
        last = end
    prose.append(text[last:])
    return "\n".join(prose), blocks


def split_long_block(block: str, max_chars: int = 3000) -> List[str]:
    """Split an oversized table on row boundaries, repeating the heading + column header."""
    if len(block) <= max_chars:
        return [block]
    lines = block.split("\n")
    head = lines[:2]
    out, cur = [], list(head)
    for ln in lines[2:]:
        if sum(len(x) + 1 for x in cur) + len(ln) > max_chars and len(cur) > 2:
            out.append("\n".join(cur))
            cur = list(head)
        cur.append(ln)
    out.append("\n".join(cur))
    return out


@dataclass
class TorqueRow:
    description: str
    nm: Optional[float]
    lb_ft: Optional[float]
    lb_in: Optional[float]

    def primary(self) -> Tuple[str, str]:
        if self.nm is not None:
            return _fmt(self.nm), "Nm"
        if self.lb_ft is not None:
            return _fmt(self.lb_ft), "lb-ft"
        return _fmt(self.lb_in), "lb-in"


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(x)


def _near(a: float, b: float) -> bool:
    return abs(a - b) <= max(1.0, 0.04 * max(a, b))


def _assign(vals: List[Optional[float]], cols: Optional[List[str]]) -> TorqueRow:
    nm = lbft = lbin = None
    if cols and len(cols) == len(vals):
        for c, v in zip(cols, vals):
            if c == "Nm":
                nm = v
            elif c == "lb-ft":
                lbft = v
            elif c == "lb-in":
                lbin = v
        return TorqueRow("", nm, lbft, lbin)
    # unknown layout: identify columns by unit ratios
    nm = vals[0]
    for v in vals[1:]:
        if v is None:
            continue
        if nm is not None and _near(v, nm * NM_TO_LBFT) and lbft is None:
            lbft = v
        elif nm is not None and _near(v, nm * NM_TO_LBIN) and lbin is None:
            lbin = v
    return TorqueRow("", nm, lbft, lbin)


def parse_torque_rows(block: str) -> List[TorqueRow]:
    """Parse the rows of one "Torque Specifications" block."""
    lines = [l.rstrip() for l in block.split("\n") if l.strip()]
    cols: Optional[List[str]] = None
    rows: List[TorqueRow] = []
    for ln in lines:
        low = ln.lower()
        if "nm" in low.split() and ("lb-ft" in low or "lb-in" in low or low.strip().endswith("nm")):
            cols = [c for c in ("Nm", "lb-ft", "lb-in") if c.lower() in low.split()]
            continue
        toks = ln.split()
        vals: List[Optional[float]] = []
        while toks and len(vals) < 3 and (NUM_RE.match(toks[-1]) or toks[-1] in DASHES):
            t = toks.pop()
            vals.insert(0, float(t) if NUM_RE.match(t) else None)
        if not vals or not toks or not any(v is not None for v in vals):
            continue
        desc = " ".join(toks)
        if not re.search(r"[A-Za-z]{3}", desc) or desc.lower().startswith(("description", "torque spec")):
            continue
        desc = re.sub(r"\s+[a-z]$", "", desc).strip()          # footnote marker ("... bolts a")
        r = _assign(vals, cols)
        r.description = desc
        if r.nm is None and r.lb_ft is None and r.lb_in is None:
            continue
        rows.append(r)
    return rows


# ----------------------------------------------------------------------------- query matching
def _stem(t: str) -> str:
    return t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t


def tokens(s: str) -> set:
    return {_stem(t) for t in re.findall(r"[a-z0-9]+", (s or "").lower())} - {_stem(x) for x in STOP}


CAPACITY_WORDS = {"capacity", "fluid", "lubricant", "oil", "coolant", "refill", "fill", "pint", "litre", "liter", "quart"}
TORQUE_WORDS = {"torque", "tighten", "tightening", "nm", "lb-ft", "lb-in", "ft-lb"}


def is_torque_query(query: str) -> bool:
    """False for fluid-capacity style questions so torque rows are not returned for them."""
    words = set(re.findall(r"[a-z\-]+", (query or "").lower()))
    return bool(words & TORQUE_WORDS) or not (words & CAPACITY_WORDS)


def query_rows(query: str, rows: List[TorqueRow], section: str = "", max_rows: int = 5):
    """Rank rows against a query; returns [(row, score)] containing only the best-scoring rows."""
    if not is_torque_query(query):
        return []
    q, sec = tokens(query), tokens(section)
    scored = []
    for r in rows:
        d = tokens(r.description)
        if not d:
            continue
        d_hit = len(q & d)
        covers_query = bool(q) and d_hit >= 2 and d_hit / len(q) >= 0.8
        if d_hit < max(1, math.ceil(0.7 * len(d))) and not covers_query:
            continue
        scored.append((r, 2.0 * d_hit + len(q & sec - d) + d_hit / len(d)))
    if not scored:
        return []
    best = max(s for _, s in scored)
    return [(r, s) for r, s in sorted(scored, key=lambda x: -x[1]) if s >= best - 1e-9][:max_rows]


# ----------------------------------------------------------------------------- provenance
def attribute_source(component: str, value: str, contexts: List[Dict]) -> Optional[Dict]:
    """Return the retrieved context that really contains this component/value (not just contexts[0])."""
    comp, best, best_s = tokens(component), None, 0.0
    for c in contexts:
        text = c.get("text", "")
        s = 0.0
        if value and re.search(r"(?<![\d.])%s(?![\d.])" % re.escape(str(value)), text):
            s += 1.0
        if comp:
            s += len(comp & tokens(text)) / len(comp)
        if s > best_s:
            best, best_s = c, s
    return best


# ----------------------------------------------------------------------------- fill capacities
SPEC_CODE_RE = re.compile(r"\b(?:WS[A-Z]|ES[A-Z])-[A-Z0-9][A-Z0-9-]*")
QTY_L_RE = re.compile(r"(?<![\w.\-])(\d+(?:\.\d+)?)\s?(L|ml)\b")
QTY_OZ_RE = re.compile(r"(?<![\w.\-(])(\d+(?:\.\d+)?)\s?oz\b")
ALT_RE = re.compile(r"\((\d+(?:\.\d+)?)\s*(pt|qt)\)")
CUT_RE = re.compile(r"\band GL-5\b|\bthe bottom of\b|\bfiller hole\b|\((?:US|Canada)\)|\bX[A-Z]-\S+|\bCX[A-Z]-\S+|"
                    r"\bXL-3\b|\bCXL-3\b|\bTA-\d+|\bLoctite|\(\d|\(to\b|\bto be\b|—|\bonly\b|\bequipped\b|\bNot to\b|"
                    r"\binstalled\b|\bTraction-Lok\b|\b\d+W\d+-|\bCXY?-|\(\s*\)|\bused with\b")


@dataclass
class CapacityEntry:
    name: str
    value: str
    unit: str
    alt: str = ""


def _clean_name(rows: List[str]) -> str:
    parts = []
    for r in rows:
        r = SPEC_CODE_RE.sub(" ", r)
        r = QTY_L_RE.sub(" ", QTY_OZ_RE.sub(" ", r))
        m = CUT_RE.search(r)
        regional = bool(m) and m.group(0) in ("(US)", "(Canada)")
        r = r[:m.start()] if m else r
        r = re.sub(r"\s+[A-Z]\d?\s*$", "", r.strip())            # trailing spec-code fragment ("... Lubricant A")
        r = re.sub(r"\s+or\s*$", "", r)
        if not r.strip():
            if parts:
                break                                              # rest of the entry is codes/notes
            continue
        parts.append(r.strip())
        if regional:
            break                                                  # US/Canada part numbers follow
    name = " ".join(parts)
    name = re.sub(r"-\s+(\d)", r"-\1", name).replace("®", "").replace("™", "")
    return re.sub(r"\s+", " ", name).strip()


def parse_capacity_entries(block: str) -> List[CapacityEntry]:
    """Parse a "Material" table (Item | Specification | Fill Capacity) into one entry per fluid/material.

    Rows of one entry wrap over several lines; an entry starts at a row that holds a specification
    code (e.g. WSL-M2C192-A) or ends in a pair of dashes (nothing to fill).
    """
    lines = [l.strip() for l in block.split("\n") if l.strip()]
    lines = [l for l in lines if not re.match(r"^(Material|Item\b.*|Capacity|Specification)$", l)]
    entries: List[List[str]] = []
    for ln in lines:
        if re.match(r"^(General Specifications|Torque Specifications|Rear Axle Backlash|Special Tool)", ln):
            break
        m_code = SPEC_CODE_RE.search(ln)
        alt_code = bool(m_code) and re.search(r"\bor\s*$", ln[:m_code.start()]) is not None   # "... Brake or WSS-M6C65-"
        starts = (bool(m_code) and not alt_code) or bool(re.search(r"—\s+—\s*$", ln))
        if starts or not entries:
            entries.append([ln])
        else:
            entries[-1].append(ln)
    out: List[CapacityEntry] = []
    for rows in entries:
        text = " ".join(rows)
        qtys = [(m.group(1), m.group(2)) for m in QTY_L_RE.finditer(text)]
        qtys += [(m.group(1), "oz") for m in QTY_OZ_RE.finditer(text)]
        if not qtys:
            continue
        alt = ALT_RE.search(text)
        name = _clean_name(rows)
        for value, unit in qtys:
            out.append(CapacityEntry(name, value, unit, f"{alt.group(1)} {alt.group(2)}" if alt and len(qtys) == 1 else ""))
    return out


def is_capacity_query(query: str) -> bool:
    words = set(re.findall(r"[a-z\-]+", (query or "").lower()))
    return bool(words & CAPACITY_WORDS) and not (words & TORQUE_WORDS)


def query_capacity(query: str, entries: List[CapacityEntry], section: str = "", max_rows: int = 5):
    if not is_capacity_query(query):
        return []
    q, sec = tokens(query), tokens(section)
    scored = []
    for e in entries:
        d = tokens(e.name)
        hit = len(q & (d | sec))
        if not d or len(q & d) < 1 or hit < max(2, math.ceil(0.6 * len(q))):
            continue
        scored.append((e, 2.0 * len(q & d) + len(q & sec - d)))
    if not scored:
        return []
    best = max(s for _, s in scored)
    return [(e, s) for e, s in sorted(scored, key=lambda x: -x[1]) if s >= best - 1e-9][:max_rows]


# ----------------------------------------------------------------------------- answer validation
GENERIC_PARTS = {"nut", "bolt", "screw", "stud", "fitting", "plug"}


def is_grounded(component: str, value: str, spec_type: str, query: str, contexts: List[Dict]) -> bool:
    """Accept an extracted spec only if the retrieved text itself backs it up and it answers the question.

    1. a retrieved chunk contains the value as a standalone number,
    2. that same chunk contains the component's words (rejects invented names such as "Wiper Arm Nut"),
    3. that chunk (plus the component name) covers >= 80% of the query's content words (rejects an
       answer about air-cleaner bolts for a "cabin air filter" question),
    4. the spec type fits the question (a torque value is not an answer to a capacity question).
    """
    if not component or not str(value).strip():
        return False
    st = (spec_type or "").lower()
    if is_capacity_query(query) and "torque" in st:
        return False
    if is_torque_query(query) and not is_capacity_query(query) and "capacity" in st:
        return False
    qc = tokens(query) - GENERIC_PARTS - {_stem(w) for w in CAPACITY_WORDS | TORQUE_WORDS}
    cc = tokens(component) - GENERIC_PARTS
    pat = re.compile(r"(?<![\d.])%s(?![\d.])" % re.escape(str(value).strip()))
    for c in contexts:
        text = c.get("text", "")
        if not pat.search(text):
            continue
        ct = tokens(text)
        if cc and len(cc & ct) < math.ceil(0.8 * len(cc)):
            continue
        if qc and len(qc & (ct | cc)) < math.ceil(0.8 * len(qc)):
            continue
        return True
    return False


# ----------------------------------------------------------------------------- hallucination guards
def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def verify_evidence(evidence: str, value: str, contexts: List[Dict]) -> Optional[Dict]:
    """The model must quote the manual. Return the chunk that contains the quote verbatim (ignoring
    whitespace and case) and the value inside it; None if the quote was paraphrased or invented."""
    ev = normalize_ws(evidence)
    if len(ev) < 6 or not str(value).strip():
        return None
    pat = re.compile(r"(?<![\d.])%s(?![\d.])" % re.escape(str(value).strip()))
    for c in contexts:
        if ev in normalize_ws(c.get("text", "")) and pat.search(ev):
            return c
    return None


def retrieval_sufficient(query: str, contexts: List[Dict], threshold: float = 0.6) -> bool:
    """False when no retrieved chunk covers most of the question's content words. The system then
    answers "not found" without asking the LLM, which can only guess at that point."""
    qc = tokens(query) - GENERIC_PARTS - {_stem(w) for w in CAPACITY_WORDS | TORQUE_WORDS}
    if not qc:
        return bool(contexts)
    best = max((len(qc & tokens(c.get("text", ""))) / len(qc) for c in contexts), default=0.0)
    return best >= threshold
