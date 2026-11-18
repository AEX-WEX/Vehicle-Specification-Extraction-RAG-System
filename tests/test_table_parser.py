"""Tests for the table-aware parsing added to fix lost torque values and wrong page attribution."""
from src.pdf_loader import TextCleaner
from src.table_parser import (attribute_source, find_section, parse_torque_rows, query_rows,
                              rows_from_words, split_table_blocks)

TABLE = """Torque Specifications
Description Nm lb-ft lb-in
Shock absorber nuts 90 66 —
Shock absorber shield bolts (SVT Raptor) 4 — 35
Spring shackle-to-frame nut 185 136 —
Spring U-bolt nuts a — — —
a Refer to the procedure in this section."""


def test_cleaner_keeps_numeric_only_lines():
    # regression: the old cleaner deleted lines that contained only digits, i.e. every table cell
    assert "90" in TextCleaner.clean_text("Shock absorber nuts\n90\n66\n")


def test_rows_rebuilt_from_word_boxes():
    words = [(10, 100, 80, 110, "Filler", 0, 0, 0), (82, 100, 110, 110, "plug", 0, 0, 1),
             (200, 101, 215, 111, "52", 0, 1, 0), (260, 100, 275, 110, "38", 0, 2, 0),
             (10, 120, 60, 130, "Axle", 1, 0, 0), (62, 120, 90, 130, "vent", 1, 0, 1), (200, 120, 215, 130, "18", 1, 1, 0)]
    assert rows_from_words(words).split("\n") == ["Filler plug  52  38", "Axle vent  18"]


def test_parse_rows_units_and_skips():
    rows = {r.description: r for r in parse_torque_rows(TABLE)}
    assert (rows["Shock absorber nuts"].nm, rows["Shock absorber nuts"].lb_ft) == (90, 66)
    assert rows["Shock absorber shield bolts (SVT Raptor)"].lb_in == 35
    assert rows["Spring shackle-to-frame nut"].primary() == ("185", "Nm")
    assert "Spring U-bolt nuts" not in rows            # all dashes: "refer to the procedure"


def test_ratio_inference_without_header():
    rows = parse_torque_rows("Torque Specifications\nAxle vent 18 — 159\nFiller plug 52 38 —")
    assert rows[0].lb_in == 159 and rows[1].lb_ft == 38


def test_query_prefers_matching_section():
    rows = parse_torque_rows(TABLE)
    hit = query_rows("rear suspension shock absorber nuts torque", rows, "SECTION 204-02: Rear Suspension")
    assert hit and hit[0][0].description == "Shock absorber nuts"


def test_section_and_table_split():
    page = "SECTION 204-02: Rear Suspension    2014 F-150 Workshop Manual\nintro\n" + TABLE
    assert find_section(page) == "SECTION 204-02: Rear Suspension"
    prose, blocks = split_table_blocks(page)
    assert len(blocks) == 1 and blocks[0].startswith("Torque Specifications") and "intro" in prose


def test_source_attribution_not_first_context():
    ctx = [{"text": "Tighten to 35 Nm. brake caliper", "chunk_id": "c1", "page_number": 10},
           {"text": "Shock absorber nuts 90 66", "chunk_id": "c2", "page_number": 128}]
    assert attribute_source("Shock absorber nuts", "90", ctx)["page_number"] == 128


def test_pick_model_falls_back_to_installed_tag():
    from src.extractor import pick_model
    assert pick_model("llama3", ["llama3:latest"]) == "llama3:latest"
    assert pick_model("llama3", ["llama3.2:3b", "mistral:7b"]) == "llama3.2:3b"
    assert pick_model("llama3", ["mistral:7b"]) == "mistral:7b"
    assert pick_model("llama3", []) is None


MATERIAL = """Material
Item Specification Fill Capacity
Additive Friction EST-M2C118-A —
Modifier
XL-3 (US); CXL-3
(Canada)
Motorcraft® SAE 75W- WSL-M2C192-A 2.84L (6.0 pt) (to
140 Synthetic Rear Axle and GL-5 the bottom of the
Lubricant filler hole)
XY-75W140-QL (US);
CXY-75W140-1L
(Canada)
Premium Long-Life ESA-M1C75-B —
Grease"""


def test_capacity_entries():
    from src.table_parser import parse_capacity_entries
    es = parse_capacity_entries(MATERIAL)
    assert len(es) == 1                       # "-1L" inside a product code is not a quantity
    assert (es[0].value, es[0].unit) == ("2.84", "L") and "Rear Axle Lubricant" in es[0].name


def test_capacity_query_routing():
    from src.table_parser import is_capacity_query, is_torque_query, parse_capacity_entries, query_capacity
    es = parse_capacity_entries(MATERIAL)
    assert is_capacity_query("rear axle lubricant capacity") and not is_torque_query("rear axle lubricant capacity")
    assert query_capacity("rear axle lubricant capacity", es, "SECTION 205-02A: Rear Drive Axle")
    assert query_capacity("brake caliper torque", es) == []


def test_grounding_rejects_irrelevant_and_wrong_type():
    from src.table_parser import is_grounded
    ctx = [{"text": "Tighten the nut to 350 Nm. Shock absorber nuts 90 66"}]
    assert is_grounded("Shock absorber nuts", "90", "Torque", "shock absorber nuts torque", ctx)
    assert not is_grounded("the nut to", "350", "Torque", "wiper arm nut torque", ctx)          # irrelevant fragment
    assert not is_grounded("Shock absorber nuts", "999", "Torque", "shock absorber nuts torque", ctx)  # value not in text
    assert not is_grounded("Rear axle lubricant", "90", "Torque", "rear axle lubricant capacity", ctx)  # wrong type


def test_grounding_rejects_invented_component_and_partial_match():
    from src.table_parser import is_grounded
    front = [{"text": "[SECTION 204-01A: Front Suspension]\nTorque Specifications\nTie-rod end nut 115 85 —\nShock rod nut 55 41 —"}]
    assert is_grounded("Tie-rod end nut", "115", "Torque", "tie rod end nut torque", front)
    assert not is_grounded("Wiper Arm Nut", "55", "Torque", "wiper arm nut torque", front)           # number exists, component invented
    air = [{"text": "Coolant expansion tank/lower Air Cleaner (ACL) housing assembly bolts 15 133"}]
    assert not is_grounded("Coolant expansion tank/lower Air Cleaner (ACL) housing assembly bolts", "15", "Torque",
                           "cabin air filter housing torque", air)                                   # query words missing from source


def test_alternative_spec_code_does_not_split_entry():
    from src.table_parser import parse_capacity_entries
    block = """Material
Item Specification Fill
Capacity
Motorcraft® High Performance WSS-M6C62-A 740 ml
DOT 3 Motor Vehicle Brake or WSS-M6C65- (1.56 pt)
Fluid (US) / Motorcraft® Brake A1
Fluid - Heavy Duty DOT 3
(Canada)"""
    es = parse_capacity_entries(block)
    assert len(es) == 1 and es[0].value == "740" and "Brake Fluid" in es[0].name
