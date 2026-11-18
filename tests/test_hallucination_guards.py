"""Anti-hallucination layers: evidence quoting, grounding, retrieval sufficiency (no Ollama needed)."""
import json

from src.extractor import OllamaExtractor, SmartExtractor
from src.table_parser import retrieval_sufficient, verify_evidence

FRONT = {"text": "[SECTION 204-01A: Front Suspension]\nTorque Specifications\nTie-rod end nut 115 85 —\nShock rod nut 55 41 —",
         "chunk_id": "chunk_00039", "page_number": 24}


class FakeOllama(OllamaExtractor):
    """Returns canned model output so the guards can be tested without a running Ollama."""

    def __init__(self, reply):
        super().__init__()
        self.reply = reply

    def _call_ollama(self, prompt):
        return json.dumps(self.reply)


def _item(component, value, unit, evidence, st="Torque"):
    return {"component": component, "spec_type": st, "value": value, "unit": unit, "evidence": evidence}


def test_verified_quote_is_accepted_with_its_page():
    reply = [_item("Tie-rod end nut", "115", "Nm", "Tie-rod end nut 115 85")]
    specs = FakeOllama(reply).extract("tie rod end nut torque", [FRONT])
    assert len(specs) == 1 and specs[0].page_number == 24 and specs[0].source_chunk_id == "chunk_00039"


def test_invented_quote_is_dropped():
    reply = [_item("Wiper Arm Nut", "55", "Nm", "Wiper arm nut torque is 55 Nm")]        # quote not in the manual text
    assert FakeOllama(reply).extract("wiper arm nut torque", [FRONT]) == []


def test_real_number_with_wrong_component_is_dropped():
    reply = [_item("Wiper Arm Nut", "55", "Nm", "Shock rod nut 55 41")]                  # real quote, wrong part
    specs = FakeOllama(reply).extract("wiper arm nut torque", [FRONT])
    assert SmartExtractor._keep_grounded(specs, [FRONT], "wiper arm nut torque", "ollama") == []


def test_verify_evidence_ignores_whitespace_but_not_content():
    assert verify_evidence("Tie-rod  end nut\n115", "115", [FRONT])
    assert verify_evidence("Tie-rod end nut 999", "999", [FRONT]) is None
    assert verify_evidence("nut", "115", [FRONT]) is None            # too short to be evidence


def test_retrieval_sufficiency_gate():
    assert retrieval_sufficient("tie rod end nut torque", [FRONT])
    assert not retrieval_sufficient("wiper arm nut torque", [FRONT])
    assert not retrieval_sufficient("cabin air filter housing torque", [FRONT])
