# Copyright (c) 2026 C10 Hackathon G1
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deep_researcher.schemas import ResearchPlan, SubQuestion


def test_plan_roundtrip():
    plan = ResearchPlan(question="Does remote work raise productivity?",
                        sub_questions=[SubQuestion(id="sq1", question="What do RCTs find?", source_types=["papers"])])
    assert plan.open_questions[0].id == "sq1"
    assert ResearchPlan.model_validate(plan.model_dump()) == plan


def test_plan_bounds():
    with pytest.raises(ValidationError):
        ResearchPlan(question="x", sub_questions=[])
    with pytest.raises(ValidationError):
        SubQuestion(id="bad id!", question="What do RCTs find?")
    with pytest.raises(ValidationError):
        SubQuestion(id="sq1", question="ok question", source_types=["tiktok"])
