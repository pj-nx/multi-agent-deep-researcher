import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def fake_graph_streams(request, monkeypatch):
    """API tests must not run the real graph (it needs an LLM key); give them canned streams.

    Opt out with ``@pytest.mark.real_graph``.
    """
    if request.node.get_closest_marker("real_graph"):
        return
    import deep_researcher.graph as G

    async def fake_stream_until_interrupt(*, question, thread_id, max_hops=1, budget_seconds=None,
                                          config=None, graph=None):
        yield "plan", {"status": "awaiting_approval", "thread_id": thread_id,
                       "plan": {"question": question, "sub_questions": [
                           {"id": "sq1", "question": "What do RCTs say?", "source_types": ["papers"]}]}}

    async def fake_resume_with_plan(*, thread_id, plan, budget_seconds=None, config=None, graph=None):
        yield "retrieve", {"node": "retrieve", "sub_question_id": "sq1", "sources": 2}
        yield "report", {"node": "report", "report": "# digest"}

    monkeypatch.setattr(G, "stream_until_interrupt", fake_stream_until_interrupt)
    monkeypatch.setattr(G, "resume_with_plan", fake_resume_with_plan)


def pytest_configure(config):
    config.addinivalue_line("markers", "real_graph: run against the real graph module (no canned streams)")
