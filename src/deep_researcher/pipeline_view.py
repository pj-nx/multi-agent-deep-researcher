"""Pipeline progress: turns the stream of graph events into a row of stage boxes (HTML).

Pure Python so it can be unit-tested; the Gradio UI renders :meth:`PipelineProgress.render` in a
``gr.HTML`` after every event.
"""
from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import Any

STAGES: tuple[tuple[str, str], ...] = (
    ("plan", "Planner"),
    ("retrieve", "Retrievers"),
    ("validate", "Validator"),
    ("analyze", "Analysis"),
    ("insight", "Insights"),
    ("critic", "Critic"),
    ("report", "Report"),
)
ORDER = [k for k, _ in STAGES]
ROLE_STAGE = {"planner": "plan", "validator": "validate", "analysis": "analyze", "insight": "insight",
              "critic": "critic", "report": "report"}

CSS = """
<style>
.dr-pipe{display:flex;flex-wrap:wrap;align-items:stretch;gap:8px;margin:2px 0 14px 0;
         font-family:Inter,ui-sans-serif,system-ui,sans-serif;font-size:12px}
.dr-box{flex:1 1 120px;min-width:120px;border:1.5px solid #D4D4D8;border-radius:10px;padding:8px 10px;
        background:#FFFFFF;color:#0A0A0A;position:relative;transition:background .2s,border-color .2s}
.dr-box .t{font-weight:600;font-size:13px;letter-spacing:-0.01em}
.dr-box .n{color:#71717A;margin-top:3px;min-height:15px;white-space:nowrap;overflow:hidden;
           text-overflow:ellipsis;font-size:11.5px}
.dr-box.done{border-color:#00CC00;background:#E6FFE6}
.dr-box.done .n{color:#006600}
.dr-box.done .t::before{content:"\\2713  ";color:#00AA00}
.dr-box.active{border-color:#0A0A0A;background:#0A0A0A;color:#FFFFFF;
               animation:dr-pulse 1.4s ease-in-out infinite}
.dr-box.active .n{color:#88FF88}
.dr-box.active .t::before{content:"\\25B6  ";color:#00CC00}
.dr-box.waiting{border-color:#0A0A0A;border-style:dashed}
.dr-box.waiting .t::before{content:"\\23F8  ";color:#0A0A0A}
.dr-box.error{border-color:#DC2626;background:#FEE2E2}
.dr-box.error .t::before{content:"\\2716  ";color:#DC2626}
.dr-hop{position:absolute;top:6px;right:8px;font-size:10px;font-weight:600;color:#006600;
        background:#E6FFE6;border-radius:999px;padding:1px 6px}
.dr-box.active .dr-hop{background:#003300;color:#88FF88}
@keyframes dr-pulse{0%,100%{box-shadow:0 0 0 0 rgba(0,204,0,.6)}
                    50%{box-shadow:0 0 0 6px rgba(0,204,0,0)}}
.dark .dr-box{border-color:#006600;background:#080808;color:#DDFFDD}
.dark .dr-box .n{color:#88FF88}
.dark .dr-box.done{border-color:#00CC00;background:#003300}
.dark .dr-box.done .n{color:#BBFFBB}
.dark .dr-box.active{border-color:#00CC00;background:#00CC00;color:#000000}
.dark .dr-box.active .n{color:#003300}
.dark .dr-box.active .t::before{color:#003300}
.dark .dr-box.waiting{border-color:#88FF88}
.dark .dr-box.waiting .t::before{color:#88FF88}
.dark .dr-box.error{border-color:#B40000;background:#2A0000}
.dark .dr-hop{background:#003300;color:#88FF88}
.dark .dr-box.active .dr-hop{background:#003300;color:#88FF88}
</style>
"""


@dataclass
class Stage:
    key: str
    label: str
    status: str = "pending"        # pending | active | waiting | done | error
    note: str = ""
    hop: int | None = None


@dataclass
class PipelineProgress:
    stages: dict[str, Stage] = field(default_factory=lambda: {k: Stage(k, label) for k, label in STAGES})
    expected: int = 0              # sub-questions in the current retrieval hop
    done_in_hop: int = 0
    hop: int = 0
    insight_round: int = 0

    # ------------------------------------------------------------------ helpers
    def _set(self, key: str, status: str | None = None, note: str | None = None,
             hop: int | None = None) -> None:
        st = self.stages[key]
        if status is not None:
            st.status = status
        if note is not None:
            st.note = note
        if hop is not None:
            st.hop = hop

    def _activate(self, key: str, note: str | None = None) -> None:
        """Mark ``key`` active and every earlier non-done stage done (a stage cannot start before its
        predecessors finished); later stages stay as they are."""
        for k in ORDER[: ORDER.index(key)]:
            if self.stages[k].status in ("active", "waiting"):
                self.stages[k].status = "done"
        self._set(key, "active", note)

    @classmethod
    def after_plan(cls, sub_questions: int) -> PipelineProgress:
        p = cls()
        p._set("plan", "done", f"{sub_questions} sub-questions")
        p.expected = sub_questions
        return p

    # ------------------------------------------------------------------ event feed
    def update(self, event: str, data: dict[str, Any]) -> None:
        if event == "activity":
            self._activity(data)
            return
        if event == "started":
            self._activate("plan", "starting...")
        elif event == "plan":
            if data.get("status") == "awaiting_approval":
                n = len((data.get("plan") or {}).get("sub_questions", []))
                self.expected = n
                self._set("plan", "done", f"{n} sub-questions")
                self._set("retrieve", "waiting", "awaiting approval")
            else:
                self._set("plan", "done", "; ".join(data.get("notes", []))[:40] or "planned")
        elif event == "approve":
            self.done_in_hop = 0
            self._activate("retrieve", f"0/{self.expected} sub-questions")
        elif event == "retrieve":
            self.done_in_hop += 1
            hop = int(data.get("hop") or 0)
            note = f"{self.done_in_hop}/{self.expected or '?'} sub-questions"
            if data.get("skipped"):
                note = f"{self.done_in_hop}/{self.expected or '?'} (skipped: budget)"
            self._activate("retrieve", note)
            self._set("retrieve", hop=hop or None)
        elif event == "validate":
            self._set("retrieve", "done")
            self._set("validate", "done", f"kept {data.get('kept', 0)}/{data.get('assessed', 0)}")
            self._activate("analyze", "analysing...")
        elif event == "analyze":
            note = f"{data.get('claims', 0)} claims, {data.get('contradictions', 0)} contradictions"
            if data.get("gaps"):
                note += f", {len(data['gaps'])} gaps"
            self._set("analyze", "done", note)
        elif event == "route_gaps":
            nxt = data.get("next")
            if nxt == "retrieve":
                self.hop = int(data.get("hop") or self.hop + 1)
                self.expected = len(data.get("new_sub_questions") or [])
                self.done_in_hop = 0
                for k in ("retrieve", "validate", "analyze"):
                    self._set(k, "pending", hop=self.hop)
                self._activate("retrieve", f"hop {self.hop}: 0/{self.expected}")
            elif nxt == "report":
                self._activate("report", "writing...")
            else:
                self._activate("insight", "generating...")
        elif event == "insight":
            n = len(data.get("insights") or [])
            self._set("insight", "done", f"{n} insights")
            self._activate("critic", "reviewing...")
        elif event == "critic":
            verdict = data.get("verdict") or "?"
            self._set("critic", "done", f"round {data.get('round')}: {verdict}")
            if verdict == "revise":
                self._set("insight", "pending")
                self._activate("insight", "revising...")
                self._set("critic", "pending", f"round {data.get('round')}: revise")
            elif verdict == "need_more_evidence":
                self._set("critic", "done", f"round {data.get('round')}: more evidence")
            else:
                self._activate("report", "writing...")
        elif event == "report":
            stats = data.get("stats") or {}
            note = f"{stats.get('kept', '?')} sources, {stats.get('claims', '?')} claims"
            self._set("report", "done", note)
            for k in ORDER:
                if self.stages[k].status in ("active", "waiting"):
                    self.stages[k].status = "done"
        elif event == "error":
            for k in ORDER:
                if self.stages[k].status == "active":
                    self._set(k, "error", str(data.get("message") or "failed")[:60])
                    break
        elif event == "done":
            pass

    def _activity(self, data: dict[str, Any]) -> None:
        kind = data.get("kind")
        stage = ROLE_STAGE.get(str(data.get("role")))
        if kind == "llm_start" and stage:
            model = str(data.get("model") or "model").split("/")[-1]
            label = data.get("label")
            note = f"{label}: {model}" if label and stage == "analyze" else f"calling {model}"
            if stage == "validate" and self.stages["retrieve"].status == "active":
                self._set("validate", "active", note)      # grading inside a retrieve branch: both run
            else:
                self._activate(stage, note)
        elif kind == "llm_end" and stage:
            tokens = data.get("tokens")
            note = f"{data.get('elapsed_s')}s" + (f", {tokens / 1000:.1f}k tok" if tokens else "")
            if self.stages[stage].status == "active":
                self._set(stage, note=note)
        elif kind == "llm_error" and stage:
            self._set(stage, "error", str(data.get("error") or "failed")[:60])
        elif kind in ("tool_start", "tool_end", "tool_timeout"):
            if self.stages["retrieve"].status == "active":
                verb = {"tool_start": "searching", "tool_end": "got", "tool_timeout": "timeout"}[kind]
                extra = f" {data.get('results')}" if kind == "tool_end" else ""
                self._set("retrieve", note=f"{data.get('sub_question_id')}: {verb} {data.get('tool')}{extra}")
        elif kind == "embed_start" and self.stages["retrieve"].status == "active":
            self._set("retrieve", note=f"{data.get('sub_question_id')}: embedding {data.get('sources')}")

    # ------------------------------------------------------------------ render
    def render(self) -> str:
        boxes = []
        for st in self.stages.values():
            hop = f"<span class='dr-hop'>hop {st.hop}</span>" if st.hop else ""
            boxes.append(f"<div class='dr-box {st.status}'>{hop}<div class='t'>{html.escape(st.label)}</div>"
                         f"<div class='n' title='{html.escape(st.note)}'>{html.escape(st.note)}</div></div>")
        return CSS + "<div class='dr-pipe'>" + "".join(boxes) + "</div>"

    def summary(self) -> dict[str, str]:
        return {k: s.status for k, s in self.stages.items()}
