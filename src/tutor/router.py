"""Rule-based router: the Capstone 6 routing table (Section 2.3) written as explicit rules.

One interface, handle(message) -> Reply, shared with the LLM Orchestrator (src/tutor/orchestrator.py), so a
scenario can be run on either. The rules never reach the answer key before an attempt is recorded.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.tutor.tools import Tutor

_LETTERS = r"[A-Da-d]"
_PREFIX = re.compile(r"^\s*(?:my\s+answers?\s*(?:is|are|:)?|answers?\s*(?:is|are|:)?|i\s+(?:choose|pick|say)|it'?s|it\s+is)\s*", re.I)
_ONLY_LETTERS = re.compile(rf"^\W*({_LETTERS})(?:\s*(?:,|;|&|and|\s)\s*({_LETTERS}))*\W*$", re.I)
_OOS = re.compile(r"\b(?:invent|make\s+up|generate|create|compose|write)\b.{0,30}\bquestions?\b|\bpredict\b|next\s+year'?s?\s+paper", re.I)
_REVEAL = re.compile(r"\b(?:tell|show|give|reveal)\b.{0,25}\b(?:answer|key|solution)\b|\bjust\s+the\s+answer\b|\bwhat\s*(?:'s|is)\s+the\s+answer\b", re.I)
_EXPLAIN = re.compile(r"\b(?:why|explain|how\s+come|more\s+detail)\b", re.I)
_PROGRESS = re.compile(r"how\s+am\s+i\s+doing|\bprogress\b|\bmastery\b|\bscore\b|\bstats?\b|how\s+many", re.I)
_NEXT = re.compile(r"\b(?:start|next|another|new\s+question|question|practice|practise|quiz|give\s+me|go\s+on)\b", re.I)
_END = re.compile(r"\b(?:end|finish|stop|bye|done)\b", re.I)

HELP = ("I can serve a past-paper question (say 'next question', or name a topic such as 'module D' or 'd_cp_02'), "
        "mark your answer (reply with the letter), explain it afterwards, and show your progress.")


@dataclass
class Reply:
    intent: str
    text: str
    tools: list = field(default_factory=list)
    data: dict = field(default_factory=dict)


class RuleRouter:
    def __init__(self, tutor: Tutor):
        self.t = tutor

    def handle(self, message: str) -> Reply:
        msg = (message or "").strip()
        t, s = self.t, self.t.state
        if not msg:
            return Reply("help", HELP)
        # 1. a bare answer to the open question
        body = _PREFIX.sub("", msg)
        if _ONLY_LETTERS.match(body):
            return self._answer(re.findall(_LETTERS, body))
        # 2. things the system does not do
        if _OOS.search(msg):
            return Reply("out_of_scope", "I only serve past-paper questions and teacher-approved items already in the bank; "
                         "I cannot invent questions or predict papers. " + HELP, [])
        # 3. reveal before attempting
        if _REVEAL.search(msg) and (s.pending_unit or not s.history):
            return Reply("reveal_declined", "I will not show the answer before you have tried. Pick the option you think is right "
                         "and I will mark it and explain.", [])
        # 4. explain what was just answered
        if _EXPLAIN.search(msg) or _REVEAL.search(msg):
            if not s.history:
                return Reply("explain_blocked", "Answer a question first, then I can explain it.", [])
            last = s.history[-1]
            e = t.explain(last["item_id"])
            return Reply("explain", self._explain_text(e), ["explain"], {"explanation": e})
        # 5. progress
        if _PROGRESS.search(msg):
            p = t.progress()
            return Reply("progress", self._progress_text(p), ["progress"], {"progress": p})
        # 6. next question
        if _NEXT.search(msg):
            return self._next(t.resolve_topic(msg))
        # 7. end
        if _END.search(msg):
            p = t.progress()
            return Reply("end", "Session summary: " + self._progress_text(p), ["progress"], {"progress": p})
        return Reply("help", HELP)

    # ---------------------------------------------------------------- helpers
    def _next(self, topic) -> Reply:
        r = self.t.next_question(topic)
        if r["status"] == "pending_open":
            return Reply("next", "You still have an open question:\n\n" + r["text"], ["next_question"], r)
        if r["status"] == "exhausted":
            alt = "; ".join(f"{a['tag']} ({a['name']}, {a['unseen_real_units']} unseen)" for a in r["alternatives"])
            tail = f" Other sub-topics in the same module with unseen questions: {alt}." if alt else ""
            return Reply("exhausted", r["message"] + tail, ["next_question"], r)
        head = (r["notice"] + "\n\n") if r.get("notice") else ""
        return Reply("next", head + r["text"] + "\n\nReply with the letter" + ("s in order." if len(r["item_ids"]) > 1 else "."),
                     ["next_question"], r)

    def _answer(self, letters: list) -> Reply:
        t, s = self.t, self.t.state
        if not s.pending_unit:
            return Reply("answer_blocked", "There is no open question. Say 'next question' to start one.", [])
        pend = list(s.pending)
        recs = [t.answer(i, L) for i, L in zip(pend, letters)]
        out = []
        for rec in recs:
            e = t.explain(rec["item_id"])
            out.append(self._explain_text(e))
        extra = ""
        if s.pending_unit:
            extra = f"\n\n{len(s.pending)} question(s) of this set still need an answer."
        return Reply("answer", "\n\n".join(out) + extra, ["answer", "explain"], {"records": recs})

    @staticmethod
    def _explain_text(e: dict) -> str:
        verdict = "Correct." if e["correct"] else f"Not quite - you chose {e['chosen']}; the answer is {e['correct_answer']}."
        refs = [r for r in e["references"] if r["grounded"]]
        ground = (f"\n\nCurriculum, Module {refs[0]['module']}:\n{refs[0]['text']}" if refs
                  else "\n\n(No curriculum text is available for this question's topic.)")
        note = f"\n\n{e['note']}" if e.get("note") else ""
        return f"{verdict}{ground}{note}"

    @staticmethod
    def _progress_text(p: dict) -> str:
        if not p["attempts"]:
            return "Nothing attempted yet."
        weak = ", ".join(p["weak_tags"]) or "none yet"
        return (f"{p['correct']} of {p['attempts']} correct; {p['units_served']} unit(s) served "
                f"({p['provisional_units_served']} admitted); weakest sub-topics so far: {weak}.")
