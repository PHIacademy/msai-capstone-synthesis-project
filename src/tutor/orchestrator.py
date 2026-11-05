"""LLM Orchestrator and its Explainer worker: the second router of the hybrid tutor (Section 4.13 of the notebook).

Same interface as RuleRouter: ``handle(message) -> Reply``. A page or a test can use either.

The Orchestrator is a bounded tool-using agent. For one student message it runs a think -> call a tool -> observe
loop of at most MAX_STEPS steps: the model reads the message and a one-line state, chooses one of five Tutor
tools, receives a short status line (never question text, the key, a verdict or an explanation) and either calls
another tool or stops. What the student reads is not written by the model: every tool writes its own text, and
the reply is assembled by code from those texts in order. What is reused from Capstone 6 is the pattern (a model
choosing among least-privilege tools, a worker with its own persona, loop and tool), not the five-agent
smolagents/LiteLLM stack. The Analyst is left out: difficulty analysis rests on ``cohort_pct``, which is not
calibrated across years and is absent for generated items (decision 13).

The Explainer is the one worker. It is called by the ``explain`` tool, only after an attempt and only for a past-paper
question that is text only. It has one tool (get_attempt_context) and at most EXPLAINER_MAX_STEPS steps: fetch the
context, then write at most 120 words. Its text goes to code, never back to the Orchestrator. The verdict, the key,
the label and the module text are added by code; a text that names another letter as correct, or is empty, is
rejected and counted, and the fixed fallback message replaces it (there is no retry). The same message is shown when
the worker's call fails. Admitted items and questions with pictures get the verdict, the key and the module text; an
admitted item adds "No AI explanation is given for this item."
"""
from __future__ import annotations

import os
import re

from src.data import IMAGE_SENTINEL, strip_paper_end
from src.tutor.router import HELP, Reply, RuleRouter
from src.tutor.tools import Tutor

# No temperature argument is passed: the client used with the Vocareum proxy refuses one (as for the Haiku backend, Section 4.8.2),
# so both the Orchestrator and the Explainer run at the API default.
MODEL = "claude-haiku-4-5-20251001"
DEFAULT_BASE_URL = "https://claude.vocareum.com"          # same default as config/generators.json (haiku_prompted)
TIMEOUT = 30                                              # seconds per model call
MAX_STEPS = 4                                             # tool steps of the Orchestrator per message
EXPLAINER_MAX_STEPS = 3                                   # model calls of the Explainer per explanation

AI_LABEL = "Explanation written by AI"
NO_AI_LINE = "No AI explanation is given for this item."
FALLBACK_MESSAGE = "The explanation is not available right now. Here is the answer and the curriculum text instead."
ROUTE_FAILURE_MESSAGE = ("The assistant is not available right now. Use the Next question button, or switch the "
                         "tutor mode to Rules in the sidebar.")
OUT_OF_SCOPE_TEXT = ("I only serve past-paper questions and teacher-approved items already in the bank; "
                     "I cannot invent questions or predict papers. " + HELP)          # same words as RuleRouter
REVEAL_TEXT = ("I will not show the answer before you have tried. Pick the option you think is right "
               "and I will mark it and explain.")

_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")

ROUTER_SYSTEM = (
    "You are a patient study coach routing one student message for a practice tutor for HKDSE ICT Paper 1A "
    "multiple-choice questions. You work in steps: call a tool, read its status line, then call another tool only "
    "if the message needs one (for example answer, then next_question); otherwise stop. You never write questions, "
    "answers or explanations yourself: the tools write everything the student reads, so do not write any reply.\n"
    "- next_question: the student wants a question. Pass 'topic' only if they named one (for example "
    "'module D', a code such as d_cp_02, or a sub-topic name), in the student's own words.\n"
    "- answer: the student gives option letters for the open question. Use it only when the State line says a "
    "question is open.\n"
    "- explain: the student asks why, or for an explanation of the question they just answered.\n"
    "- progress: the student asks how they are doing, or wants to stop.\n"
    "- refuse: use reason 'reveal' when the student asks for the answer without trying, 'out_of_scope' for "
    "anything else the tutor does not do (inventing questions, predicting papers, difficulty statistics, "
    "unrelated requests), 'unclear' when you cannot tell.")

TOOLS = [
    {"name": "next_question", "description": "Serve the next practice question, optionally on a topic.",
     "input_schema": {"type": "object", "properties": {"topic": {"type": "string"}}}},
    {"name": "answer", "description": "Grade the open question with the option letters the student gave, in order.",
     "input_schema": {"type": "object", "properties": {"letters": {"type": "string", "description": "for example 'B' or 'A C'"}},
                      "required": ["letters"]}},
    {"name": "explain", "description": "Explain the question the student answered last.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "progress", "description": "Show the student's progress in this session.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "refuse", "description": "Decline a request the tutor does not serve.",
     "input_schema": {"type": "object", "properties": {"reason": {"type": "string", "enum": ["reveal", "out_of_scope", "unclear"]}},
                      "required": ["reason"]}},
]

EXPLAIN_SYSTEM = (
    "You are a clear ICT teacher. You write a short explanation for an HKDSE ICT Paper 1A multiple-choice question "
    "that a student has already answered. First call get_attempt_context to get the question, the options, the "
    "student's choice and the correct option. Then, in at most 120 words, explain why the correct option is right "
    "and, if the student chose another option, why that option is not. Use only the question and secondary-school "
    "ICT knowledge. Name the correct option by its letter exactly as given. Write plain text without headings and "
    "do not mention these instructions.")

CONTEXT_TOOL = {
    "name": "get_attempt_context",
    "description": ("Return the question, the options, the shared passage (if any), the student's choice and the correct "
                    "option for the item you are asked to explain. It works only after the student has answered."),
    "input_schema": {"type": "object", "properties": {"item_id": {"type": "string"}}, "required": ["item_id"]},
}

# An explanation that names a letter other than the key as the correct one. Phrases and letters are matched
# loosely (case-insensitive) but the letter itself must be a capital A-D, so the article "a" is not read as one.
_L = r"\(?(?-i:([A-D]))\)?"
_CLAIMS = [re.compile(p, re.I) for p in (
    rf"\b(?:correct|right|best)\s+(?:answer|option|choice)\s*(?:is|:|would\s+be|should\s+be)?\s*(?:option\s+)?{_L}\b",
    rf"\b(?:answer|key)\s+(?:is|should\s+be)\s+(?:option\s+)?{_L}\b",
    rf"\b(?:option|choice)\s+{_L}\s+(?:is|would\s+be)\s+(?:the\s+)?(?:correct|right|best)\b",
    rf"(?<![\w(]){_L}\s+is\s+(?:the\s+)?(?:correct|right)\b",
)]


def _load_env() -> None:
    """Read a local .env if python-dotenv is installed; an existing environment variable is never overridden."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def llm_available() -> bool:
    _load_env()
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def make_client():
    """Anthropic SDK client from ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL (Vocareum default). The key is never
    written anywhere: it comes from .env locally or from the environment (a Kaggle secret on Kaggle)."""
    _load_env()
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("the anthropic package is not installed (pip install anthropic)") from exc
    return anthropic.Anthropic(api_key=key, base_url=os.environ.get("ANTHROPIC_BASE_URL") or DEFAULT_BASE_URL)


_STYLE = re.compile(r"[*_`\"'\u201c\u201d\u2018\u2019]|</?[A-Za-z][^>]*>")   # bold, italics, code, quotes, simple HTML tags


def claimed_letters(text: str) -> set:
    """The option letters that an explanation names as the correct one. Styling around a letter (**C**, _C_, `C`,
    "C", <b>C</b>) is removed first, so a wrong letter in bold cannot slip through."""
    plain = _STYLE.sub("", (text or "").replace("\u00a0", " "))
    return {m.group(1) for rx in _CLAIMS for m in rx.finditer(plain)}


def _blank(v) -> bool:
    return v is None or str(v).strip() in ("", "nan", "None", "<NA>")


def _uses(resp) -> list:
    return [b for b in resp.content if getattr(b, "type", None) == "tool_use"]


class Explainer:
    """The one worker: persona (EXPLAIN_SYSTEM), a loop of at most EXPLAINER_MAX_STEPS model calls and one tool,
    get_attempt_context. Stateless per call. It returns text; the Orchestrator's code checks and labels it."""

    def __init__(self, owner: "Orchestrator"):
        self.o = owner

    def context(self, item_id: str, assigned: str) -> str:
        """The tool: read only, for the one assigned item, refused before an attempt and for any item that may not get
        model text (an admitted item, a question with a picture)."""
        t = self.o.t
        if item_id != assigned:
            raise ValueError("this worker may only explain the item it was given")
        rec = next((h for h in reversed(t.state.history) if h["item_id"] == item_id), None)
        if rec is None:
            raise ValueError("no attempt is recorded for this item")
        row = t.by_id.loc[item_id]
        if not Orchestrator.ai_eligible(row):
            raise ValueError("this item does not get a model-written explanation")
        parts = []
        if not _blank(row["cluster_desc"]):                  # for a pair the shared passage is part of the question
            parts.append("Shared passage:\n" + str(row["cluster_desc"]).strip())
        parts.append("Question:\n" + str(row["question_text"]).strip())
        parts.append("Options:\n" + strip_paper_end(str(row["answer_options"])).strip())
        parts.append(f"Student's choice: {rec['chosen']}\nCorrect option: {rec['correct_answer']}")
        return "\n\n".join(parts)

    def write(self, item_id: str) -> str:
        """The explanation text. Raises when the model call fails, the step limit is reached, or the text was written
        without fetching the context."""
        o = self.o
        messages = [{"role": "user", "content": f"Explain item {item_id} for the student. First call "
                                                  "get_attempt_context, then write the explanation."}]
        fetched = False
        for step in range(EXPLAINER_MAX_STEPS):
            resp = o._create(max_tokens=400, system=EXPLAIN_SYSTEM, tools=[CONTEXT_TOOL],
                             tool_choice={"type": "any" if step == 0 else "auto", "disable_parallel_tool_use": True},
                             messages=list(messages))
            o.stats["explainer_calls"] += 1
            uses = _uses(resp)
            if uses:
                messages.append({"role": "assistant", "content": resp.content})
                results = []
                for i, b in enumerate(uses):
                    block = {"type": "tool_result", "tool_use_id": getattr(b, "id", f"x{step}-{i}")}
                    try:
                        if i:
                            raise ValueError("one tool call per step")
                        block["content"] = self.context(str((b.input or {}).get("item_id", "")), item_id)
                        fetched = True
                    except ValueError as exc:
                        block["content"], block["is_error"] = str(exc), True
                    results.append(block)
                messages.append({"role": "user", "content": results})
                continue
            if not fetched:
                raise RuntimeError("the explanation was written without the context")
            return "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text").strip()
        raise RuntimeError("the Explainer reached its step limit")


class Orchestrator:
    def __init__(self, tutor: Tutor, client=None, model: str = MODEL, max_steps: int = MAX_STEPS):
        self.t = tutor
        self.client = client                       # None: made on first use, so building the class never needs a key
        self.model = model
        self.max_steps = max_steps
        self._rules = RuleRouter(tutor)            # only for text the two routers share (question, grading, progress)
        self._worker = Explainer(self)
        self._served_now = False                   # a question was served in this very turn
        self._timeout_ok = True                    # some clients refuse a per-call timeout; then it is dropped
        self.errors: list = []                     # the last failures as "Type: message" (never the key), for the notebook
        self.stats = {"routing_calls": 0, "steps": 0, "step_limit_hits": 0, "explainer_calls": 0,
                      "ai_explanations": 0, "rejections": 0, "failures": 0}

    # ------------------------------------------------------------------ the model
    def _create(self, **kw):
        if self.client is None:
            self.client = make_client()
        if self._timeout_ok:
            try:
                return self.client.messages.create(model=self.model, timeout=TIMEOUT, **kw)
            except TypeError as exc:
                if "timeout" not in str(exc):
                    raise
                self._timeout_ok = False
        return self.client.messages.create(model=self.model, **kw)

    def _note(self, exc: Exception) -> str:
        msg = f"{type(exc).__name__}: {str(exc)[:200]}"
        self.errors = (self.errors + [msg])[-5:]
        return msg

    def _state(self) -> str:
        s = self.t.state
        return f"State: open_question={'yes' if s.pending_unit else 'no'}; answered={len(s.history)}"

    # ------------------------------------------------------------------ interface: the loop
    def handle(self, message: str) -> Reply:
        msg = (message or "").strip()
        if not msg:
            return Reply("help", HELP)
        self._served_now = False
        parts: list = []
        messages = [{"role": "user", "content": f"{self._state()}\n\n{msg}"}]
        steps, stopped = 0, False
        for step in range(self.max_steps):
            try:
                resp = self._create(max_tokens=300, system=ROUTER_SYSTEM, tools=TOOLS,
                                    tool_choice={"type": "any" if step == 0 else "auto", "disable_parallel_tool_use": True},
                                    messages=list(messages))
                self.stats["routing_calls"] += 1
            except Exception as exc:               # no key, no package, network, timeout
                self.stats["failures"] += 1
                msg_err = self._note(exc)
                if not parts:
                    return Reply("model_unavailable", ROUTE_FAILURE_MESSAGE, [], {"router": "model", "error": msg_err})
                stopped = True                      # keep what the tools have already written
                break
            uses = _uses(resp)
            if not uses:                            # the model stops
                stopped = True
                break
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for i, b in enumerate(uses):
                if i == 0:
                    reply, status = self._run_tool(b.name, dict(b.input or {}))
                    if reply is not None and all(reply.text != p.text for p in parts):
                        parts.append(reply)
                else:
                    status = "skipped: one tool per step"
                results.append({"type": "tool_result", "tool_use_id": getattr(b, "id", f"t{step}-{i}"),
                                "content": f"{status}\n{self._state()}"})     # the observation: a status line and the state
            messages.append({"role": "user", "content": results})
            steps += 1
            self.stats["steps"] += 1
        if not stopped:
            self.stats["step_limit_hits"] += 1
        return self._assemble(parts, steps)

    def _assemble(self, parts: list, steps: int) -> Reply:
        """The reply is built by code from the texts the tools wrote; anything the model wrote is ignored."""
        if not parts:
            reply = Reply("help", HELP)
        else:
            data: dict = {}
            for p in parts:
                data.update(p.data)
            reply = Reply(parts[0].intent, "\n\n".join(p.text for p in parts),
                          [t for p in parts for t in p.tools], data)
            reply.data["parts"] = [p.intent for p in parts]
        reply.data["router"], reply.data["steps"] = "model", steps
        return reply

    # ------------------------------------------------------------------ the five tools
    def _run_tool(self, name: str, args: dict):
        """(Reply or None, status line). The status line is all that the model sees of what the tool did."""
        t, s = self.t, self.t.state
        if name == "next_question":
            topic = str(args.get("topic") or "").strip()
            resolved = t.resolve_topic(topic) if topic else None
            if topic and resolved is None:         # never fall back to another topic silently
                return (Reply("topic_unknown", "I could not match that topic. Name a module (for example 'module D') "
                              "or a sub-topic code such as d_cp_02.", []), "topic not matched")
            r = self._rules._next(resolved)
            st = r.data.get("status")
            if r.intent == "exhausted":
                return r, "nothing left for this topic"
            if st == "pending_open":
                return (None if self._served_now else r), "a question is already open"
            self._served_now = True
            n = len(r.data.get("item_ids", []))
            return r, "served: 1 question" if n == 1 else f"served: a set of {n} questions"
        if name == "answer":
            raw = args.get("letters", "")
            letters = [c.upper() for c in re.findall(r"[A-Da-d]", " ".join(map(str, raw)) if isinstance(raw, list) else str(raw))]
            if not letters:
                return Reply("help", "Reply with the letter of your answer, for example B.", []), "no letters given"
            if self._served_now:                   # the student has not seen this question yet
                return (Reply("answer_blocked", "Answer the question after you have seen it.", []),
                        "refused: the question was only just served")
            r = self._rules._answer(letters)
            if r.intent != "answer":
                return r, "no open question"
            return r, "answer recorded" + (f"; {len(s.pending)} still open" if s.pending_unit else "")
        if name == "explain":
            return self._explain_with_status()
        if name == "progress":
            p = t.progress()
            return Reply("progress", RuleRouter._progress_text(p), ["progress"], {"progress": p}), "progress shown"
        if name == "refuse":
            reason = args.get("reason")
            if reason == "reveal":                 # same rule as RuleRouter: after an attempt, with nothing open, it is an explain request
                if s.pending_unit or not s.history:
                    return Reply("reveal_declined", REVEAL_TEXT, []), "refused: reveal"
                return self._explain_with_status()
            if reason == "out_of_scope":
                return Reply("out_of_scope", OUT_OF_SCOPE_TEXT, []), "refused: out of scope"
            return Reply("help", HELP), "refused: unclear"
        return None, "unknown tool"

    # ------------------------------------------------------------------ explanations
    @staticmethod
    def ai_eligible(row) -> bool:
        """A model-written explanation is allowed only for a past-paper question that is text only: source hkeaa,
        no question picture, no option picture, no picture left in the stem or the shared passage."""
        if row["source"] != "hkeaa" or bool(row["provisional"]):
            return False
        if not _blank(row["qn_image_filename"]) or not _blank(row.get("ans_image_filename")):
            return False
        if str(row["answer_options"]).strip() == IMAGE_SENTINEL:
            return False
        return not any(_MD_IMAGE.search(str(row[c])) for c in ("question_text", "cluster_desc") if not _blank(row[c]))

    @staticmethod
    def _verdict(e: dict) -> str:
        if e["correct"]:
            return f"Correct. The answer is {e['correct_answer']}."
        return f"Not quite - you chose {e['chosen']}; the answer is {e['correct_answer']}."

    @staticmethod
    def _ground(e: dict) -> str:
        refs = [r for r in e["references"] if r["grounded"]]
        text = (f"Curriculum, Module {refs[0]['module']}:\n{refs[0]['text']}" if refs
                else "(No curriculum text is available for this question's topic.)")
        return text + (f"\n\n{e['note']}" if e.get("note") else "")

    def _explain_with_status(self):
        reply = self._explain()
        if reply.intent == "explain_blocked":
            return reply, "no attempt yet"
        d = reply.data
        return reply, "fallback shown" if (d.get("failed") or d.get("rejected")) else "explanation shown"

    def _explain(self) -> Reply:
        t, s = self.t, self.t.state
        if not s.history:                          # code, not the model, decides: nothing is explained before an attempt
            return Reply("explain_blocked", "Answer a question first, then I can explain it.", [])
        last = s.history[-1]
        e = t.explain(last["item_id"])
        row = t.by_id.loc[last["item_id"]]
        verdict, ground = self._verdict(e), self._ground(e)
        data = {"explanation": e, "ai": False, "rejected": False, "failed": False}
        if e["provisional"]:
            return Reply("explain", f"{verdict}\n\n{ground}\n\n{NO_AI_LINE}", ["explain"], data)
        if not self.ai_eligible(row):
            return Reply("explain", f"{verdict}\n\n{ground}", ["explain"], data)
        text, problem = self._ai_text(last["item_id"], e)
        if text is None:
            data["rejected"], data["failed"] = problem == "rejected", problem == "failed"
            return Reply("explain", f"{FALLBACK_MESSAGE}\n\n{verdict}\n\n{ground}", ["explain"], data)
        data["ai"] = True
        return Reply("explain", f"{verdict}\n\n**{AI_LABEL}**\n\n{text}\n\n{ground}", ["explain", "model_explanation"], data)

    def _ai_text(self, item_id: str, e: dict):
        """(text, None) or (None, 'failed' | 'rejected'). The Explainer worker writes; this code checks."""
        try:
            text = self._worker.write(item_id)
        except Exception as exc:
            self.stats["failures"] += 1
            self._note(exc)
            return None, "failed"
        if not text or len(text) > 2000 or (claimed_letters(text) - {e["correct_answer"]}):
            self.stats["rejections"] += 1          # empty, runaway, or it names another letter as correct
            return None, "rejected"
        self.stats["ai_explanations"] += 1
        return text, None
