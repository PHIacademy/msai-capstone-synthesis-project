"""Tests for src/tutor/orchestrator.py with a stub client (offline: no key, no network, no model)."""
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.tutor import Orchestrator, PROVENANCE_LABEL, Reply, RuleRouter, Tutor, llm_available    # noqa: E402
from src.tutor import orchestrator as O                                                           # noqa: E402
from src.tutor.orchestrator import Explainer                                                      # noqa: E402
from tests.test_tutor import TAX, make_eb, serve_and_answer                                        # noqa: E402

CURRICULUM = {"D": "Module D text", "A": "Module A text"}
_ids = itertools.count(1)


def tool(name, **args):
    return NS(content=[NS(type="tool_use", id=f"tu{next(_ids)}", name=name, input=args)])


def text(s):
    return NS(content=[NS(type="text", text=s)])


def done():
    return text("done")


def ctx(item):
    return tool("get_attempt_context", item_id=item)


class Exhausted(BaseException):
    """Raised when a test asks for more model calls than it scripted (not caught by the class under test)."""


class Stub:
    """Plays back scripted replies in order and records every request. An Exception in the script is raised."""
    def __init__(self, *replies):
        self.replies, self.calls, self.messages = list(replies), [], self

    def create(self, **kw):
        self.calls.append(kw)
        if not self.replies:
            raise Exhausted("more model calls than scripted")
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def make(*replies, eb=None, **kw):
    t = Tutor(eb if eb is not None else make_eb(), TAX, CURRICULUM)
    stub = Stub(*replies)
    return Orchestrator(t, stub, **kw), t, stub


def routing_calls(stub):
    return [c for c in stub.calls if c["system"] == O.ROUTER_SYSTEM]


def worker_calls(stub):
    return [c for c in stub.calls if c["system"] == O.EXPLAIN_SYSTEM]


def observation(call):
    """The tool results the orchestrating model was shown at the end of a request."""
    return [b["content"] for b in call["messages"][-1]["content"]]


def to_explain(*after, eb=None, wrong=True):
    """An Orchestrator whose student has been served r1 (key B) and answered it, ready for 'why?'."""
    o, t, stub = make(tool("next_question", topic="d_cp_02"), done(), tool("answer", letters="A" if wrong else "B"), done(),
                      *after, eb=eb)
    o.handle("next"); o.handle("A")
    return o, t, stub


# ------------------------------------------------------------------ the loop and what the model is shown
def test_same_interface_and_same_question_text_as_the_rule_router():
    o, t, stub = make(tool("next_question", topic="d_cp_02"), done())
    r = o.handle("give me a question on data types")
    assert isinstance(r, Reply) and r.intent == "next" and r.data["router"] == "model" and r.data["steps"] == 1
    rules = RuleRouter(Tutor(make_eb(), TAX, CURRICULUM)).handle("next question on d_cp_02")
    assert r.text == rules.text and len(stub.calls) == 2


def test_no_call_passes_a_temperature_because_the_client_refuses_one():
    o, t, stub = to_explain(tool("explain"), ctx("r1"), text("The correct answer is B."), done())
    o.handle("why?")
    assert stub.calls and all("temperature" not in c for c in stub.calls)


def test_a_client_that_refuses_a_timeout_argument_is_called_without_it():
    class NoTimeout(Stub):
        def create(self, **kw):
            if "timeout" in kw:
                self.calls.append(kw)
                raise TypeError("Messages.create() got an unexpected keyword argument 'timeout'")
            return super().create(**kw)
    stub = NoTimeout(tool("next_question", topic="d_cp_02"), done())
    o = Orchestrator(Tutor(make_eb(), TAX, CURRICULUM), stub)
    r = o.handle("next")
    assert r.intent == "next" and "Stem of r1" in r.text and o.stats["failures"] == 0
    assert stub.calls[0].get("timeout") == O.TIMEOUT and all("timeout" not in c for c in stub.calls[1:])


def test_a_failure_keeps_its_type_and_message_but_the_student_sees_the_fixed_text():
    o, t, stub = make(RuntimeError("proxy refused the request"))
    r = o.handle("next")
    assert r.text == O.ROUTE_FAILURE_MESSAGE and r.data["error"] == "RuntimeError: proxy refused the request"
    assert o.errors == ["RuntimeError: proxy refused the request"]


def test_a_missing_anthropic_package_is_named_in_the_error(monkeypatch):
    monkeypatch.setattr(O, "_load_env", lambda: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder-not-a-key")
    monkeypatch.setitem(sys.modules, "anthropic", None)                      # makes "import anthropic" fail
    r = Orchestrator(Tutor(make_eb(), TAX, CURRICULUM)).handle("next question")
    assert r.intent == "model_unavailable" and "pip install anthropic" in r.data["error"]


def test_the_first_call_is_forced_later_calls_may_stop_and_only_message_and_state_are_sent():
    o, t, stub = make(tool("next_question", topic="d_cp_02"), done())
    o.handle("next question on d_cp_02")
    first, second = stub.calls
    assert first["tool_choice"] == {"type": "any", "disable_parallel_tool_use": True}
    assert second["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert {x["name"] for x in first["tools"]} == {"next_question", "answer", "explain", "progress", "refuse"}
    assert first["messages"] == [{"role": "user", "content": "State: open_question=no; answered=0\n\nnext question on d_cp_02"}]
    assert observation(second) == ["served: 1 question\nState: open_question=yes; answered=0"]
    sent = json.dumps(stub.calls, default=str)
    assert "Stem of" not in sent and "Module D text" not in sent and "Correct option" not in sent


def test_the_observation_after_answering_is_a_status_line_without_verdict_or_key():
    o, t, stub = make(tool("next_question", topic="d_cp_02"), done(), tool("answer", letters="A"), done())
    o.handle("next"); o.handle("A")
    assert observation(stub.calls[-1]) == ["answer recorded\nState: open_question=no; answered=1"]
    sent = json.dumps(stub.calls, default=str)
    assert "Not quite" not in sent and "the answer is" not in sent and "Correct." not in sent


def test_a_pair_that_is_only_half_answered_says_how_many_are_open():
    o, t, stub = make(tool("next_question", topic="d_cp_03"), done(), tool("answer", letters="B"), done())
    o.handle("next"); r = o.handle("B")
    assert observation(stub.calls[-1]) == ["answer recorded; 1 still open\nState: open_question=yes; answered=1"]
    assert "still need an answer" in r.text


def test_a_message_without_a_tool_call_or_with_an_unknown_tool_gets_the_help_text():
    for replies in ([text("hello")], [tool("delete_everything"), done()]):
        o, t, stub = make(*replies)
        r = o.handle("hi")
        assert r.intent == "help" and not t.state.history and o.stats["failures"] == 0


def test_an_empty_message_costs_no_call():
    o, t, stub = make()
    assert o.handle("   ").intent == "help" and stub.calls == []


def test_prose_written_by_the_model_never_reaches_the_student():
    o, t, stub = make(tool("next_question", topic="d_cp_02"), text("Sure! The answer to your question is B."))
    r = o.handle("next")
    assert "Sure!" not in r.text and "Stem of r1" in r.text


def test_only_the_first_tool_of_a_step_runs():
    two = NS(content=[NS(type="tool_use", id="a1", name="progress", input={}),
                      NS(type="tool_use", id="a2", name="next_question", input={"topic": "d_cp_02"})])
    o, t, stub = make(two, done())
    r = o.handle("progress and a question")
    assert r.intent == "progress" and not t.state.served
    assert observation(stub.calls[-1])[1].startswith("skipped: one tool per step")


# ------------------------------------------------------------------ several steps, and their limits
def test_two_requests_in_one_message_are_answered_in_order():
    o, t, stub = make(tool("next_question", topic="d_cp_02"), done(),
                      tool("answer", letters="B"), tool("next_question", topic="d_cp_02"), done())
    o.handle("next")
    r = o.handle("That was B, and now give me another one.")
    assert r.intent == "answer" and r.text.startswith("Correct.") and "Stem of r2" in r.text
    assert r.data["parts"] == ["answer", "next"] and r.data["steps"] == 2
    assert t.state.history[0]["correct"] and t.state.pending == ["r2"]


def test_the_step_limit_stops_the_loop_and_is_counted():
    o, t, stub = make(*[tool("progress") for _ in range(5)])
    r = o.handle("how am I doing")
    assert len(stub.calls) == 4 and len(stub.replies) == 1 and r.intent == "progress"
    assert o.stats["steps"] == 4 and o.stats["step_limit_hits"] == 1


def test_the_step_limit_can_be_set():
    o, t, stub = make(tool("progress"), tool("progress"), tool("progress"), max_steps=2)
    o.handle("x")
    assert len(stub.calls) == 2 and o.stats["step_limit_hits"] == 1


def test_a_repeated_next_question_in_one_turn_does_not_repeat_the_question():
    o, t, stub = make(tool("next_question", topic="d_cp_02"), tool("next_question", topic="d_cp_02"), done())
    r = o.handle("question please")
    assert r.text.count("Stem of r1") == 1 and "still have an open question" not in r.text
    assert observation(stub.calls[-1])[0].startswith("a question is already open")


def test_a_question_cannot_be_answered_in_the_turn_that_serves_it():
    o, t, stub = make(tool("next_question", topic="d_cp_02"), tool("answer", letters="B"), done())
    r = o.handle("give me a question, the answer is B")
    assert not t.state.history and t.state.pending == ["r1"] and "Answer the question after you have seen it." in r.text
    assert observation(stub.calls[-1])[0].startswith("refused: the question was only just served")


def test_a_failure_after_a_step_keeps_what_the_tools_have_written():
    o, t, stub = make(tool("next_question", topic="d_cp_02"), RuntimeError("down"))
    r = o.handle("next")
    assert r.intent == "next" and "Stem of r1" in r.text and o.stats["failures"] == 1


def test_when_the_first_call_fails_the_session_is_untouched():
    o, t, stub = make(RuntimeError("down"))
    r = o.handle("next question")
    assert r.intent == "model_unavailable" and r.text == O.ROUTE_FAILURE_MESSAGE
    assert t.state.pending_unit is None and not t.state.served and o.stats["failures"] == 1


def test_without_a_key_nothing_crashes_and_the_page_can_tell(monkeypatch):
    monkeypatch.setattr(O, "_load_env", lambda: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert llm_available() is False
    o = Orchestrator(Tutor(make_eb(), TAX, CURRICULUM))                      # building it needs no key
    assert o.handle("next question").intent == "model_unavailable"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder-not-a-key")
    assert llm_available() is True


def test_the_client_is_made_from_the_environment_with_the_vocareum_default(monkeypatch):
    seen = {}
    monkeypatch.setattr(O, "_load_env", lambda: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder-not-a-key")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    anthropic = pytest.importorskip("anthropic")
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: seen.update(kw) or "client")
    assert O.make_client() == "client"
    assert seen == {"api_key": "placeholder-not-a-key", "base_url": O.DEFAULT_BASE_URL}


# ------------------------------------------------------------------ the tools are the Tutor's, so its rules hold
def test_answer_grades_through_the_tutor_and_bad_letters_are_refused():
    o, t, stub = make(tool("next_question", topic="d_cp_02"), done(), tool("answer", letters=""), done(),
                      tool("answer", letters="E"), done(), tool("answer", letters="b"), done())
    o.handle("next")
    assert o.handle("hmm").intent == "help" and o.handle("E").intent == "help" and not t.state.history
    r = o.handle("b")
    assert r.intent == "answer" and r.text.startswith("Correct.") and t.state.history[0]["chosen"] == "B"


def test_a_pair_is_answered_in_order():
    o, t, stub = make(tool("next_question", topic="d_cp_03"), done(), tool("answer", letters="B C"), done())
    q = o.handle("control structures please")
    assert "Question 2 of 2" in q.text and "Reply with the letters in order" in q.text
    r = o.handle("B and C")
    assert [h["correct"] for h in t.state.history] == [True, True] and r.intent == "answer"


def test_answer_without_an_open_question_is_blocked_by_the_tutor():
    o, t, stub = make(tool("answer", letters="B"), done())
    r = o.handle("B")
    assert r.intent == "answer_blocked" and not t.state.history
    assert observation(stub.calls[-1])[0].startswith("no open question")


def test_an_unmatched_topic_never_changes_topic_silently():
    o, t, stub = make(tool("next_question", topic="quantum cooking"), done())
    r = o.handle("next question on quantum cooking")
    assert r.intent == "topic_unknown" and t.state.pending_unit is None and not t.state.served
    assert observation(stub.calls[-1])[0].startswith("topic not matched")


def test_progress_and_refusals_use_the_same_words_as_the_rule_router():
    o, t, stub = make(tool("next_question", topic="d_cp_02"), done(), tool("answer", letters="B"), done(),
                      tool("progress"), done(), tool("refuse", reason="out_of_scope"), done(),
                      tool("refuse", reason="unclear"), done())
    o.handle("q"); o.handle("B")
    rules = RuleRouter(Tutor(make_eb(), TAX, CURRICULUM))
    rules.handle("next question on d_cp_02"); rules.handle("B")
    assert o.handle("how am I doing").text == rules.handle("How am I doing?").text
    oos = o.handle("make up a question")
    assert oos.intent == "out_of_scope" and oos.text == rules.handle("Can you invent a new question for me?").text
    assert o.handle("???").intent == "help"


def test_reveal_requests_follow_the_same_rule_as_the_rule_router():
    o, t, stub = make(tool("refuse", reason="reveal"), done(), tool("next_question", topic="d_cp_02"), done(),
                      tool("refuse", reason="reveal"), done())
    assert o.handle("tell me the answer").intent == "reveal_declined"          # nothing attempted yet
    o.handle("next")
    r = o.handle("just tell me")
    assert r.intent == "reveal_declined" and "the answer is" not in r.text and not t.state.history


def test_a_reveal_request_after_an_attempt_with_nothing_open_becomes_an_explanation():
    o, t, stub = to_explain(tool("refuse", reason="reveal"), ctx("r1"), text("The correct answer is B because two is right."), done())
    r = o.handle("what was the answer?")
    assert r.intent == "explain" and "the answer is B" in r.text


def test_explain_before_any_attempt_is_blocked_by_code_even_with_a_question_open():
    o, t, stub = make(tool("explain"), done(), tool("next_question", topic="d_cp_02"), done(), tool("explain"), done())
    assert o.handle("why").intent == "explain_blocked"
    o.handle("next")
    r = o.handle("tell me the answer")                                       # misrouted to explain
    assert r.intent == "explain_blocked" and "the answer is" not in r.text.lower()
    assert observation(stub.calls[-1])[0].startswith("no attempt yet")
    assert o.stats["explainer_calls"] == 0 and worker_calls(stub) == []


# ------------------------------------------------------------------ the Explainer worker
def test_a_text_only_past_paper_gets_the_workers_explanation_with_verdict_and_key_added_by_code():
    o, t, stub = to_explain(tool("explain"), ctx("r1"),
                            text("The correct answer is B because two is the value asked for. Option A is not right."), done())
    r = o.handle("why?")
    assert r.intent == "explain" and r.tools == ["explain", "model_explanation"]
    assert r.text.startswith("Not quite - you chose A; the answer is B.")
    assert "**Explanation written by AI**" in r.text and "two is the value asked for" in r.text
    assert "Module D text" in r.text and O.NO_AI_LINE not in r.text
    w1, w2 = worker_calls(stub)
    assert w1["tools"] == [O.CONTEXT_TOOL] and w1["tool_choice"]["type"] == "any" and w2["tool_choice"]["type"] == "auto"
    assert "Explain item r1" in w1["messages"][0]["content"]
    got = observation(w2)[0]
    assert "Student's choice: A" in got and "Correct option: B" in got and "Stem of r1" in got
    assert o.stats["explainer_calls"] == 2 and o.stats["ai_explanations"] == 1 and r.data["ai"] is True


def test_the_orchestrating_model_never_sees_what_the_worker_wrote_or_was_given():
    o, t, stub = to_explain(tool("explain"), ctx("r1"), text("The correct answer is B because two is right."), done())
    o.handle("why?")
    last = routing_calls(stub)[-1]
    assert observation(last) == ["explanation shown\nState: open_question=no; answered=1"]
    sent = json.dumps(routing_calls(stub), default=str)
    assert "two is right" not in sent and "Correct option" not in sent and "Stem of" not in sent


def test_a_correct_answer_gets_the_verdict_and_key_too():
    o, t, stub = to_explain(tool("explain"), ctx("r1"), text("B is correct because two is the value asked for."), done(), wrong=False)
    assert o.handle("explain").text.startswith("Correct. The answer is B.")


def test_for_a_pair_the_shared_passage_goes_into_the_workers_context():
    o, t, stub = make(tool("next_question", topic="d_cp_03"), done(), tool("answer", letters="B C"), done(),
                      tool("explain"), ctx("c2"), text("The correct answer is C: the loop runs three times."), done())
    o.handle("next"); o.handle("B C")
    r = o.handle("why")
    got = observation(worker_calls(stub)[1])[0]
    assert "Answer Questions 1 and 2 with reference to the code." in got and "Stem of c2" in got and "Correct option: C" in got
    assert O.AI_LABEL in r.text


def test_an_admitted_item_gets_verdict_key_module_text_and_the_line_but_no_worker():
    o, t, stub = make(tool("next_question", topic="d_cp_02"), done(), tool("answer", letters="A"), done(), tool("explain"), done())
    serve_and_answer(t, ("tag", "d_cp_02")); serve_and_answer(t, ("tag", "d_cp_02"))     # both past-paper units used
    q = o.handle("next")
    assert PROVENANCE_LABEL in q.text
    o.handle("A")
    r = o.handle("why?")
    assert worker_calls(stub) == [] and o.stats["explainer_calls"] == 0
    assert r.text.startswith("Not quite - you chose A; the answer is C.")
    assert "Module D text" in r.text and O.NO_AI_LINE in r.text and O.AI_LABEL not in r.text and PROVENANCE_LABEL in r.text


def test_a_question_with_a_picture_gets_verdict_key_and_module_text_only():
    o, t, stub = make(tool("next_question", topic="a_ip_01"), done(), tool("answer", letters="A"), done(), tool("explain"), done())
    o.handle("next"); o.handle("A")
    r = o.handle("why?")
    assert worker_calls(stub) == [] and o.stats["explainer_calls"] == 0
    assert r.text.startswith("Not quite - you chose A; the answer is B.") and "Module A text" in r.text
    assert O.AI_LABEL not in r.text and O.NO_AI_LINE not in r.text


@pytest.mark.parametrize("change,expected", [
    ({}, True),
    ({"qn_image_filename": "question/2012/q1.png"}, False),
    ({"ans_image_filename": "question/2012/o1.png"}, False),
    ({"answer_options": "[IMAGE]"}, False),
    ({"question_text": "Look at ![fig](question/2012/f.png) and answer"}, False),
    ({"cluster_desc": "Passage ![fig](question/2012/f.png)"}, False),
    ({"source": "generated", "provisional": True}, False),
])
def test_who_may_get_a_model_explanation(change, expected):
    eb = make_eb()
    for col, v in change.items():
        eb[col] = eb[col].astype(object)
        eb.loc[eb.item_id == "r1", col] = v
    t = Tutor(eb, TAX, CURRICULUM)
    assert Orchestrator.ai_eligible(t.by_id.loc["r1"]) is expected


def test_the_workers_tool_is_read_only_for_its_own_item_and_only_after_an_attempt():
    o, t, stub = make()
    w = Explainer(o)
    with pytest.raises(ValueError):
        w.context("r1", "r1")                                                # no attempt yet
    serve_and_answer(t, ("tag", "d_cp_02"))
    assert "Correct option: B" in w.context("r1", "r1")
    with pytest.raises(ValueError):
        w.context("r2", "r1")                                                # not the item it was given
    for _ in range(2):
        serve_and_answer(t, ("tag", "d_cp_02"))                              # r2, then the first admitted item
    with pytest.raises(ValueError):
        w.context("g1", "g1")                                                # an admitted item gets no model text


def test_a_worker_asking_for_another_item_gets_an_error_and_the_text_is_not_used():
    o, t, stub = to_explain(tool("explain"), ctx("r2"), text("The correct answer is B."), done())
    r = o.handle("why?")
    assert observation(worker_calls(stub)[1])[0] == "this worker may only explain the item it was given"
    assert worker_calls(stub)[1]["messages"][-1]["content"][0]["is_error"] is True
    assert r.text.startswith(O.FALLBACK_MESSAGE) and r.data["failed"] is True


# ------------------------------------------------------------------ the consistency check and the fallback
@pytest.mark.parametrize("bad", [
    "The correct answer is C because it is the data type.",
    "Option C is correct, since three is the value.",
    "The right option is C.",
    "The answer is C.",
    "C is the right choice here.",
    "The best answer would be D: four is the value.",
])
def test_an_explanation_that_names_another_letter_is_rejected_and_replaced_by_the_fallback(bad):
    o, t, stub = to_explain(tool("explain"), ctx("r1"), text(bad), done())
    r = o.handle("why?")
    assert r.text.startswith(O.FALLBACK_MESSAGE) and "Not quite - you chose A; the answer is B." in r.text
    assert "Module D text" in r.text and bad not in r.text and O.AI_LABEL not in r.text
    assert o.stats["rejections"] == 1 and r.data["rejected"] is True and r.data["ai"] is False
    assert observation(routing_calls(stub)[-1])[0].startswith("fallback shown")
    assert len(worker_calls(stub)) == 2                                      # no retry


@pytest.mark.parametrize("bad", [
    "The correct answer is **C** because it is the data type.",
    "The correct answer is _C_.",
    "The correct answer is `C`.",
    'The correct answer is "C".',
    "The right option is <b>C</b>.",
    "**C** is the correct option.",
    "Correct answer: **D**",
])
def test_a_wrong_letter_in_bold_italics_quotes_or_code_is_still_rejected(bad):
    o, t, stub = to_explain(tool("explain"), ctx("r1"), text(bad), done())
    r = o.handle("why?")
    assert r.text.startswith(O.FALLBACK_MESSAGE) and o.stats["rejections"] == 1 and O.AI_LABEL not in r.text


@pytest.mark.parametrize("good", [
    "The correct answer is **B**. Option C is not correct because three is too large.",
    "**B** is correct; A and D are distractors.",
])
def test_the_key_in_bold_is_accepted(good):
    o, t, stub = to_explain(tool("explain"), ctx("r1"), text(good), done())
    r = o.handle("why?")
    assert good in r.text and o.stats["rejections"] == 0 and o.stats["ai_explanations"] == 1


def test_claimed_letters_sees_through_styling_and_does_not_mistake_the_article():
    assert O.claimed_letters("The correct answer is **C**.") == {"C"}
    assert O.claimed_letters("A computer is a machine, and a good one.") == set()
    assert O.claimed_letters("Option **A** is not correct, so the answer is **B**.") == {"B"}


@pytest.mark.parametrize("good", [
    "The correct answer is B. Option C is not correct because three is too large.",
    "B is correct; A and D are distractors.",
    "A computer stores a whole number as an integer, so the right option is B.",
])
def test_an_explanation_that_names_the_key_is_accepted(good):
    o, t, stub = to_explain(tool("explain"), ctx("r1"), text(good), done())
    r = o.handle("why?")
    assert good in r.text and o.stats["rejections"] == 0 and o.stats["ai_explanations"] == 1


def test_an_empty_explanation_is_rejected_too():
    o, t, stub = to_explain(tool("explain"), ctx("r1"), text("   "), done())
    r = o.handle("why?")
    assert r.text.startswith(O.FALLBACK_MESSAGE) and o.stats["rejections"] == 1


def test_when_the_workers_call_fails_the_fixed_message_and_the_plain_feedback_are_shown():
    o, t, stub = to_explain(tool("explain"), TimeoutError("slow"), done())
    r = o.handle("why?")
    assert r.text == (f"{O.FALLBACK_MESSAGE}\n\nNot quite - you chose A; the answer is B.\n\n"
                      "Curriculum, Module D:\nModule D text")
    assert o.stats["failures"] == 1 and r.data["failed"] is True and t.state.history[-1]["chosen"] == "A"


def test_a_text_written_without_fetching_the_context_is_not_used():
    o, t, stub = to_explain(tool("explain"), text("The correct answer is B."), done())
    r = o.handle("why?")
    assert r.text.startswith(O.FALLBACK_MESSAGE) and o.stats["failures"] == 1


def test_the_worker_stops_at_its_step_limit():
    o, t, stub = to_explain(tool("explain"), ctx("r1"), ctx("r1"), ctx("r1"), done())
    r = o.handle("why?")
    assert len(worker_calls(stub)) == O.EXPLAINER_MAX_STEPS == 3
    assert r.text.startswith(O.FALLBACK_MESSAGE) and o.stats["failures"] == 1


# ------------------------------------------------------------------ the two routers behind one interface
def test_a_scripted_dialogue_gives_the_same_replies_on_both_routers():
    msgs = ["next question on d_cp_02", "A", "How am I doing?"]
    o, t, stub = make(tool("next_question", topic="d_cp_02"), done(), tool("answer", letters="A"), done(),
                      tool("progress"), done())
    rules = RuleRouter(Tutor(make_eb(), TAX, CURRICULUM))
    for m in msgs:
        a, b = o.handle(m), rules.handle(m)
        assert (a.intent, a.text) == (b.intent, b.text)
