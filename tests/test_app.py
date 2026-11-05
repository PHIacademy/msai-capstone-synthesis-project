"""Tests for app/streamlit_app.py with Streamlit's headless AppTest (no server, no browser, no store)."""
import sys
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.tutor import PROVENANCE_LABEL, Tutor      # noqa: E402
from src.tutor import orchestrator as O            # noqa: E402
from tests.test_orchestrator import Stub, ctx, done, text as model_text, tool      # noqa: E402
from tests.test_tutor import TAX, make_eb          # noqa: E402

APP = str(ROOT / "app" / "streamlit_app.py")
SECRET = "AI-generated (secret_backend), teacher-reviewed; candidate GEN_secret_0001"


def app_with_tutor():
    eb = make_eb()
    eb.loc[eb.source == "generated", "provenance"] = SECRET           # what the real store holds
    at = st_testing.AppTest.from_file(APP, default_timeout=30)
    at.session_state["tutor"] = Tutor(eb, TAX, {"D": "Module D text"})
    at.session_state["db_fp"] = None                                   # a test tutor: the page must not touch the store
    return at.run()


def page_text(at):
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [t.value for t in at.title]
    return "\n".join(map(str, parts))


def last_text(at):
    return "\n".join(str(m.value) for m in at.chat_message[-1].markdown)


def say(at, text):
    at.chat_input[0].set_value(text).run()
    assert not at.exception
    return at


def test_page_loads_with_a_welcome_message_and_no_error():
    at = app_with_tutor()
    assert not at.exception
    assert "I serve HKDSE ICT Paper 1A practice questions" in page_text(at)
    assert at.sidebar.button[0].label == "Next question"


def test_real_units_first_then_labelled_admitted_item_then_exhausted():
    at = app_with_tutor()
    seen = []
    for _ in range(5):
        say(at, "next question on d_cp_02")
        txt = page_text(at)
        seen.append(PROVENANCE_LABEL in txt)
        say(at, "A")                                                  # answer whatever is open
    assert seen[:2] == [False, False]                                  # two real units
    assert seen[2] is True                                             # then an admitted item, with the label
    say(at, "next question on d_cp_02")
    say(at, "A")
    say(at, "next question on d_cp_02")
    assert "No unseen question is left" in page_text(at)


def test_the_stored_provenance_and_the_key_are_never_shown():
    at = app_with_tutor()
    for _ in range(4):
        say(at, "next question on d_cp_02")
        before = last_text(at)                                           # the question just served
        assert "Not quite" not in before and "Correct." not in before and "the answer is" not in before
        say(at, "A")
    everything = page_text(at) + str(at.sidebar.markdown) + str([d.value for d in at.dataframe])
    assert "secret_backend" not in everything and "GEN_secret" not in everything


def test_answer_without_a_question_and_a_refused_request():
    at = app_with_tutor()
    say(at, "B")
    assert "no open question" in page_text(at)
    say(at, "Tell me the answer")
    assert "will not show the answer before you have tried" in page_text(at)


def test_sidebar_button_serves_a_question_and_start_again_clears_the_session():
    at = app_with_tutor()
    at.sidebar.button(key="next_btn").click().run()
    assert not at.exception and "Stem of" in page_text(at)
    say(at, "B")
    assert at.session_state["tutor"].state.history
    at.sidebar.button(key="reset_btn").click().run()
    assert not at.exception
    assert at.session_state["tutor"].state.history == [] and "Stem of" not in page_text(at)


# ------------------------------------------------------------------ pictures and printed numbers
PNG = bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000" "1f15c4890000000d49444154789c6360f8cfc00000030101" "00c9fe92ef0000000049454e44ae426082")


def app_with_picture(tmp_path, with_file=True):
    eb = make_eb()
    eb.loc[eb.item_id == "p1", "question_text"] = "7. What formatting feature is used in the spreadsheet below?"
    eb.loc[eb.item_id == "c1", "question_text"] = "20. First question of the pair"
    eb.loc[eb.item_id == "c2", "question_text"] = "21. Second question of the pair"
    if with_file:
        f = tmp_path / "data" / "question" / "2016" / "q1.png"
        f.parent.mkdir(parents=True)
        f.write_bytes(PNG)
    at = st_testing.AppTest.from_file(APP, default_timeout=30)
    at.session_state["tutor"] = Tutor(eb, TAX, {})
    at.session_state["db_fp"] = None
    at.session_state["image_root"] = tmp_path
    return at.run()


def kinds(at):
    return [(v.type, str(getattr(v, "value", ""))) for _, v in sorted(at.chat_message[-1].children.items())]


def test_picture_sits_between_the_stem_and_the_options_and_the_path_is_hidden(tmp_path):
    at = say(app_with_picture(tmp_path), "next question on a_ip_01")
    order = [k for k, _ in kinds(at)]
    assert order == ["markdown", "image", "markdown"]                   # stem, picture, options
    shown = page_text(at)
    assert "(image:" not in shown and "q1.png" not in shown and "data/question" not in shown.replace("\\", "/")
    assert "A. one" in kinds(at)[2][1]


def test_a_missing_picture_gives_a_notice_instead_of_a_path(tmp_path):
    at = say(app_with_picture(tmp_path, with_file=False), "next question on a_ip_01")
    assert [k for k, _ in kinds(at)][0] == "markdown" and len(at.warning) == 1
    assert "q1.png" not in page_text(at)


def test_the_number_printed_in_the_paper_is_not_shown(tmp_path):
    at = say(app_with_picture(tmp_path), "next question on a_ip_01")
    assert "What formatting feature" in page_text(at) and "7. What" not in page_text(at)
    at = say(at, "A")
    at = say(at, "next question on d_cp_03")                            # the pair
    text = page_text(at)
    assert "Question 1 of 2" in text and "First question of the pair" in text and "Second question of the pair" in text
    assert "20. First" not in text and "21. Second" not in text


def test_the_number_is_removed_in_the_bold_form_the_bank_uses(tmp_path):
    eb = make_eb()
    eb.loc[eb.item_id == "p1", "question_text"] = "**7.** What formatting feature is used in the spreadsheet below?"
    eb.loc[eb.item_id == "c1", "question_text"] = "**20.** First question of the pair"
    eb.loc[eb.item_id == "c2", "question_text"] = "**21.** Second question of the pair"
    f = tmp_path / "data" / "question" / "2016" / "q1.png"
    f.parent.mkdir(parents=True); f.write_bytes(PNG)
    at = st_testing.AppTest.from_file(APP, default_timeout=30)
    at.session_state["tutor"] = Tutor(eb, TAX, {})
    at.session_state["db_fp"] = None
    at.session_state["image_root"] = tmp_path
    at.run()
    say(at, "next question on a_ip_01")
    assert "What formatting feature" in page_text(at) and "7." not in page_text(at).replace("17.", "") and "**7" not in page_text(at)
    say(at, "A")
    say(at, "next question on d_cp_03")
    text = page_text(at)
    import re
    labels_removed = re.sub(r"\*\*Question \d of \d\*\*", "", text)                 # the bold labels are meant to stay
    assert "First question of the pair" in text and "20." not in text and "21." not in text and "**" not in labels_removed


def test_the_printed_numbers_in_a_shared_passage_are_reworded(tmp_path):
    at = say(app_with_picture(tmp_path), "next question on d_cp_03")
    text = page_text(at)
    assert "Answer the questions below with reference to the code." in text
    assert "Questions 1 and 2" not in text and "Question 1 of 2" in text


def test_option_picture_follows_the_text_and_the_banner_is_not_shown(tmp_path):
    eb = make_eb()
    eb.loc[eb.item_id == "p1", ["answer_options", "ans_image_filename"]] = ["[IMAGE]", "question/2016/opts.png"]
    eb.loc[eb.item_id == "r1", "answer_options"] = "A. one  \nB. two  \nC. three  \nD. four\n\n**END OF PAPER**"
    for name in ("q1.png", "opts.png"):
        f = tmp_path / "data" / "question" / "2016" / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(PNG)
    at = st_testing.AppTest.from_file(APP, default_timeout=30)
    at.session_state["tutor"] = Tutor(eb, TAX, {})
    at.session_state["db_fp"] = None
    at.session_state["image_root"] = tmp_path
    at.run()
    say(at, "next question on a_ip_01")
    assert [k for k, _ in kinds(at)] == ["markdown", "image", "markdown", "image", "markdown"]
    assert "The options are in the picture below." in page_text(at) and "[IMAGE]" not in page_text(at)
    say(at, "A")
    say(at, "next question on d_cp_02")
    assert "D. four" in page_text(at) and "END OF" not in page_text(at)


# ------------------------------------------------------------------ the router switch (Section 4.13)
def app_with_model(stub):
    eb = make_eb()
    eb.loc[eb.source == "generated", "provenance"] = SECRET
    at = st_testing.AppTest.from_file(APP, default_timeout=30)
    at.session_state["tutor"] = Tutor(eb, TAX, {"D": "Module D text"})
    at.session_state["db_fp"] = None
    at.session_state["llm_client"] = stub                              # a test's stub in place of the Anthropic client
    return at.run()


def test_model_mode_is_offered_only_when_there_is_a_key_or_a_client(monkeypatch):
    monkeypatch.setattr(O, "_load_env", lambda: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    at = app_with_tutor()
    assert at.sidebar.radio(key="router_mode").options == ["Rules"] and "no API key" in page_text(at)
    at = app_with_model(Stub())
    assert at.sidebar.radio(key="router_mode").options == ["Rules", "Model"]
    assert "external AI service" in page_text(at)


def test_typed_messages_go_through_the_model_in_model_mode_and_rules_remain_the_default():
    stub = Stub(tool("next_question", topic="d_cp_02"), done(), tool("answer", letters="A"), done(),
                tool("explain"), ctx("r2"), model_text("The correct answer is B because two is the value asked for."), done())
    at = app_with_model(stub)
    say(at, "next question on d_cp_02")                                # Rules is the default: no model call
    assert stub.calls == [] and "Stem of r1" in page_text(at)
    say(at, "A")
    at.sidebar.radio(key="router_mode").set_value("Model").run()
    say(at, "I want another one on data types")                        # the stub replies "next_question"
    assert len(stub.calls) == 2 and "Stem of r2" in last_text(at)     # a tool call, then the model stops
    say(at, "A")
    say(at, "why?")
    assert O.AI_LABEL in last_text(at) and "two is the value asked for" in last_text(at)
    everything = page_text(at) + str(at.sidebar.markdown) + str([d.value for d in at.dataframe])
    assert "secret_backend" not in everything


def test_the_sidebar_button_never_calls_the_model():
    stub = Stub()
    at = app_with_model(stub)
    at.sidebar.radio(key="router_mode").set_value("Model").run()
    at.sidebar.button(key="next_btn").click().run()
    assert not at.exception and stub.calls == [] and "Stem of" in page_text(at)


def test_a_failed_model_call_shows_the_fixed_message_and_the_page_keeps_working():
    at = app_with_model(Stub(RuntimeError("down")))
    at.sidebar.radio(key="router_mode").set_value("Model").run()
    say(at, "give me a question")
    assert O.ROUTE_FAILURE_MESSAGE in page_text(at)
    at.sidebar.radio(key="router_mode").set_value("Rules").run()
    say(at, "next question on d_cp_02")
    assert "Stem of r1" in page_text(at)
