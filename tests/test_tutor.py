"""Tests for src/tutor (no model, no database: a small synthetic effective bank)."""
import pandas as pd
import pytest

from src.tutor import PROVENANCE_LABEL, RuleRouter, Tutor, load_curriculum

COLS = ["item_id", "source", "qn_id", "year", "question_text", "answer_options", "correct_answer", "cohort_pct",
        "cluster", "cluster_desc", "qn_image_filename", "qn_image_type", "ans_image_filename", "tags",
        "primary_module", "label_level", "label_source", "unit", "provisional", "provenance"]
OPTS = "A. one  \nB. two  \nC. three  \nD. four"
TAX = pd.DataFrame({"qn_type_id": ["d_cp_02", "d_cp_03", "e_si_01", "a_ip_01"], "module": list("DDEA"),
                    "module_name": ["x"] * 4, "subtopic_name": ["Data Type", "Control Structures", "Technological Innovations", "Introduction to Information Processing"]})


def row(item, tags, key="B", year=2015, src="hkeaa", prov=False, unit=None, desc=None, level="official", module=None, img=None):
    mod = module if module is not None else (tags[0][0].upper() if tags else "")
    return {"item_id": item, "source": src, "qn_id": None if src == "generated" else item, "year": year,
            "question_text": f"Stem of {item}", "answer_options": OPTS, "correct_answer": key, "cohort_pct": 50.0,
            "cluster": 1 if desc else 0, "cluster_desc": desc, "qn_image_filename": img, "qn_image_type": None,
            "ans_image_filename": None, "tags": tags, "primary_module": mod, "label_level": level,
            "label_source": "official", "unit": unit or item, "provisional": prov, "provenance": "x"}


def make_eb():
    rows = [row("r1", ["d_cp_02"], year=2012), row("r2", ["d_cp_02"], year=2013),            # d_cp_02: two real units
            row("g1", ["d_cp_02"], src="generated", prov=True, year=pd.NA, key="C"),        # and two admitted
            row("g2", ["d_cp_02"], src="generated", prov=True, year=pd.NA, key="D"),
            row("c1", ["d_cp_03"], year=2014, unit="cluster::shared", desc="Answer Questions 1 and 2 with reference to the code."),
            row("c2", ["d_cp_03"], year=2014, key="C", unit="cluster::shared", desc="Answer Questions 1 and 2 with reference to the code."),
            row("m1", [], src="generated", prov=True, year=pd.NA, level="module", module="E"),   # module-only admitted item
            row("p1", ["a_ip_01"], year=2016, img="question/2016/q1.png")]
    return pd.DataFrame(rows)[COLS]


@pytest.fixture
def tutor():
    return Tutor(make_eb(), TAX, {"D": "Module D text", "A": "Module A text"})


def serve_and_answer(t, topic, letter="A"):
    r = t.next_question(topic)
    for i in list(t.state.pending):
        t.answer(i, letter)
    return r


def test_real_units_come_first_then_provisional_with_notice(tutor):
    first = tutor.next_question(("tag", "d_cp_02"))
    assert first["status"] == "served" and not first["provisional"] and first["notice"] is None
    for i in list(tutor.state.pending):
        tutor.answer(i, "A")
    second = serve_and_answer(tutor, ("tag", "d_cp_02"))
    assert not second["provisional"]
    third = tutor.next_question(("tag", "d_cp_02"))
    assert third["provisional"] and PROVENANCE_LABEL in third["notice"] and PROVENANCE_LABEL in third["text"]


def test_exhausted_pool_is_explicit_and_never_switches_topic(tutor):
    r = tutor.next_question(("tag", "e_si_01"))
    assert r["status"] == "exhausted" and r["alternatives"] == [] and tutor.state.pending_unit is None
    for _ in range(4):
        serve_and_answer(tutor, ("tag", "d_cp_02"))
    r = tutor.next_question(("tag", "d_cp_02"))
    assert r["status"] == "exhausted"
    assert [a["tag"] for a in r["alternatives"]] == ["d_cp_03"]


def test_key_never_in_the_rendered_unit_and_cluster_served_together(tutor):
    r = tutor.next_question(("tag", "d_cp_03"))
    assert r["item_ids"] == ["c1", "c2"]
    assert "shared" not in r["text"] and "Answer Questions 1 and 2" in r["text"] and "Question 2 of 2" in r["text"]
    assert "correct" not in r["text"].lower()


def test_single_scoring_and_pending_gate(tutor):
    tutor.next_question(("tag", "d_cp_02"))
    with pytest.raises(ValueError):
        tutor.answer("r2", "A")                         # not the pending item
    rec = tutor.answer("r1", "b")
    assert rec["correct"] and rec["correct_answer"] == "B"
    with pytest.raises(ValueError):
        tutor.answer("r1", "B")                         # already scored
    with pytest.raises(ValueError):
        tutor.explain("r2")                             # no attempt yet


def test_next_while_pending_returns_same_unit(tutor):
    a = tutor.next_question(("tag", "d_cp_02"))
    b = tutor.next_question(("tag", "d_cp_03"))
    assert b["status"] == "pending_open" and b["unit"] == a["unit"]


def test_module_only_label_credits_module_only(tutor):
    tutor.next_question(("module", "E"))                # only the module-only admitted item exists for E
    rec = tutor.answer(tutor.state.pending[0], "A")
    assert rec["tags"] == [] and rec["module"] == "E"
    assert tutor.state.mastery_tag == {} and tutor.state.mastery_module["E"]["attempts"] == 1


def test_default_target_goes_to_weakest_attempted_tag(tutor):
    serve_and_answer(tutor, ("tag", "a_ip_01"), "B")    # correct (key B)
    serve_and_answer(tutor, ("tag", "d_cp_02"), "A")    # wrong
    r = tutor.next_question(None)
    assert "d_cp_02" in tutor.units[r["unit"]]["tags"]


def test_images_are_surfaced_as_paths(tutor):
    r = tutor.next_question(("tag", "a_ip_01"))
    assert "data/question/2016/q1.png" in r["text"].replace("\\", "/")


def test_constructor_refuses_leaks_and_non_provisional_generated():
    eb = make_eb(); eb["gen_key"] = "A"
    with pytest.raises(ValueError):
        Tutor(eb, TAX)
    eb = make_eb(); eb.loc[eb.source == "generated", "provisional"] = False
    with pytest.raises(ValueError):
        Tutor(eb, TAX)


def test_explain_uses_module_text_and_marks_admitted(tutor):
    for _ in range(3):
        serve_and_answer(tutor, ("tag", "d_cp_02"))
    e = tutor.explain(tutor.state.history[-1]["item_id"])
    assert e["provisional"] and "admitted item" in e["note"] and e["references"][0]["text"] == "Module D text"


def test_router_scenarios(tutor):
    r = RuleRouter(tutor)
    assert r.handle("Can you invent a new question for me?").intent == "out_of_scope"
    assert r.handle("B").intent == "answer_blocked"
    assert r.handle("Tell me the answer").intent == "reveal_declined"
    q = r.handle("give me a question on d_cp_02")
    assert q.intent == "next" and tutor.state.pending
    assert r.handle("just tell me the answer without trying").intent == "reveal_declined"
    a = r.handle("my answer is B")
    assert a.intent == "answer" and "Correct." in a.text and not tutor.state.pending
    assert r.handle("why?").intent == "explain"
    assert r.handle("How am I doing?").intent == "progress"
    assert r.handle("next question on technological innovations").intent == "exhausted"
    c = r.handle("next question on control structures")
    assert "Reply with the letters in order" in c.text
    p = r.handle("B")
    assert p.intent == "answer" and "still need an answer" in p.text


def test_curriculum_loader(tmp_path):
    md = "intro\n"
    for L, n in zip("ABCDE", ["Information Processing", "Computer System Fundamentals", "Internet and its Applications",
                              "Computational Thinking and Programming", "Social Implications"]):
        md += f"\n{L}. {n}\n\nIntroduction\n\nText {L}\n\nDetails\n\nignored\n"
    p = tmp_path / "c.md"; p.write_text(md, encoding="utf-8")
    cur = load_curriculum(p)
    assert sorted(cur) == list("ABCDE") and "Text D" in cur["D"] and "ignored" not in cur["D"]
    assert load_curriculum(tmp_path / "missing.md") == {}


def test_paper_end_banner_is_not_shown():
    eb = make_eb()
    eb.loc[eb.item_id == "p1", "answer_options"] = OPTS + "\n\n**END OF PAPER**"
    eb.loc[eb.item_id == "r1", "answer_options"] = OPTS + "\n\n**END OF SECTION A**"
    t = Tutor(eb, TAX, {})
    for unit in ("p1", "r1"):
        txt = t.render(unit)
        assert "END OF" not in txt and "D. four" in txt


def test_option_pictures_are_shown():
    eb = make_eb()
    eb.loc[eb.item_id == "r1", ["answer_options", "ans_image_filename"]] = ["[IMAGE]", "question/2015/q18_options.png"]
    eb.loc[eb.item_id == "r2", "ans_image_filename"] = "question/2015/q25_options.png"
    t = Tutor(eb, TAX, {})
    a = t.render("r1").replace("\\", "/")
    assert "[IMAGE]" not in a and "The options are in the picture below." in a and a.count("q18_options.png") == 1
    b = t.render("r2").replace("\\", "/")
    assert "D. four" in b and b.index("D. four") < b.index("q25_options.png")
