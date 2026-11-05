"""Tests for src/gate.py (Section 3.5): checks 1 to 6, no model."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src import data as D  # noqa: E402
from src import gate as G  # noqa: E402
from src import store as S  # noqa: E402

tax = D.load_taxonomy(ROOT / "data/taxonomy.csv")
df = D.derive(D.load_bank(ROOT / "data/final_dataset.csv", tax))
corpus = G.RealCorpus.from_bank(df)

STEM = "A school stores the examination marks of every student in a table with one row for each student. Which operation would a teacher use to arrange the rows from the highest mark to the lowest mark?"
OPTS = {"A": "Sorting the table by the mark column in descending order", "B": "Merging the table with another table of names",
        "C": "Encrypting every row with a password", "D": "Compressing the table into a single archive file"}


def run(stem=STEM, options=None, key="A", **kw):
    return G.evaluate(stem, options if options is not None else dict(OPTS), key, corpus=corpus, **kw)


def failed(res, name):
    return res["checks"][name]["level"]


def test_a_clean_candidate_passes():
    r = run()
    assert r["verdict"] == "pass" and r["reasons"] == [] and r["not_run"] == []
    assert set(r) == {"checks", "verdict", "reasons", "not_run"}          # no model-supplied fields
    assert G.evaluate(STEM, OPTS, "A")["not_run"] == ["4_copy_ratio"]       # no corpus: the copy check is listed as not run


@pytest.mark.parametrize("phrase", ["at the end of the loop", "End of file marker", "the end of a program", "weekend of"])
def test_ordinary_end_of_wording_is_not_a_banner(phrase):
    r = run(STEM + f" Consider the value {phrase}.")
    assert failed(r, "1_format_length") is None and r["verdict"] == "pass"


@pytest.mark.parametrize("where", ["stem", "option"])
@pytest.mark.parametrize("banner", ["**END OF PAPER**", "END OF SECTION A"])
def test_paper_end_banner_is_rejected(where, banner):
    o = dict(OPTS)
    s = STEM
    if where == "stem":
        s = STEM + " " + banner
    else:
        o["D"] = o["D"] + "  " + banner
    r = run(s, o)
    assert r["verdict"] == "reject" and failed(r, "1_format_length") == "reject"
    assert "banner" in r["checks"]["1_format_length"]["value"]


def test_format_failures_reject():
    assert run(options={k: v for k, v in OPTS.items() if k != "D"})["verdict"] == "reject"
    assert run(options={**OPTS, "C": "  "})["verdict"] == "reject"
    assert run("", dict(OPTS))["verdict"] == "reject"
    assert run("Too short?", {"A": "a", "B": "b", "C": "c", "D": "d"})["verdict"] == "reject"
    assert run(STEM + " " + "extra words " * 120)["verdict"] == "reject"
    r = G.evaluate(STEM, OPTS, "A", valid_format=False, parse_error="no options found")
    assert r["verdict"] == "reject" and "no options found" in r["checks"]["1_format_length"]["value"]


def test_repeated_phrase_loop_is_rejected():
    r = run(STEM + " the data the data the data the data the data the data the data the data the data")
    assert r["verdict"] == "reject" and "repeated phrase" in r["checks"]["1_format_length"]["value"]


def test_figure_placeholders_and_dangling_references():
    assert run("[Figure: a network diagram] " + STEM)["verdict"] == "reject"
    assert run("![](img/a.png) " + STEM)["verdict"] == "reject"
    dangling = "According to the diagram, which device connects the computers in the office to the internet through one shared cable?"
    assert failed(run(dangling), "2_figure_reference") == "reject"
    assert failed(run(dangling + "\n```\nA -- B\n```"), "2_figure_reference") is None      # content is supplied
    assert failed(run("Which of the following statements about a computer network with many users is correct? It uses shared media."),
                  "2_figure_reference") is None                                          # standard exam wording is fine


def test_duplicate_and_overlapping_options():
    assert run(options={**OPTS, "B": OPTS["C"].upper() + " "})["verdict"] == "reject"          # identical after trimming and case
    contained = run(options={**OPTS, "B": "Sorting the table by the mark column"})
    assert contained["verdict"] == "flag" and failed(contained, "3_similar_options") == "flag"
    near = run(options={**OPTS, "B": OPTS["A"].replace("descending", "ascending") + "."})
    assert near["checks"]["3_similar_options"]["value"]["max_similarity"] > G.DUP_SIM_FLAG and near["verdict"] == "flag"


def test_copy_ratio_levels():
    base = [w for w in "the quick brown fox jumps over the lazy dog while seven small birds sing a song in the green park today".split()]
    c = G.RealCorpus([" ".join(base)])
    text = " ".join(base)
    assert c.copy_ratio(text) == 1.0
    assert G.check_copy(text, c)["level"] == "reject"
    n = len(G.grams(text, 6))
    half = " ".join(base[:10]) + " zebra " + " ".join("novel%d" % i for i in range(12))       # a few shared sequences only
    ratio = c.copy_ratio(half)
    assert 0 < ratio < G.COPY_FLAG and G.check_copy(half, c)["passed"] is True
    assert G.check_copy(text, None)["passed"] is None and n > 0
    # thresholds of the design
    assert (G.COPY_FLAG, G.COPY_REJECT) == (0.4, 0.5)


def test_verbatim_real_question_is_rejected_by_the_copy_check():
    real = df[df.text_only & ~df.withdrawn & (df.cluster != 1)].iloc[40]
    prep = D.prepare_bank(df[df.qn_id == real.qn_id]).iloc[0]
    opts = D.parse_options(prep["answer_options"])
    r = G.evaluate(prep["question_text"], opts, real["correct_answer"], corpus=corpus)
    assert r["checks"]["4_copy_ratio"]["value"] >= 0.9 and r["verdict"] == "reject"


def test_key_length_and_stem_overlap_are_soft():
    long_key = run(options={**OPTS, "A": OPTS["A"] + " and then storing the sorted result in a new table for later use"})
    assert failed(long_key, "5_key_length") == "flag" and long_key["verdict"] == "flag"
    stem = "A teacher wants a table sorted in descending order by the mark column for every student. Which operation does this?"
    r = run(stem, {**OPTS, "A": "Sorting by the mark column in descending order"})
    assert failed(r, "6_stem_option_overlap") == "flag"
    both = run(stem, {**OPTS, "A": "Sorting by the mark column in descending order", "B": "Sorting by the mark column in descending order of age"})
    assert failed(both, "6_stem_option_overlap") is None          # more than one option repeats the stem


def test_missing_key_is_flagged_not_trusted():
    r = G.evaluate(STEM, OPTS, None, corpus=corpus)
    assert r["verdict"] == "flag" and "no key from the generator" in r["reasons"]
    assert r["checks"]["5_key_length"]["passed"] is None


def test_a_hard_failure_outweighs_a_soft_one():
    r = run("[Figure: x] " + STEM, {**OPTS, "B": "Sorting the table by the mark column"})
    assert r["verdict"] == "reject"


def test_a_shared_cluster_description_is_part_of_the_copy_corpus():
    desc = df[(df.cluster == 1) & df.cluster_desc.notna()].cluster_desc.iloc[0]
    assert corpus.copy_ratio(G.normalise(desc)) >= 0.9
    assert G.RealCorpus.from_bank(df.drop(columns=["cluster_desc"])).copy_ratio(G.normalise(desc)) < 0.9


def test_real_questions_are_rarely_rejected_without_the_copy_check():
    """Section 1.8 calibration: the same rules on the 400 real reference items."""
    ref = df[df.text_only & ~df.withdrawn & (df.cluster != 1) & (df.year <= 2024)]
    prep = D.prepare_bank(ref)
    res = []
    for _, r in prep.iterrows():
        o = D.parse_options(r["answer_options"])
        if len(o) == 4:
            res.append(G.evaluate(r["question_text"], o, str(r["correct_answer"]).strip(), corpus=None)["verdict"])
    s = pd.Series(res).value_counts(normalize=True)
    assert len(res) == 400
    assert s.get("reject", 0) <= 0.03          # the 1st/99th percentile length band rejects 2% by construction
    assert s.get("flag", 0) <= 0.13


def test_gate_result_is_stored_and_needs_a_candidate(tmp_path):
    con = S.init_db(tmp_path / "s.sqlite", tax)
    r = run()
    with pytest.raises(sqlite3_error()):
        G.write_gate_result(con, "GEN_x_1", r)                      # no candidate row yet: foreign key
    with con:
        con.execute("INSERT INTO candidate(cand_id,backend,target_module,stem,valid_format) VALUES ('GEN_x_1','x','A',?,1)", (STEM,))
    G.write_gate_result(con, "GEN_x_1", r)
    row = con.execute("SELECT verdict, solver_answer, solver_agrees_with_key FROM gate_result").fetchone()
    assert row == ("pass", None, None)                              # the solver columns are reserved and stay empty
    with pytest.raises(sqlite3_error()):
        G.write_gate_result(con, "GEN_x_1", r)                      # one result per candidate


def sqlite3_error():
    import sqlite3
    return sqlite3.IntegrityError
