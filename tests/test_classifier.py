"""Tests for src/classifier.py against the real bank (small grid, so they run in seconds).

Run:  python3 -m pytest tests/test_classifier.py
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src import classifier as C  # noqa: E402
from src import data as D  # noqa: E402
from src import store as S  # noqa: E402

BANK, TAX = ROOT / "data/final_dataset.csv", ROOT / "data/taxonomy.csv"
SHEET = ROOT / "outputs/blind_label_sheet_filled.xlsx"
_c = {}


def df():
    if "df" not in _c:
        _c["tax"] = D.load_taxonomy(TAX)
        _c["df"] = D.derive(D.prepare_bank(D.load_bank(BANK, _c["tax"])))
    return _c["df"]


def clf():
    if "clf" not in _c:
        _c["clf"] = C.fit_classifier(df(), C.TRAIN_LAST_YEAR, fast=True)
    return _c["clf"]


def fresh_db():
    df()
    return S.init_db(Path(tempfile.mkdtemp()) / "s.sqlite", _c["tax"])


def test_training_and_test_frames_follow_the_temporal_split():
    d = df()
    tr, te = C.training_frame(d), C.test_frame(d)
    assert tr["year"].max() == 2022 and set(te["year"]) == {2023, 2024}
    assert not tr["unlabelled"].any() and not te["unlabelled"].any()
    assert not tr["withdrawn"].any() and 2025 not in set(tr["year"]) | set(te["year"])
    assert not set(tr["qn_id"]) & set(te["qn_id"])


def test_grouped_folds_never_split_a_cluster():
    tr = C.training_frame(df())
    units = tr["unit"].to_numpy()
    for a, b in C.cv_splits(tr):
        assert not set(units[a]) & set(units[b])


def test_text_has_preamble_and_no_markup():
    t = C.classifier_text("**1.** Which ![x](a.png) is `ok`?", "Shared case")
    assert t.startswith("Shared case") and "![" not in t and "**" not in t and "`" not in t
    assert C.classifier_text("Q", float("nan")) == "Q"


def test_eligible_subtopics_match_section_1_8():
    assert len(C.eligible_subtopics(df())) == 17          # "17 of 26 qualify"


def test_predictions_are_well_formed_and_subtopic_belongs_to_module():
    d = df()
    p = C.predict_frame(clf(), C.test_frame(d))
    assert set(p["module"]) <= set(D.MODULES) and p["confidence"].between(0, 1).all()
    made = p[p["subtopic"].apply(lambda x: isinstance(x, str))]
    assert len(made) > 0
    assert all(s[0].upper() == m for s, m in zip(made["subtopic"], made["module"]))
    assert not made["abstain"].any()                       # no sub-topic after an abstained module
    assert set(made["subtopic"]) <= set(C.eligible_subtopics(d))


def test_abstention_threshold_comes_from_training_years_only():
    m = clf().module
    assert m.oof_table is not None and m.threshold in set(C.THRESHOLD_GRID) | {C.MODULE_CONF}
    ok = m.oof_table[m.oof_table["threshold"] == m.threshold]
    assert ok["accuracy_answered"].iloc[0] >= C.TARGET_ACC or m.threshold == C.MODULE_CONF
    assert C.choose_threshold(pd.DataFrame({"threshold": [0.3, 0.5], "coverage": [1, 1],
                                            "accuracy_answered": [0.5, 0.5]})) == C.MODULE_CONF


def test_score_modules_strict_and_lenient():
    s = C.score_modules(["A", "B", "C", "D"], [["A"], ["B", "C"], ["C"], ["D"]], ["A", "C", "C", "E"],
                        [False, False, False, True])
    assert s["answered"] == 3 and s["strict_acc_answered"] == round(2 / 3, 3)
    assert s["lenient_acc_answered"] == 1.0 and s["strict_acc_all"] == 0.5 and s["abstention_rate"] == 0.25


def test_temporal_evaluation_beats_chance_and_reports_both_scorings():
    ev = C.temporal_evaluation(df(), fast=True)
    assert ev["n_test"] == 75 and ev["classifier"]["lenient_acc_answered"] >= ev["classifier"]["strict_acc_answered"]
    assert ev["classifier"]["strict_acc_answered"] > 0.8 and ev["baseline_proxy"]["abstention_rate"] == 0
    assert set(ev["per_year"].index) == {2023, 2024} and set(ev["per_module"].index) == set(D.MODULES)


def test_propose_writes_classifier_rows_only_and_skips_teacher_labelled():
    d, con = df(), fresh_db()
    pool = D.unlabelled_for_approval(d)
    teacher_done = pool["qn_id"].iloc[:3].tolist()
    S.add_labels(con, d, [{"qn_id": q, "module": "A", "label_source": "teacher_blind",
                           "approved_by": "T", "approved_at": "2026-10-26"} for q in teacher_done])
    out = C.propose_labels(con, d, clf())
    rows = pd.read_sql_query("SELECT * FROM label_proposal WHERE label_source='classifier'", con)
    assert len(out) == len(pool) - 3 and not set(out["qn_id"]) & set(teacher_done)
    assert len(rows) == int(out["written"].sum()) and rows["approved_by"].isna().all()
    assert (rows["confidence"] >= clf().module.threshold).all()
    assert set(S.approved_labels(con)["qn_id"]) == set(teacher_done)       # a proposal is never an approval
    again = C.propose_labels(con, d, clf())                                # nothing is written twice
    assert again.empty or not again["written"].any()
    assert len(pd.read_sql_query("SELECT * FROM label_proposal WHERE label_source='classifier'", con)) == len(rows)


def test_total_abstention_writes_nothing(monkeypatch):
    d, con = df(), fresh_db()
    c = clf()
    monkeypatch.setattr(c.module, "threshold", 1.01)
    out = C.propose_labels(con, d, c)
    assert out["abstain"].all() and not out["written"].any()
    assert con.execute("SELECT COUNT(*) FROM label_proposal").fetchone()[0] == 0


def test_official_labels_are_never_proposed_over():
    d, con = df(), fresh_db()
    labelled = d[~d["unlabelled"]]["qn_id"].iloc[0]
    with pytest.raises(ValueError):
        S.add_label(con, d, labelled, "A", label_source="classifier")


@pytest.mark.skipif(not SHEET.exists(), reason="returned sheet not saved")
def test_old_syllabus_ids_and_blind_evaluation():
    ids = C.old_syllabus_ids(SHEET)
    assert len(ids) == 5 and all(i.startswith("ICT_") for i in ids)
    con = S.connect_ro(ROOT / "data/state.sqlite")
    try:
        ev = C.evaluate_blind(df(), con, ids, clf=clf())
    finally:
        con.close()
    assert ev["n"] == 30 and ev["n_excluded"] == 5 and ev["without_excluded"]["module"]["n"] == 25
