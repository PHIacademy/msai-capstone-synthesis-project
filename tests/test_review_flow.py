"""End-to-end test of the label flow on a scratch store: blind sheet -> classifier proposals -> review sheet -> import.

The 'teacher' is synthetic (module A for the blind sheet; Accept, one Change and module A for the rest), so this
tests the code paths and the all-or-nothing checks, not the classifier's quality."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from openpyxl import load_workbook  # noqa: E402

from src import classifier as C  # noqa: E402
from src import data as D  # noqa: E402
from src import review as R  # noqa: E402
from src import store as S  # noqa: E402

tax = D.load_taxonomy(ROOT / "data/taxonomy.csv")
df = D.derive(D.prepare_bank(D.load_bank(ROOT / "data/final_dataset.csv", tax)))


@pytest.fixture(scope="module")
def flow(tmp_path_factory):
    d = tmp_path_factory.mktemp("flow")
    con = S.init_db(d / "state.sqlite", tax)
    # 1. blind step
    sample = R.draw_blind_sample(df)
    R.export_blind_sheet(df, tax, sample, d / "blind.xlsx", d / "blind_manifest.csv")
    wb = load_workbook(d / "blind.xlsx")
    ws = wb[R.SHEET]
    ws[R.NAME_CELL], ws[R.DATE_CELL] = "Teacher T", "2026-10-26"
    for r in range(R.FIRST_ROW, R.FIRST_ROW + len(sample)):
        ws.cell(row=r, column=6, value="A")
    wb.save(d / "blind_done.xlsx")
    n_blind = R.import_blind_sheet(d / "blind_done.xlsx", d / "blind_manifest.csv", con, df, tax)
    return {"d": d, "con": con, "n_blind": n_blind}


def test_blind_import_and_a_second_import_is_refused(flow):
    assert flow["n_blind"] == 30
    with pytest.raises(ValueError, match="already stored"):
        R.import_blind_sheet(flow["d"] / "blind_done.xlsx", flow["d"] / "blind_manifest.csv", flow["con"], df, tax)


def test_proposals_then_review_sheet_then_import(flow):
    d, con = flow["d"], flow["con"]
    clf = C.fit_classifier(df, int(df.loc[~df.unlabelled, "year"].max()), fast=True)
    out = C.propose_labels(con, df, clf)
    assert len(out) == 48 and int(out.written.sum()) + int(out.abstain.sum()) == 48
    assert "subtopic_conf" in out.columns
    rows = pd.read_sql_query("SELECT label_source, COUNT(*) n, SUM(approved_by IS NOT NULL) a FROM label_proposal GROUP BY 1", con)
    assert rows.set_index("label_source").loc["classifier", "a"] == 0           # a proposal is never an approval
    assert len(S.build_effective_bank(df, con)) == 519 + int((R.D.unlabelled_for_approval(df).qn_id.isin(S.approved_labels(con).qn_id) & (R.D.unlabelled_for_approval(df).year == 2025)).sum())
    again = C.propose_labels(con, df, clf)                                      # safe to repeat
    assert int(again.written.sum()) == 0

    with_prop, no_prop = R.draw_review_set(df, con)
    assert len(with_prop) == int(out.written.sum()) and len(no_prop) == int(out.abstain.sum())
    R.export_review_sheet(df, tax, with_prop, no_prop, d / "review.xlsx", d / "review_manifest.csv")
    wb = load_workbook(d / "review.xlsx")
    for name in (R.BLIND2_SHEET, R.REVIEW_SHEET):
        wb[name][R.NAME_CELL], wb[name][R.DATE_CELL] = "Teacher T", "2026-10-28"
    ws = wb[R.BLIND2_SHEET]
    for r in range(R.FIRST_ROW, R.FIRST_ROW + len(no_prop)):
        ws.cell(row=r, column=6, value="A"); ws.cell(row=r, column=8, value=0.5)
    ws = wb[R.REVIEW_SHEET]
    for i in range(len(with_prop)):
        r = R.FIRST_ROW + i
        if i == 0:                                    # one Change, to a different module than proposed
            other = "B" if with_prop.proposed_module.iloc[0] != "B" else "C"
            ws.cell(row=r, column=9, value="Change"); ws.cell(row=r, column=10, value=other)
        else:
            ws.cell(row=r, column=9, value="Accept")
        ws.cell(row=r, column=12, value=0.3)
    wb.save(d / "review_done.xlsx")

    bad = load_workbook(d / "review_done.xlsx")
    bad[R.REVIEW_SHEET].cell(row=R.FIRST_ROW + 1, column=12).value = None      # one missing time -> whole sheet refused
    bad.save(d / "review_bad.xlsx")
    with pytest.raises(ValueError, match="time taken"):
        R.import_review_sheet(d / "review_bad.xlsx", d / "review_manifest.csv", con, df, tax)
    assert con.execute("SELECT COUNT(*) FROM label_proposal WHERE label_source='teacher_approved'").fetchone()[0] == 0

    res = R.import_review_sheet(d / "review_done.xlsx", d / "review_manifest.csv", con, df, tax)
    assert res == {"imported": 48, "accepted": len(with_prop) - 1, "changed": 1, "labelled_blind": len(no_prop)}
    eb = S.build_effective_bank(df, con)
    assert len(eb) == 559 and set(eb.label_level) <= {"official", "subtopic", "module"}
    assert len(C.propose_labels(con, df, clf)) == 0                              # nothing left to propose
    empty = C.propose_labels(con, df, clf)
    assert list(empty.columns)[-2:] == ["subtopic_conf", "written"]              # the empty result keeps every column
    with pytest.raises(ValueError, match="already stored"):
        R.import_review_sheet(d / "review_done.xlsx", d / "review_manifest.csv", con, df, tax)
