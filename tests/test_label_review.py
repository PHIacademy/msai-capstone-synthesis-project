"""Tests for the label review sheet (src/review.py, second part) and the label_review log in src/store.py."""
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from openpyxl import load_workbook  # noqa: E402

from src import data as D  # noqa: E402
from src import review as R  # noqa: E402
from src import store as S  # noqa: E402

tax = D.load_taxonomy(ROOT / "data/taxonomy.csv")
df = D.derive(D.load_bank(ROOT / "data/final_dataset.csv", tax))
SUB_MODULE = dict(zip(tax.qn_type_id, tax.module))


def make_store(n_abstain=12):
    """A store with 30 blind labels and classifier rows for all but n_abstain of the other 48."""
    d = Path(tempfile.mkdtemp())
    con = S.init_db(d / "s.sqlite", tax)
    blind = R.draw_blind_sample(df, 30, 42)
    S.add_labels(con, df, [{"qn_id": q, "module": "A", "label_source": "teacher_blind", "approved_by": "T",
                            "approved_at": "2026-10-26"} for q in blind.qn_id])
    rest = D.unlabelled_for_approval(df)
    rest = rest[~rest.qn_id.isin(blind.qn_id)].sort_values("qn_id")
    assert len(rest) == 48
    mods = ["A", "B", "C", "D", "E"]
    labels = []
    for k, q in enumerate(rest.qn_id.tolist()[n_abstain:]):
        m = mods[k % 5]
        sub = next(s for s, mod in SUB_MODULE.items() if mod == m) if k % 3 == 0 else None
        labels.append({"qn_id": q, "module": m, "subtopic": sub, "label_source": "classifier", "confidence": 0.5 + k / 100})
    S.add_labels(con, df, labels)
    return d, con


def export(d, con):
    w, b = R.draw_review_set(df, con)
    R.export_review_sheet(df, tax, w, b, d / "r.xlsx", d / "m.csv", drawn_on="2026-10-27")
    return w, b


def fill(d, name="Teacher A", date="2026-10-28", change_every=4, mins=1.5, edit=None):
    wb = load_workbook(d / "r.xlsx")
    for sn in (R.BLIND2_SHEET, R.REVIEW_SHEET):
        ws = wb[sn]
        ws[R.NAME_CELL], ws[R.DATE_CELL] = name, date
        r, k = R.FIRST_ROW, 0
        while ws.cell(r, 2).value:
            if sn == R.BLIND2_SHEET:
                ws.cell(r, 6).value, ws.cell(r, 8).value = "B", mins
            else:
                if k % change_every == 0:
                    cur = ws.cell(r, 6).value
                    ws.cell(r, 9).value, ws.cell(r, 10).value = "Change", "E" if cur != "E" else "D"
                else:
                    ws.cell(r, 9).value = "Accept"
                ws.cell(r, 12).value = mins
                k += 1
            r += 1
    if edit:
        edit(wb)
    wb.save(d / "r_done.xlsx")
    return d / "r_done.xlsx"


def test_split_is_36_and_12_reproducible_and_blind_part_has_no_prediction():
    d, con = make_store()
    w, b = R.draw_review_set(df, con)
    w2, _ = R.draw_review_set(df, con)
    assert (len(w), len(b)) == (36, 12) and w.qn_id.tolist() == w2.qn_id.tolist()
    assert w.qn_id.tolist() != sorted(w.qn_id)            # shuffled
    export(d, con)
    wb = load_workbook(d / "r.xlsx")
    assert wb.sheetnames == ["Label", "Review", "Reference"] or set(wb.sheetnames) == {"Instructions", "Label", "Review", "Reference"}
    hdr = [c.value for c in wb[R.BLIND2_SHEET][R.HEADER_ROW]]
    assert not any("lassifier" in str(h) or "onfidence" in str(h) for h in hdr)
    m = pd.read_csv(d / "m.csv")
    assert m.part.value_counts().to_dict() == {"with_proposal": 36, "blind_no_proposal": 12}


def test_import_stores_labels_and_review_log():
    d, con = make_store()
    export(d, con)
    res = R.import_review_sheet(fill(d), d / "m.csv", con, df, tax)
    assert res == {"imported": 48, "accepted": 27, "changed": 9, "labelled_blind": 12}
    ap = S.approved_labels(con)
    assert (ap.label_source == "teacher_approved").sum() == 48 and len(ap) == 78
    log = pd.read_sql_query("SELECT * FROM label_review", con)
    assert len(log) == 48 and (log.seconds_spent == 90).all() and set(log.reviewer) == {"Teacher A"}
    # an accepted label equals the proposal; a changed one differs
    m = pd.read_csv(d / "m.csv").set_index("qn_id")
    lab = ap.set_index("qn_id")
    acc = log[log.decision == "accept"]
    assert all(lab.loc[q, "module"] == m.loc[q, "proposed_module"] for q in acc.qn_id)
    chg = log[log.decision == "change"]
    assert all(lab.loc[q, "module"] != m.loc[q, "proposed_module"] for q in chg.qn_id)
    # the whole bank is now approved: effective bank = 519 core + 78 earlier-or-new approved
    eff = S.build_effective_bank(df, con)
    assert len(eff) == 559                      # 560 less the one withdrawn question: every question is now served


def test_second_import_is_refused_and_changes_nothing():
    d, con = make_store()
    export(d, con)
    f = fill(d)
    R.import_review_sheet(f, d / "m.csv", con, df, tax)
    with pytest.raises(ValueError, match="already stored"):
        R.import_review_sheet(f, d / "m.csv", con, df, tax)
    assert con.execute("SELECT COUNT(*) FROM label_review").fetchone()[0] == 48


@pytest.mark.parametrize("what,match", [
    ("noname", "name is empty"),
    ("notime", "time taken"),
    ("badtime", "time taken"),
    ("nodecision", "Accept or Change"),
    ("accept_with_final", "Accept, but"),
    ("change_same", "equals the proposal"),
    ("change_nomodule", "no final module"),
    ("blind_nomodule", "module is empty"),
    ("badsub", "not in module"),
    ("diffname", "different names"),
    ("missingrow", "differ from the manifest"),
])
def test_bad_sheets_are_refused_and_nothing_is_stored(what, match):
    d, con = make_store()
    w, b = export(d, con)

    def edit(wb):
        rv, lb = wb[R.REVIEW_SHEET], wb[R.BLIND2_SHEET]
        r0 = R.FIRST_ROW
        if what == "noname":
            rv[R.NAME_CELL] = ""
        elif what == "notime":
            rv.cell(r0, 12).value = None
        elif what == "badtime":
            rv.cell(r0, 12).value = 9999
        elif what == "nodecision":
            rv.cell(r0, 9).value = None
        elif what == "accept_with_final":
            rv.cell(r0, 9).value, rv.cell(r0, 10).value = "Accept", "A"
        elif what == "change_same":
            rv.cell(r0, 9).value, rv.cell(r0, 10).value = "Change", rv.cell(r0, 6).value
            rv.cell(r0, 11).value = None if rv.cell(r0, 7).value == "(none)" else rv.cell(r0, 7).value
        elif what == "change_nomodule":
            rv.cell(r0, 9).value, rv.cell(r0, 10).value = "Change", None
        elif what == "blind_nomodule":
            lb.cell(r0, 6).value = None
        elif what == "badsub":
            m = "A"
            rv.cell(r0, 9).value, rv.cell(r0, 10).value = "Change", m
            rv.cell(r0, 11).value = next(s for s, mod in SUB_MODULE.items() if mod != m)
        elif what == "diffname":
            lb[R.NAME_CELL] = "Someone else"
        elif what == "missingrow":
            lb.delete_rows(r0)

    f = fill(d, edit=edit)
    with pytest.raises(ValueError, match=match):
        R.import_review_sheet(f, d / "m.csv", con, df, tax)
    assert con.execute("SELECT COUNT(*) FROM label_review").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM label_proposal WHERE label_source='teacher_approved'").fetchone()[0] == 0


def test_refused_if_proposal_in_store_changed_since_export():
    d, con = make_store()
    export(d, con)
    with con:
        con.execute("UPDATE label_proposal SET predicted_module='E', predicted_subtopic=NULL "
                    "WHERE label_source='classifier' AND predicted_module='A'")
    with pytest.raises(ValueError, match="no longer matches"):
        R.import_review_sheet(fill(d), d / "m.csv", con, df, tax)


def test_label_review_table_is_created_on_an_older_store_and_is_idempotent():
    d, con = make_store()
    assert con.execute("SELECT 1 FROM sqlite_master WHERE name='label_review'").fetchone() is None
    S.ensure_label_review(con)
    S.ensure_label_review(con)
    assert con.execute("SELECT 1 FROM sqlite_master WHERE name='label_review'").fetchone()
