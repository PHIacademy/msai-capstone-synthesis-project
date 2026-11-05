"""Tests for the blind label sheet (src/review.py, first part): draw, export, import and the checks that refuse a bad sheet."""
import sys
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


def exported(tmp_path):
    sample = R.draw_blind_sample(df)
    R.export_blind_sheet(df, tax, sample, tmp_path / "b.xlsx", tmp_path / "m.csv", drawn_on="2026-10-24")
    return sample


def filled(tmp_path, edit=None, name="Teacher A", date="2026-10-26"):
    wb = load_workbook(tmp_path / "b.xlsx")
    ws = wb[R.SHEET]
    ws[R.NAME_CELL], ws[R.DATE_CELL] = name, date
    r = R.FIRST_ROW
    while ws.cell(r, 2).value:
        ws.cell(r, 6).value = "A"
        r += 1
    if edit:
        edit(ws)
    wb.save(tmp_path / "b_done.xlsx")
    return tmp_path / "b_done.xlsx"


def test_the_draw_is_reproducible_and_comes_from_the_78_awaiting_questions():
    a, b = R.draw_blind_sample(df), R.draw_blind_sample(df)
    assert a.qn_id.tolist() == b.qn_id.tolist() and len(a) == 30
    assert set(a.qn_id) <= set(D.unlabelled_for_approval(df).qn_id) and a.qn_id.is_unique
    assert a.qn_id.tolist() != R.draw_blind_sample(df, 30, 7).qn_id.tolist()


def test_the_sample_size_must_be_between_one_and_the_population():
    for n in (0, 79):
        with pytest.raises(ValueError, match="between 1 and 78"):
            R.draw_blind_sample(df, n)


def test_the_sheet_shows_no_prediction_answer_key_or_cohort_percentage(tmp_path):
    exported(tmp_path)
    wb = load_workbook(tmp_path / "b.xlsx")
    header = [str(c.value) for c in wb[R.SHEET][R.HEADER_ROW]]
    assert header == R.COLUMNS
    text = " ".join(str(c.value) for row in wb[R.SHEET].iter_rows() for c in row if c.value).lower()
    assert "confidence" not in text and "cohort" not in " ".join(header).lower()
    assert any("BLIND" in str(c.value) for row in wb[R.INSTRUCTIONS].iter_rows() for c in row)
    m = pd.read_csv(tmp_path / "m.csv")
    assert list(m.columns) == ["order", "qn_id", "year", "needs_figure", "seed", "sample_size", "population", "drawn_on"]
    assert len(m) == 30 and set(m.population) == {78} and set(m.seed) == {42}


def test_the_options_never_show_the_paper_end_banner_and_figures_are_marked(tmp_path):
    sample = exported(tmp_path)
    ws = load_workbook(tmp_path / "b.xlsx")[R.SHEET]
    rows = [(ws.cell(r, 2).value, ws.cell(r, 4).value, ws.cell(r, 5).value) for r in range(R.FIRST_ROW, R.FIRST_ROW + len(sample))]
    assert [q for q, _, _ in rows] == sample.qn_id.tolist()                 # the order on the sheet is the order of the draw
    assert not any("END OF" in str(o) for _, o, _ in rows)
    m = pd.read_csv(tmp_path / "m.csv")
    assert m.qn_id.tolist() == [q for q, _, _ in rows]
    assert all(f for (_, _, f), need in zip(rows, m.needs_figure) if need)          # every question that needs a look says where
    assert sum(1 for _, _, f in rows if f) >= int(m.needs_figure.sum()) > 0


def test_a_returned_sheet_is_stored_as_blind_labels(tmp_path):
    exported(tmp_path)
    con = S.init_db(tmp_path / "s.sqlite", tax)
    assert R.import_blind_sheet(filled(tmp_path), tmp_path / "m.csv", con, df, tax) == 30
    lab = S.approved_labels(con)
    assert len(lab) == 30 and set(lab.label_source) == {"teacher_blind"}
    assert set(lab.approved_by) == {"Teacher A"} and set(lab.approved_at) == {"2026-10-26"} and set(lab.module) == {"A"}


def edit_noname(ws):
    ws[R.NAME_CELL] = ""


def edit_nomodule(ws):
    ws.cell(R.FIRST_ROW, 6).value = None


def edit_wrong_module_subtopic(ws):
    ws.cell(R.FIRST_ROW, 6).value = "A"
    ws.cell(R.FIRST_ROW, 7).value = next(s for s, mod in SUB_MODULE.items() if mod != "A")


def edit_missing_row(ws):
    ws.delete_rows(R.FIRST_ROW)


@pytest.mark.parametrize("edit,match", [
    (edit_noname, "teacher name is empty"),
    (edit_nomodule, "module is empty or not A-E"),
    (edit_wrong_module_subtopic, "not in module A"),
    (edit_missing_row, "differ from the manifest"),
])
def test_bad_sheets_are_refused_and_nothing_is_stored(tmp_path, edit, match):
    exported(tmp_path)
    con = S.init_db(tmp_path / "s.sqlite", tax)
    with pytest.raises(ValueError, match=match):
        R.import_blind_sheet(filled(tmp_path, edit=edit), tmp_path / "m.csv", con, df, tax)
    assert con.execute("SELECT COUNT(*) FROM label_proposal").fetchone()[0] == 0
