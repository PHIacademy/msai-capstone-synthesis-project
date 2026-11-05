"""Teacher label approval, blind step (Section 2.2 step 2, Section 3.6 of the notebook).

Place at src/review.py. It holds two steps. The blind step: draw the random sample of unlabelled
questions, export it as a spreadsheet the teacher fills in WITHOUT seeing any prediction, and import
the returned sheet into the governed store as 'teacher_blind' labels (the ground truth for measuring
classifier accuracy, including on the 2025 paper). The label review step (second half of the file): the
teacher accepts or changes the classifier's proposals and labels the rest, stored as 'teacher_approved'
with a review log. The candidate review queue and rejection codes come later.

Blindness rules built in
- The sheet carries no prediction, no answer key and no cohort difficulty.
- Rows are in random order; IDs are pre-filled so the import can check the teacher returned
  exactly the sampled questions.
- The text shown is the text the classifier will see (code-only images appear as their OCR
  text); figure and table images are marked, not embedded.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import re
from pathlib import Path

import pandas as pd

from src import data as D
from src import store as S

BLIND_N_DEFAULT = 30          # starting value: indicative 95% interval about +/-0.16 at 70% accuracy (Section 1.8, n = 30)
BLIND_SEED_DEFAULT = 42       # the seed used throughout Capstone 5

SHEET, REFERENCE, INSTRUCTIONS = "Labels", "Reference", "Instructions"
NAME_CELL, DATE_CELL, HEADER_ROW, FIRST_ROW = "C2", "C3", 6, 7
COLUMNS = ["No.", "Question ID", "Question", "Options", "Figure / table to check",
           "Module (A-E)", "Sub-topic (only if sure)", "Notes", "Check"]
_MD_IMAGE_ANY = re.compile(r"!\[([^\]]*)\]\(([^)]*)\)")


# ------------------------------------------------------------------ sampling
def draw_blind_sample(df: pd.DataFrame, n: int = BLIND_N_DEFAULT,
                      seed: int = BLIND_SEED_DEFAULT) -> pd.DataFrame:
    """Simple random sample of the 78 questions that need a label (2025 and earlier
    unclassified). Reproducible: same bank, n and seed give the same questions in the same
    order, and that order is the order on the sheet."""
    pool = D.unlabelled_for_approval(df)
    if not 1 <= n <= len(pool):
        raise ValueError(f"n must be between 1 and {len(pool)}")
    return pool.sample(n=n, random_state=seed)


# ----------------------------------------------------------- what the teacher sees
def _clean(text: str) -> str:
    """Make stored markdown readable in a cell: image links become [IMAGE: ...] markers."""
    t = _MD_IMAGE_ANY.sub(lambda m: f"[IMAGE: {m.group(1) or Path(m.group(2)).stem}]", str(text))
    return t.replace("**", "").strip()


def _question_cell(row: pd.Series) -> str:
    q = _clean(row["question_text"])
    if row["cluster"] == 1 and isinstance(row["cluster_desc"], str) and row["cluster_desc"].strip():
        return f"[Shared context]\n{_clean(row['cluster_desc'])}\n\n[This question]\n{q}"
    return q


def _options_cell(row: pd.Series) -> str:
    # The bank keeps the paper-end banner at the end of option D of the last question of each paper;
    # data.py removes it (the data themselves are unchanged).
    text = D.strip_paper_end(row["answer_options"])
    opts = D.parse_options(text)
    if opts:
        return "\n".join(f"{k}. {v}" for k, v in opts.items())
    return "[options are an image]" if D.IMAGE_SENTINEL in text else _clean(text)


def _image_info(row: pd.Series) -> tuple:
    """(cell text, needs_look). Names the figure or table the teacher has to look up in the past
    paper (not embedded). A shared image whose code already appears as a fenced text block was
    OCR-converted, so it is named but not counted as something to look up."""
    lines, needs_look = [], False
    if row["figure_dependent"]:
        needs_look = True
        for kind, path in zip(D.split_list(row["qn_image_type"]), D.split_list(row["qn_image_filename"])):
            lines.append(f"{kind}: {Path(path).name}")
    desc = row["cluster_desc"]
    if row["cluster"] == 1 and isinstance(desc, str):
        shown_as_text = "```" in desc
        for _, path in _MD_IMAGE_ANY.findall(desc):
            if shown_as_text:
                lines.append(f"shared image, code shown as text: {Path(path).name}")
            else:
                lines.append(f"shared figure: {Path(path).name}")
                needs_look = True
    lines = list(dict.fromkeys(lines))
    if needs_look:
        qn = int(row["qn_id"].rsplit("_q", 1)[1])
        lines.append(f"-> {row['year']} Paper 1A, Q{qn}")
    return "\n".join(lines), needs_look


def _height(*cells_and_widths) -> float:
    """Row height (points) from the longest wrapped cell; capped at Excel's 409."""
    lines = max(sum(max(1, math.ceil(len(ln) / w)) for ln in str(txt).split("\n"))
                for txt, w in cells_and_widths)
    return min(409.0, max(32.0, 15.0 * lines + 6))


# ------------------------------------------------------------------- export
def export_blind_sheet(df: pd.DataFrame, tax: pd.DataFrame, sample: pd.DataFrame,
                       xlsx_path: str | Path, manifest_path: str | Path,
                       seed: int = BLIND_SEED_DEFAULT, drawn_on: str | None = None) -> None:
    """Write the teacher's workbook and the sample manifest (IDs only, no labels)."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill, Protection
    from openpyxl.worksheet.datavalidation import DataValidation
    from openpyxl.worksheet.properties import PageSetupProperties

    prep = D.prepare_bank(sample)         # code-only images count as text, as for the classifier
    sample = sample.assign(question_text=prep["question_text"].values)
    n = len(sample)
    last = FIRST_ROW + n - 1
    img_info = [_image_info(r) for _, r in sample.iterrows()]
    has_fig = pd.Series([need for _, need in img_info], index=sample.index)   # a figure/table the teacher must open
    n_fig = int(has_fig.sum())
    drawn_on = drawn_on or dt.date.today().isoformat()

    wb = Workbook()
    ws_i, ws, ws_r = wb.active, wb.create_sheet(SHEET), wb.create_sheet(REFERENCE)
    ws_i.title = INSTRUCTIONS
    wrap_top = Alignment(wrap_text=True, vertical="top")
    bold, head_fill = Font(bold=True), PatternFill("solid", fgColor="D9E2F3")
    input_fill = PatternFill("solid", fgColor="FFF2CC")

    # --- instructions
    ws_i.column_dimensions["A"].width = 100
    ws_i.page_setup.orientation, ws_i.page_setup.paperSize = "landscape", ws_i.PAPERSIZE_A4
    ws_i.page_setup.fitToWidth, ws_i.page_setup.fitToHeight = 1, 0
    ws_i.sheet_properties.pageSetUpPr = PageSetupProperties(fitToPage=True)
    text = [
        ("Blind labelling sheet: HKDSE ICT Paper 1A", True),
        ("", False),
        (f"Purpose. Your labels are the ground truth used to measure how accurate the classifier is, "
         f"including on the 2025 paper. This is the BLIND step: no prediction exists for these {n} questions "
         f"and none is shown. Please work from your own judgement only (no classifier output, no AI tool).", False),
        ("", False),
        ("For each row on the 'Labels' sheet:", True),
        ("1. Module (A-E): the module the question mainly tests. Use the dropdown; titles are above the table.", False),
        ("2. Sub-topic: choose one only if you are sure (list on the 'Reference' sheet); otherwise leave it blank. "
         "It must belong to the module you chose; the Check column warns if it does not.", False),
        ("3. If a question really spans two modules, choose the main one and name the other in Notes.", False),
        ("4. Notes: anything you want recorded (optional).", False),
        ("", False),
        ("Also fill in your name and the date (YYYY-MM-DD) at the top of the 'Labels' sheet. Every row must be completed.", False),
        ("", False),
        (f"Figures and tables. {n_fig} of the {n} questions refer to a figure or table that is NOT embedded in "
         f"this sheet. The 'Figure / table to check' column gives the file name and the paper, year and question "
         f"number to open in the past paper." if n_fig else "No question in this sample depends on a figure or table.", False),
        ("Questions of a shared-context pair show the shared text first. Code that was an image in the paper is shown as text.", False),
        ("", False),
        ("Not shown on purpose: the official answer key and the cohort percentage.", False),
        ("Only the yellow cells can be edited. Please return the file under the same name with a suffix, for example _done.", False),
    ]
    for r, (t, b) in enumerate(text, start=1):
        c = ws_i.cell(row=r, column=1, value=t)
        c.alignment, c.font = wrap_top, Font(bold=b, size=14 if r == 1 else 11)

    # --- reference (taxonomy)
    for j, h in enumerate(["Sub-topic id", "Module", "Module title", "Sub-topic name"], start=1):
        c = ws_r.cell(row=1, column=j, value=h)
        c.font, c.fill = bold, head_fill
    for i, r in enumerate(tax.itertuples(index=False), start=2):
        for j, v in enumerate([r.qn_type_id, r.module, r.module_name, r.subtopic_name], start=1):
            ws_r.cell(row=i, column=j, value=v)
    for col, w in zip("ABCD", (14, 9, 42, 52)):
        ws_r.column_dimensions[col].width = w
    ref_last = len(tax) + 1

    # --- labels sheet
    widths = [6, 24, 70, 52, 34, 13, 18, 36, 28]
    for j, w in enumerate(widths, start=1):
        ws.column_dimensions[chr(64 + j)].width = w
    ws["A1"], ws["A1"].font = "Blind labelling sheet: HKDSE ICT Paper 1A (no predictions shown)", Font(bold=True, size=13)
    ws["B2"], ws["B3"], ws["B4"] = "Teacher name", "Date (YYYY-MM-DD)", "Completed"
    for a in ("B2", "B3", "B4"):
        ws[a].font = bold
    for a in (NAME_CELL, DATE_CELL):
        ws[a].fill, ws[a].protection = input_fill, Protection(locked=False)
    ws[DATE_CELL].number_format = "@"
    ws["C4"] = f'=COUNTA(F{FIRST_ROW}:F{last})&" of {n}"'
    ws["B5"] = "Modules"
    ws["B5"].font = bold
    ws["C5"] = "   |   ".join(f"{k}  {v}" for k, v in D.OFFICIAL_MODULE_NAMES.items())
    ws["C5"].alignment = wrap_top
    ws.row_dimensions[5].height = 48

    for j, h in enumerate(COLUMNS, start=1):
        c = ws.cell(row=HEADER_ROW, column=j, value=h)
        c.font, c.fill, c.alignment = bold, head_fill, Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[HEADER_ROW].height = 32

    for i, (_, row) in enumerate(sample.iterrows()):
        r = FIRST_ROW + i
        q, o, im = _question_cell(row), _options_cell(row), img_info[i][0]
        vals = [i + 1, row["qn_id"], q, o, im]
        for j, v in enumerate(vals, start=1):
            c = ws.cell(row=r, column=j, value=v)
            c.alignment = wrap_top
        for j in (6, 7, 8):
            c = ws.cell(row=r, column=j)
            c.fill, c.protection, c.alignment = input_fill, Protection(locked=False), wrap_top
        ws.cell(row=r, column=9, value=(
            f'=IF(AND(G{r}<>"",F{r}=""),"choose a module first",'
            f'IF(AND(G{r}<>"",UPPER(F{r})<>UPPER(LEFT(G{r},1))),"sub-topic is not in this module",""))')
        ).alignment = wrap_top
        ws.row_dimensions[r].height = _height((q, widths[2]), (o, widths[3]), (im, widths[4]))

    dv_mod = DataValidation(type="list", formula1='"A,B,C,D,E"', allow_blank=True, showErrorMessage=True,
                            errorTitle="Module", error="Choose A, B, C, D or E.")
    dv_sub = DataValidation(type="list", formula1=f"={REFERENCE}!$A$2:$A${ref_last}", allow_blank=True,
                            showErrorMessage=True, errorTitle="Sub-topic", error="Choose an id from the Reference sheet, or leave blank.")
    ws.add_data_validation(dv_mod)
    ws.add_data_validation(dv_sub)
    dv_mod.add(f"F{FIRST_ROW}:F{last}")
    dv_sub.add(f"G{FIRST_ROW}:G{last}")
    ws.freeze_panes = f"C{FIRST_ROW}"
    ws.page_setup.orientation, ws.page_setup.paperSize = "landscape", ws.PAPERSIZE_A4
    ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0      # one page wide, as many tall as needed
    ws.sheet_properties.pageSetUpPr = PageSetupProperties(fitToPage=True)
    ws.print_title_rows = f"{HEADER_ROW}:{HEADER_ROW}"
    ws.protection.sheet = True
    ws.protection.formatColumns = False     # False = allowed: the teacher may resize columns and rows
    ws.protection.formatRows = False
    ws.sheet_view.zoomScale = 100
    wb.active = wb.index(ws)
    Path(xlsx_path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(xlsx_path)

    manifest = pd.DataFrame({
        "order": range(1, n + 1), "qn_id": sample["qn_id"].values, "year": sample["year"].astype(int).values,
        "needs_figure": has_fig.values, "seed": seed, "sample_size": n,
        "population": len(D.unlabelled_for_approval(df)), "drawn_on": drawn_on})
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)


# ------------------------------------------------------------------- import
def _read_date(v) -> str:
    if isinstance(v, (dt.datetime, dt.date)):
        return (v.date() if isinstance(v, dt.datetime) else v).isoformat()
    s = str(v or "").strip()
    try:
        return dt.date.fromisoformat(s).isoformat()
    except ValueError:
        raise ValueError(f"date {s!r} is not in the form YYYY-MM-DD") from None


def read_blind_sheet(xlsx_path: str | Path) -> tuple:
    """Return (teacher, date, rows) where rows are dicts qn_id, module, subtopic, notes."""
    from openpyxl import load_workbook
    ws = load_workbook(xlsx_path, data_only=True)[SHEET]
    teacher = str(ws[NAME_CELL].value or "").strip()
    date = _read_date(ws[DATE_CELL].value)
    rows, r = [], FIRST_ROW
    while ws.cell(row=r, column=2).value not in (None, ""):
        rows.append({"qn_id": str(ws.cell(row=r, column=2).value).strip(),
                     "module": str(ws.cell(row=r, column=6).value or "").strip().upper(),
                     "subtopic": str(ws.cell(row=r, column=7).value or "").strip(),
                     "notes": str(ws.cell(row=r, column=8).value or "").strip(), "row": r})
        r += 1
    return teacher, date, rows


def import_blind_sheet(xlsx_path: str | Path, manifest_path: str | Path, con, df: pd.DataFrame,
                       tax: pd.DataFrame) -> int:
    """Check the returned sheet and store its labels as 'teacher_blind'. All or nothing.

    Refuses the sheet if the teacher name or date is missing, if the questions are not exactly
    those in the manifest, if any module is missing or invalid, if a sub-topic is unknown or
    belongs to another module, or if blind labels for these questions already exist."""
    teacher, date, rows = read_blind_sheet(xlsx_path)
    expected = pd.read_csv(manifest_path)["qn_id"].tolist()
    problems = []
    if not teacher:
        problems.append(f"teacher name is empty ({NAME_CELL})")
    ids = [r["qn_id"] for r in rows]
    if sorted(ids) != sorted(expected):
        problems.append(f"questions differ from the manifest: missing {sorted(set(expected) - set(ids))[:5]}, "
                        f"unexpected {sorted(set(ids) - set(expected))[:5]}, duplicates {len(ids) - len(set(ids))}")
    sub_module = dict(zip(tax["qn_type_id"], tax["module"]))
    for r in rows:
        if r["module"] not in D.MODULES:
            problems.append(f"row {r['row']} ({r['qn_id']}): module is empty or not A-E")
        elif r["subtopic"] and sub_module.get(r["subtopic"]) != r["module"]:
            problems.append(f"row {r['row']} ({r['qn_id']}): sub-topic {r['subtopic']!r} is unknown or not in module {r['module']}")
    have = {x[0] for x in con.execute("SELECT qn_id FROM label_proposal WHERE label_source='teacher_blind'")}
    dup = sorted(have & set(ids))
    if dup:
        problems.append(f"blind labels already stored for {dup[:5]}")
    if problems:
        raise ValueError("sheet not imported:\n  " + "\n  ".join(problems))
    S.add_labels(con, df, [{"qn_id": r["qn_id"], "module": r["module"], "subtopic": r["subtopic"] or None,
                            "label_source": "teacher_blind", "approved_by": teacher, "approved_at": date}
                           for r in rows])
    return len(rows)


# =====================================================================================
# Label review sheet (Section 3.6, the step after the blind one)
# =====================================================================================
# The 48 questions still awaiting approval after the blind labels. For the questions where the classifier
# gave a module (36) the teacher sees the proposal and accepts or changes it; for the others (12) the
# teacher labels without any prediction, on a separate sheet, as in the blind step. Every question gets a
# 'teacher_approved' label and one label_review row (what was shown, accepted or changed, minutes taken).
REVIEW_SHEET, BLIND2_SHEET = "Review", "Label"
REVIEW_COLS = ["No.", "Question ID", "Question", "Options", "Figure / table to check", "Classifier module",
               "Classifier sub-topic", "Confidence", "Decision (Accept / Change)", "Final module (only if Change)",
               "Final sub-topic (only if Change, and sure)", "Time taken (minutes)", "Notes", "Check"]
LABEL_COLS = ["No.", "Question ID", "Question", "Options", "Figure / table to check", "Module (A-E)",
              "Sub-topic (only if sure)", "Time taken (minutes)", "Notes", "Check"]
MAX_MINUTES = 120


def draw_review_set(df: pd.DataFrame, con, seed: int = BLIND_SEED_DEFAULT) -> tuple:
    """Split the questions still awaiting approval into (with_proposal, no_proposal), each in a fixed
    random order. with_proposal carries the classifier row (proposal_id, module, subtopic, confidence)."""
    S.ensure_label_review(con)
    pool = D.unlabelled_for_approval(df)
    pool = pool[~pool["qn_id"].isin(set(S.approved_labels(con)["qn_id"]))]
    prop = pd.read_sql_query(
        "SELECT id AS proposal_id, qn_id, predicted_module AS proposed_module, predicted_subtopic AS proposed_subtopic, "
        "confidence FROM label_proposal WHERE label_source='classifier' ORDER BY id", con).drop_duplicates("qn_id", keep="last")
    pool = pool.sort_values("qn_id").merge(prop, on="qn_id", how="left")
    shuffled = lambda x: x.sample(frac=1, random_state=seed).reset_index(drop=True)
    return shuffled(pool[pool["proposal_id"].notna()]), shuffled(pool[pool["proposal_id"].isna()])


def export_review_sheet(df: pd.DataFrame, tax: pd.DataFrame, with_prop: pd.DataFrame, no_prop: pd.DataFrame,
                        xlsx_path: str | Path, manifest_path: str | Path, seed: int = BLIND_SEED_DEFAULT,
                        drawn_on: str | None = None) -> None:
    """Write the teacher's workbook (sheets Instructions, Label, Review, Reference) and the manifest."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill, Protection
    from openpyxl.worksheet.datavalidation import DataValidation
    from openpyxl.worksheet.properties import PageSetupProperties

    drawn_on = drawn_on or dt.date.today().isoformat()
    wb = Workbook()
    ws_i = wb.active
    ws_i.title = INSTRUCTIONS
    wrap_top = Alignment(wrap_text=True, vertical="top")
    bold, head_fill, input_fill = Font(bold=True), PatternFill("solid", fgColor="D9E2F3"), PatternFill("solid", fgColor="FFF2CC")
    n1, n2 = len(with_prop), len(no_prop)

    def prep(frame):
        p = D.prepare_bank(frame)
        frame = frame.assign(question_text=p["question_text"].values)
        return frame, [_image_info(r) for _, r in frame.iterrows()]

    wp, wp_img = prep(with_prop)
    npf, np_img = prep(no_prop)

    text = [
        ("Label review sheet: HKDSE ICT Paper 1A", True), ("", False),
        (f"Purpose. {n1 + n2} questions still need a teacher-approved label. Your approval is what lets the tutor use "
         f"them. Please do the 'Label' sheet FIRST, then the 'Review' sheet.", False), ("", False),
        (f"Sheet 'Label' ({n2} questions): the classifier gave no answer. Label them from your own judgement, as in the "
         f"blind step: choose the module (and a sub-topic only if you are sure).", True),
        (f"Sheet 'Review' ({n1} questions): the classifier's module, sub-topic and confidence are shown. For each row "
         f"choose Accept if the proposal is right. Choose Change if it is wrong, then give the final module (and a "
         f"sub-topic only if you are sure). Leave the final columns blank when you Accept. The proposal is a suggestion: "
         f"please judge each question yourself.", True), ("", False),
        ("On both sheets:", True),
        ("1. Time taken (minutes): enter the minutes you spent on that question, to the nearest half minute. "
         "This is used to measure whether the proposals save you time, so please be as accurate as you can.", False),
        ("2. Notes are optional. Mention a second module, or an old-syllabus topic, if relevant.", False),
        ("3. Fill in your name and the date (YYYY-MM-DD) at the top of BOTH sheets. Every row must be completed.", False),
        ("4. Figures and tables are not embedded; the 'Figure / table to check' column says which past paper question to open.", False),
        ("5. Only the yellow cells can be edited. Return the file under the same name with a suffix, for example _done.", False),
        ("", False), ("Not shown on purpose: the official answer key and the cohort percentage.", False)]
    ws_i.column_dimensions["A"].width = 100
    for r, (t, b) in enumerate(text, start=1):
        c = ws_i.cell(row=r, column=1, value=t)
        c.alignment, c.font = wrap_top, Font(bold=b, size=14 if r == 1 else 11)

    def build(title, sheet_name, cols, frame, imgs, with_prediction):
        ws = wb.create_sheet(sheet_name)
        n = len(frame)
        last = FIRST_ROW + n - 1
        widths = ([6, 24, 66, 46, 30, 12, 16, 11, 16, 14, 18, 12, 30, 28] if with_prediction
                  else [6, 24, 70, 52, 34, 13, 18, 12, 36, 28])
        for j, w in enumerate(widths, start=1):
            ws.column_dimensions[chr(64 + j)].width = w
        ws["A1"], ws["A1"].font = title, Font(bold=True, size=13)
        ws["B2"], ws["B3"], ws["B4"], ws["B5"] = "Teacher name", "Date (YYYY-MM-DD)", "Completed", "Modules"
        for a in ("B2", "B3", "B4", "B5"):
            ws[a].font = bold
        for a in (NAME_CELL, DATE_CELL):
            ws[a].fill, ws[a].protection = input_fill, Protection(locked=False)
        ws[DATE_CELL].number_format = "@"
        done_col = "I" if with_prediction else "F"
        ws["C4"] = f'=COUNTA({done_col}{FIRST_ROW}:{done_col}{last})&" of {n}"'
        ws["C5"] = "   |   ".join(f"{k}  {v}" for k, v in D.OFFICIAL_MODULE_NAMES.items())
        ws["C5"].alignment = wrap_top
        ws.row_dimensions[5].height = 48
        for j, h in enumerate(cols, start=1):
            c = ws.cell(row=HEADER_ROW, column=j, value=h)
            c.font, c.fill, c.alignment = bold, head_fill, Alignment(wrap_text=True, vertical="center")
        ws.row_dimensions[HEADER_ROW].height = 48
        input_cols = (9, 10, 11, 12, 13) if with_prediction else (6, 7, 8, 9)
        for i, (_, row) in enumerate(frame.iterrows()):
            r = FIRST_ROW + i
            q, o, im = _question_cell(row), _options_cell(row), imgs[i][0]
            vals = [i + 1, row["qn_id"], q, o, im]
            if with_prediction:
                sub = row["proposed_subtopic"] if isinstance(row["proposed_subtopic"], str) else "(none)"
                vals += [row["proposed_module"], sub, float(row["confidence"])]
            for j, v in enumerate(vals, start=1):
                ws.cell(row=r, column=j, value=v).alignment = wrap_top
            if with_prediction:
                ws.cell(row=r, column=8).number_format = "0%"
            for j in input_cols:
                c = ws.cell(row=r, column=j)
                c.fill, c.protection, c.alignment = input_fill, Protection(locked=False), wrap_top
            if with_prediction:
                chk = (f'=IF(I{r}="","",IF(AND(I{r}="Change",J{r}=""),"choose the final module",'
                       f'IF(AND(I{r}="Accept",OR(J{r}<>"",K{r}<>"")),"leave the final columns blank when accepting",'
                       f'IF(AND(K{r}<>"",J{r}=""),"choose a module first",'
                       f'IF(AND(K{r}<>"",UPPER(J{r})<>UPPER(LEFT(K{r},1))),"sub-topic is not in this module",'
                       f'IF(L{r}="","enter the time taken",""))))))')
                ws.cell(row=r, column=14, value=chk).alignment = wrap_top
            else:
                chk = (f'=IF(AND(G{r}<>"",F{r}=""),"choose a module first",'
                       f'IF(AND(G{r}<>"",UPPER(F{r})<>UPPER(LEFT(G{r},1))),"sub-topic is not in this module",'
                       f'IF(AND(F{r}<>"",H{r}=""),"enter the time taken","")))')
                ws.cell(row=r, column=10, value=chk).alignment = wrap_top
            ws.row_dimensions[r].height = _height((q, widths[2]), (o, widths[3]), (im, widths[4]))
        dv_mod = DataValidation(type="list", formula1='"A,B,C,D,E"', allow_blank=True, showErrorMessage=True,
                                errorTitle="Module", error="Choose A, B, C, D or E.")
        dv_sub = DataValidation(type="list", formula1=f"={REFERENCE}!$A$2:$A${len(tax) + 1}", allow_blank=True,
                                showErrorMessage=True, errorTitle="Sub-topic", error="Choose an id from the Reference sheet, or leave blank.")
        dv_min = DataValidation(type="decimal", operator="between", formula1="0.1", formula2=str(MAX_MINUTES),
                                allow_blank=True, showErrorMessage=True, errorTitle="Minutes",
                                error=f"Enter minutes as a number between 0.1 and {MAX_MINUTES}.")
        for dv in (dv_mod, dv_sub, dv_min):
            ws.add_data_validation(dv)
        if with_prediction:
            dv_dec = DataValidation(type="list", formula1='"Accept,Change"', allow_blank=True, showErrorMessage=True,
                                    errorTitle="Decision", error="Choose Accept or Change.")
            ws.add_data_validation(dv_dec)
            dv_dec.add(f"I{FIRST_ROW}:I{last}")
            dv_mod.add(f"J{FIRST_ROW}:J{last}")
            dv_sub.add(f"K{FIRST_ROW}:K{last}")
            dv_min.add(f"L{FIRST_ROW}:L{last}")
        else:
            dv_mod.add(f"F{FIRST_ROW}:F{last}")
            dv_sub.add(f"G{FIRST_ROW}:G{last}")
            dv_min.add(f"H{FIRST_ROW}:H{last}")
        ws.freeze_panes = f"C{FIRST_ROW}"
        ws.page_setup.orientation, ws.page_setup.paperSize = "landscape", ws.PAPERSIZE_A4
        ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0
        ws.sheet_properties.pageSetUpPr = PageSetupProperties(fitToPage=True)
        ws.print_title_rows = f"{HEADER_ROW}:{HEADER_ROW}"
        ws.protection.sheet = True
        ws.protection.formatColumns = False
        ws.protection.formatRows = False
        return ws

    ws_l = build("Label sheet: no prediction shown (do this one first)", BLIND2_SHEET, LABEL_COLS, npf, np_img, False)
    ws_v = build("Review sheet: classifier proposals shown", REVIEW_SHEET, REVIEW_COLS, wp, wp_img, True)
    ws_r = wb.create_sheet(REFERENCE)
    for j, h in enumerate(["Sub-topic id", "Module", "Module title", "Sub-topic name"], start=1):
        c = ws_r.cell(row=1, column=j, value=h)
        c.font, c.fill = bold, head_fill
    for i, r in enumerate(tax.itertuples(index=False), start=2):
        for j, v in enumerate([r.qn_type_id, r.module, r.module_name, r.subtopic_name], start=1):
            ws_r.cell(row=i, column=j, value=v)
    for col, w in zip("ABCD", (14, 9, 42, 52)):
        ws_r.column_dimensions[col].width = w
    wb.active = wb.index(ws_l)
    Path(xlsx_path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(xlsx_path)

    rows = []
    for part, frame, imgs in (("blind_no_proposal", npf, np_img), ("with_proposal", wp, wp_img)):
        for i, (_, r) in enumerate(frame.iterrows()):
            has = part == "with_proposal"
            rows.append({"part": part, "order": i + 1, "qn_id": r["qn_id"], "year": int(r["year"]),
                         "needs_figure": imgs[i][1],
                         "proposal_id": int(r["proposal_id"]) if has else None,
                         "proposed_module": r["proposed_module"] if has else None,
                         "proposed_subtopic": r["proposed_subtopic"] if has and isinstance(r["proposed_subtopic"], str) else None,
                         "confidence": float(r["confidence"]) if has else None,
                         "seed": seed, "drawn_on": drawn_on})
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest_path, index=False)


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_review_sheet(xlsx_path: str | Path) -> dict:
    """Return {'Label': (teacher, date, rows), 'Review': (teacher, date, rows)}; a blank cell is ''."""
    from openpyxl import load_workbook
    wb = load_workbook(xlsx_path, data_only=True)
    out = {}
    for name in (BLIND2_SHEET, REVIEW_SHEET):
        ws = wb[name]
        teacher = str(ws[NAME_CELL].value or "").strip()
        try:
            date = _read_date(ws[DATE_CELL].value)
        except ValueError as e:
            date = f"!{e}"
        cell = lambda r, c: str(ws.cell(row=r, column=c).value or "").strip()
        rows, r = [], FIRST_ROW
        while ws.cell(row=r, column=2).value not in (None, ""):
            if name == REVIEW_SHEET:
                rows.append({"row": r, "qn_id": cell(r, 2), "decision": cell(r, 9).capitalize(),
                             "module": cell(r, 10).upper(), "subtopic": cell(r, 11),
                             "minutes": _num(ws.cell(row=r, column=12).value), "notes": cell(r, 13)})
            else:
                rows.append({"row": r, "qn_id": cell(r, 2), "module": cell(r, 6).upper(), "subtopic": cell(r, 7),
                             "minutes": _num(ws.cell(row=r, column=8).value), "notes": cell(r, 9)})
            r += 1
        out[name] = (teacher, date, rows)
    return out


def import_review_sheet(xlsx_path: str | Path, manifest_path: str | Path, con, df: pd.DataFrame,
                        tax: pd.DataFrame) -> dict:
    """Check the returned sheet and store its labels as 'teacher_approved', with one label_review row each.

    All or nothing. Refuses the sheet if a name or date is missing or the two sheets disagree; if the questions
    differ from the manifest; if a Review row has no Accept/Change; if Accept has final columns filled or
    Change leaves the proposal unchanged or lacks a module; if a module is invalid, a sub-topic is unknown or in
    another module; if minutes are missing or outside 0.1-120; or if these questions already have a teacher label."""
    S.ensure_label_review(con)
    man = pd.read_csv(manifest_path)
    sheets = read_review_sheet(xlsx_path)
    problems = []
    names = {sheets[k][0] for k in sheets}
    dates = {sheets[k][1] for k in sheets}
    if "" in names:
        problems.append(f"teacher name is empty on a sheet ({NAME_CELL})")
    if len(names - {""}) > 1 or len(dates) > 1:
        problems.append(f"the two sheets have different names {sorted(names)} or dates {sorted(dates)}")
    for d in dates:
        if d.startswith("!"):
            problems.append(d[1:])
    sub_module = dict(zip(tax["qn_type_id"], tax["module"]))
    have_prop = {q: (i, m) for i, q, m in con.execute(
        "SELECT id, qn_id, predicted_module FROM label_proposal WHERE label_source='classifier'")}
    items = []
    for sheet_name, part in ((BLIND2_SHEET, "blind_no_proposal"), (REVIEW_SHEET, "with_proposal")):
        teacher, date, rows = sheets[sheet_name]
        m = man[man["part"] == part]
        ids = [r["qn_id"] for r in rows]
        if sorted(ids) != sorted(m["qn_id"]):
            problems.append(f"{sheet_name}: questions differ from the manifest: missing {sorted(set(m['qn_id']) - set(ids))[:5]}, "
                            f"unexpected {sorted(set(ids) - set(m['qn_id']))[:5]}, duplicates {len(ids) - len(set(ids))}")
        mrow = m.set_index("qn_id")
        for r in rows:
            tag = f"{sheet_name} row {r['row']} ({r['qn_id']})"
            mod, sub, mins = r["module"], r["subtopic"], r["minutes"]
            if mins is None or not 0.1 <= mins <= MAX_MINUTES:
                problems.append(f"{tag}: time taken must be a number of minutes between 0.1 and {MAX_MINUTES}")
            if mod and mod not in D.MODULES:
                problems.append(f"{tag}: module {mod!r} is not A-E")
            elif sub and mod and sub_module.get(sub) != mod:
                problems.append(f"{tag}: sub-topic {sub!r} is unknown or not in module {mod}")
            elif sub and not mod:
                problems.append(f"{tag}: sub-topic given without a module")
            if r["qn_id"] not in mrow.index:
                continue
            pm = mrow.loc[r["qn_id"]]
            if part == "blind_no_proposal":
                if not mod:
                    problems.append(f"{tag}: module is empty")
                final, final_sub, decision, pid = mod, sub, "label", None
            else:
                pid = int(pm["proposal_id"])
                stored = have_prop.get(r["qn_id"])
                if stored is None or stored[0] != pid or stored[1] != pm["proposed_module"]:
                    problems.append(f"{tag}: the classifier proposal in the store no longer matches the manifest")
                dec = r["decision"]
                if dec not in ("Accept", "Change"):
                    problems.append(f"{tag}: decision must be Accept or Change")
                    continue
                psub = pm["proposed_subtopic"] if isinstance(pm["proposed_subtopic"], str) else ""
                if dec == "Accept":
                    if mod or sub:
                        problems.append(f"{tag}: Accept, but a final module or sub-topic is filled in")
                    final, final_sub, decision = pm["proposed_module"], psub, "accept"
                else:
                    if not mod:
                        problems.append(f"{tag}: Change, but no final module")
                    elif mod == pm["proposed_module"] and sub == psub:
                        problems.append(f"{tag}: Change, but the final label equals the proposal (use Accept)")
                    final, final_sub, decision = mod, sub, "change"
            items.append({"qn_id": r["qn_id"], "module": final, "subtopic": final_sub or None, "part": part,
                          "proposal_id": pid, "decision": decision, "reviewer": teacher, "decided_at": date,
                          "seconds_spent": round((mins or 0) * 60), "note": r["notes"]})
    done = {x[0] for x in con.execute("SELECT qn_id FROM label_proposal WHERE label_source='teacher_approved'")}
    dup = sorted(done & {i["qn_id"] for i in items})
    if dup:
        problems.append(f"teacher-approved labels already stored for {dup[:5]}")
    if problems:
        raise ValueError("sheet not imported:\n  " + "\n  ".join(problems))
    S.record_label_reviews(con, df, items)
    out = pd.DataFrame(items)
    return {"imported": len(items), "accepted": int((out.decision == "accept").sum()),
            "changed": int((out.decision == "change").sum()), "labelled_blind": int((out.decision == "label").sum())}


# ---------------------------------------------------------------------- CLI
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Blind label sheet: export for the teacher, import when returned.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--n", type=int, default=BLIND_N_DEFAULT)
    e.add_argument("--seed", type=int, default=BLIND_SEED_DEFAULT)
    e.add_argument("--out", default="outputs/blind_label_sheet.xlsx")
    e.add_argument("--manifest", default="outputs/blind_sample_manifest.csv")
    re_ = sub.add_parser("review-export")
    re_.add_argument("--db", default=str(S.DEFAULT_DB))
    re_.add_argument("--seed", type=int, default=BLIND_SEED_DEFAULT)
    re_.add_argument("--out", default="outputs/label_review_sheet.xlsx")
    re_.add_argument("--manifest", default="outputs/label_review_manifest.csv")
    ri = sub.add_parser("review-import")
    ri.add_argument("sheet")
    ri.add_argument("--db", default=str(S.DEFAULT_DB))
    ri.add_argument("--manifest", default="outputs/label_review_manifest.csv")
    i = sub.add_parser("import")
    i.add_argument("sheet")
    i.add_argument("--manifest", default="outputs/blind_sample_manifest.csv")
    i.add_argument("--db", default=str(S.DEFAULT_DB))
    for p in (e, i, re_, ri):
        p.add_argument("--bank", default="data/final_dataset.csv")
        p.add_argument("--taxonomy", default="data/taxonomy.csv")
    a = ap.parse_args(argv)
    tax = D.load_taxonomy(a.taxonomy)
    df = D.derive(D.load_bank(a.bank, tax))
    if a.cmd == "review-export":
        con = S.init_db(a.db, tax)
        w, b = draw_review_set(df, con, a.seed)
        export_review_sheet(df, tax, w, b, a.out, a.manifest, seed=a.seed)
        print(f"{len(w)} with a proposal, {len(b)} without -> {a.out}; manifest {a.manifest}")
    elif a.cmd == "review-import":
        con = S.init_db(a.db, tax)
        print(import_review_sheet(a.sheet, a.manifest, con, df, tax))
    elif a.cmd == "export":
        s = draw_blind_sample(df, a.n, a.seed)
        export_blind_sheet(df, tax, s, a.out, a.manifest, seed=a.seed)
        print(f"{len(s)} questions -> {a.out} ({int((s.year == D.NEW_PAPER_YEAR).sum())} from {D.NEW_PAPER_YEAR}); manifest {a.manifest}")
    else:
        con = S.init_db(a.db, tax)
        print(f"imported {import_blind_sheet(a.sheet, a.manifest, con, df, tax)} blind labels into {a.db}")


if __name__ == "__main__":
    main()
