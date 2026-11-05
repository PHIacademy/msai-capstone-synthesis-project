"""Teacher review of generated candidates (Section 4.8 of the notebook).

Place at src/candidate_review.py.

Queue: candidates whose gate verdict is pass or flag and that have no decision yet (reject stays out of the
teacher's queue). The sheet hides the generator, the generator's key and the gate verdict; a flagged item shows
only the plain reason. The teacher solves each item, so the key in the store is the teacher's own.

Answer-position shuffle: the options of every item are shuffled with a fixed seed before export, so the
position of the correct answer does not follow the generator (Gemma favoured C, Haiku favoured A). Items whose
options refer to each other by position or order ('All of the above', 'Both A and B', '(1) and (2) only') are
NOT shuffled and are marked so in the manifest. The permutation is stored in the manifest, and the admitted item
is saved in the order the teacher saw and approved.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
import re
from pathlib import Path

import pandas as pd

from src import data as D
from src import store as S
from src.review import (FIRST_ROW, HEADER_ROW, MAX_MINUTES, NAME_CELL, DATE_CELL, _read_date, _num)

SHEET, INSTRUCTIONS = "Candidates", "Instructions"
DECISIONS = {"Approve": "approve", "Edit": "edit_approve", "Reject": "reject"}
REJECT_CODES = {      # the rubric codes of the bake-off (Section 3.6)
    "F": "format: four options, no leftover text, no figure reference",
    "K": "key: exactly one defensible correct answer",
    "D": "distractors: plausible, distinct, none correct or absurd, key not given away",
    "C": "curriculum fit: belongs to the requested module and the HKDSE ICT curriculum",
    "O": "originality: too close to a real question",
    "X": "other (explain in Notes)",
}
COLUMNS = ["No.", "Item ID", "Topic", "Question", "Options", "Automatic check note", "Your answer (A-D)",
           "Decision", "Edited question", "Edited A", "Edited B", "Edited C", "Edited D",
           "Reject reason 1", "Reject reason 2", "Time taken (min)", "Notes", "Check"]
(C_ID, C_KEY, C_DEC, C_EQ, C_EA, C_R1, C_R2, C_MIN, C_NOTE) = 2, 7, 8, 9, 10, 14, 15, 16, 17
_ORDER_DEPENDENT = re.compile(
    r"(?i)\b(all|none|both|neither|any)\s+of\s+(the\s+)?(above|these|following|them)\b|\babove\b"
    r"|\b[A-D]\s*(,|and|&|or)\s*[A-D]\b|\(\s*(i{1,3}|iv|[1-4])\s*\)|^\s*(i{1,3}|iv|[1-4])\s*(,|and|&|only)\b")


def option_text_hash(c: dict) -> str:
    return hashlib.sha1(json.dumps([c["stem"]] + [c.get(f"opt_{k.lower()}") for k in D.LETTERS]).encode()).hexdigest()[:12]


def shuffle_options(cand_id: str, options: dict, key: str | None, seed: int = 42) -> dict:
    """Deterministic shuffle of one item. Returns options (new order), key (new letter or None), perm (old letters
    in the new order, e.g. 'CADB'), shuffled (False when the item is order-dependent or incomplete)."""
    keep = {"options": dict(options), "key": key if key in D.LETTERS else None, "perm": "ABCD", "shuffled": False}
    if any(not str(options.get(k) or "").strip() for k in D.LETTERS):
        return keep
    if any(_ORDER_DEPENDENT.search(str(options[k])) for k in D.LETTERS):
        return keep
    rng, order = random.Random(f"{seed}:{cand_id}"), list(D.LETTERS)
    while order == list(D.LETTERS):       # an identity draw is drawn again, so every shuffled item moves
        rng.shuffle(order)
    new = {n: options[o] for n, o in zip(D.LETTERS, order)}
    nk = D.LETTERS[order.index(key)] if key in D.LETTERS else None
    return {"options": new, "key": nk, "perm": "".join(order), "shuffled": True}


def candidate_queue(con) -> pd.DataFrame:
    """Candidates with gate verdict pass or flag and no teacher decision yet, in cand_id order."""
    q = pd.read_sql_query(
        "SELECT c.*, g.verdict, g.reasons_json FROM candidate c JOIN gate_result g USING(cand_id) "
        "WHERE g.verdict IN ('pass','flag') AND c.cand_id NOT IN (SELECT cand_id FROM review_decision) "
        "ORDER BY c.cand_id", con)
    return q


def _check_note(row) -> str:
    if row["verdict"] != "flag":
        return ""
    try:
        r = json.loads(row["reasons_json"] or "[]")
    except ValueError:
        r = []
    return "; ".join(r if isinstance(r, list) else [str(r)])


def export_candidate_sheet(con, tax: pd.DataFrame, xlsx_path, manifest_path, seed: int = 42,
                           drawn_on: str | None = None) -> pd.DataFrame:
    """Write the workbook and the manifest for the current queue. Item order is a fixed-seed shuffle, so
    backends are mixed. Returns the manifest."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill, Protection
    from openpyxl.worksheet.datavalidation import DataValidation

    q = candidate_queue(con)
    if q.empty:
        raise ValueError("no candidates are waiting for review")
    drawn_on = drawn_on or dt.date.today().isoformat()
    order = list(range(len(q)))
    random.Random(f"{seed}:order").shuffle(order)
    q = q.iloc[order].reset_index(drop=True)
    names = {r.qn_type_id: r.subtopic_name for r in tax.itertuples(index=False)}
    mods = D.OFFICIAL_MODULE_NAMES

    wb = Workbook()
    wi = wb.active
    wi.title = INSTRUCTIONS
    wi.column_dimensions["A"].width = 100
    codes = [f"{k} = {v}" for k, v in REJECT_CODES.items()]
    text = ["Candidate review sheet: HKDSE ICT Paper 1A", "",
            f"Purpose. {len(q)} new multiple-choice questions were written by AI. Your decision decides whether a question "
            "may be used (as provisional, i.e. not an official past paper item).",
            "For each row: 1) answer the question yourself and enter A-D in 'Your answer'; 2) choose Approve, Edit or Reject.",
            "Approve: the question is fine as written. Edit: fix the wording or an option in the 'Edited' columns (leave the "
            "others blank), then your answer refers to the edited version. Reject: choose one or two reasons.",
            "Enter the minutes you spent on the row (nearest half minute). Fill in your name and the date (YYYY-MM-DD).",
            "The writer of each question and the generator's own answer are not shown, on purpose. The order of the options may "
            "differ from what the writer produced.", "", "Reject reasons:"] + codes
    for r, t in enumerate(text, 1):
        c = wi.cell(row=r, column=1, value=t)
        c.alignment, c.font = Alignment(wrap_text=True, vertical="top"), Font(bold=r == 1, size=14 if r == 1 else 11)

    ws = wb.create_sheet(SHEET)
    wrap = Alignment(wrap_text=True, vertical="top")
    bold, head, inp = Font(bold=True), PatternFill("solid", fgColor="D9E2F3"), PatternFill("solid", fgColor="FFF2CC")
    last = FIRST_ROW + len(q) - 1
    widths = [6, 10, 30, 62, 50, 26, 12, 12, 40, 22, 22, 22, 22, 16, 16, 12, 28, 30]
    for j, w in enumerate(widths, 1):
        ws.column_dimensions[chr(64 + j)].width = w
    ws["A1"], ws["A1"].font = "Candidate review", Font(bold=True, size=13)
    for a, v in (("B2", "Teacher name"), ("B3", "Date (YYYY-MM-DD)"), ("B4", "Completed")):
        ws[a], ws[a].font = v, bold
    for a in (NAME_CELL, DATE_CELL):
        ws[a].fill, ws[a].protection = inp, Protection(locked=False)
    ws[DATE_CELL].number_format = "@"
    ws["C4"] = f'=COUNTA(H{FIRST_ROW}:H{last})&" of {len(q)}"'
    for j, h in enumerate(COLUMNS, 1):
        c = ws.cell(row=HEADER_ROW, column=j, value=h)
        c.font, c.fill, c.alignment = bold, head, Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[HEADER_ROW].height = 36

    rows = []
    for i, c in q.iterrows():
        cand = c.to_dict()
        sid = f"R{i + 1:03d}"
        opts = {k: cand[f"opt_{k.lower()}"] for k in D.LETTERS}
        sh = shuffle_options(cand["cand_id"], opts, cand["gen_key"], seed)
        topic = f"{cand['target_module']}. {mods.get(cand['target_module'], '')}"
        if cand["target_subtopic"]:
            topic += f" / {names.get(cand['target_subtopic'], cand['target_subtopic'])}"
        otxt = "\n".join(f"{k}. {sh['options'][k]}" for k in D.LETTERS)
        r = FIRST_ROW + i
        for j, v in enumerate([i + 1, sid, topic, cand["stem"], otxt, _check_note(cand)], 1):
            ws.cell(row=r, column=j, value=v).alignment = wrap
        for j in range(7, 18):
            cc = ws.cell(row=r, column=j)
            cc.fill, cc.protection, cc.alignment = inp, Protection(locked=False), wrap
        ws.cell(row=r, column=18, value=(
            f'=IF(H{r}="","",IF(AND(H{r}<>"Reject",G{r}=""),"enter your answer",'
            f'IF(AND(H{r}="Reject",N{r}=""),"choose a reject reason",'
            f'IF(AND(H{r}="Edit",COUNTA(I{r}:M{r})=0),"fill an edited column",'
            f'IF(AND(H{r}="Approve",COUNTA(I{r}:O{r})>0),"clear edited and reject columns when approving",'
            f'IF(P{r}="","enter the time taken",""))))))')).alignment = wrap
        lines = max(len(cand["stem"]) // 55 + 1, sum(len(str(v)) // 45 + 1 for v in sh["options"].values()))
        ws.row_dimensions[r].height = min(15 * lines + 6, 400)
        rows.append({"sheet_id": sid, "order": i + 1, "cand_id": cand["cand_id"], "backend": cand["backend"],
                     "verdict": cand["verdict"], "shuffled": sh["shuffled"], "perm": sh["perm"],
                     "gen_key_original": cand["gen_key"] if cand["gen_key"] in D.LETTERS else None,
                     "gen_key_shown": sh["key"], "text_hash": option_text_hash(cand), "seed": seed,
                     "drawn_on": drawn_on})
    dvs = [(f'"{",".join(D.LETTERS)}"', "G", "Answer", "Choose A, B, C or D."),
           ('"Approve,Edit,Reject"', "H", "Decision", "Choose Approve, Edit or Reject."),
           (f'"{",".join(REJECT_CODES)}"', "N", "Reason", "Choose a reason from the list."),
           (f'"{",".join(REJECT_CODES)}"', "O", "Reason", "Choose a reason from the list.")]
    for f1, col, t, e in dvs:
        dv = DataValidation(type="list", formula1=f1, allow_blank=True, showErrorMessage=True, errorTitle=t, error=e)
        ws.add_data_validation(dv)
        dv.add(f"{col}{FIRST_ROW}:{col}{last}")
    dm = DataValidation(type="decimal", operator="between", formula1="0.1", formula2=str(MAX_MINUTES), allow_blank=True,
                        showErrorMessage=True, errorTitle="Minutes", error=f"Enter a number between 0.1 and {MAX_MINUTES}.")
    ws.add_data_validation(dm)
    dm.add(f"P{FIRST_ROW}:P{last}")
    ws.freeze_panes = f"D{FIRST_ROW}"
    ws.protection.sheet = True
    ws.protection.formatColumns = ws.protection.formatRows = False
    wb.active = wb.index(ws)
    Path(xlsx_path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(xlsx_path)
    man = pd.DataFrame(rows)
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    man.to_csv(manifest_path, index=False)
    return man


def read_candidate_sheet(xlsx_path) -> tuple:
    from openpyxl import load_workbook
    ws = load_workbook(xlsx_path, data_only=True)[SHEET]
    teacher = str(ws[NAME_CELL].value or "").strip()
    try:
        date = _read_date(ws[DATE_CELL].value)
    except ValueError as e:
        date = f"!{e}"
    cell = lambda r, c: str(ws.cell(row=r, column=c).value or "").strip()
    rows, r = [], FIRST_ROW
    while ws.cell(row=r, column=C_ID).value not in (None, ""):
        rows.append({"row": r, "sheet_id": cell(r, C_ID), "key": cell(r, C_KEY).upper(), "decision": cell(r, C_DEC).capitalize(),
                     "edit_stem": cell(r, C_EQ), "edit_opts": {k: cell(r, C_EA + n) for n, k in enumerate(D.LETTERS)},
                     "codes": [x for x in (cell(r, C_R1).upper(), cell(r, C_R2).upper()) if x],
                     "minutes": _num(ws.cell(row=r, column=C_MIN).value), "notes": cell(r, C_NOTE)})
        r += 1
    return teacher, date, rows


def import_candidate_sheet(xlsx_path, manifest_path, con) -> dict:
    """Check the returned sheet against the manifest and the store, then store every decision in one transaction.
    Nothing is written if any row has a problem."""
    teacher, date, rows = read_candidate_sheet(xlsx_path)
    man = pd.read_csv(manifest_path, dtype={"perm": str}).set_index("sheet_id")
    problems = []
    if not teacher:
        problems.append("teacher name is empty")
    if date.startswith("!"):
        problems.append(date[1:])
    got = [r["sheet_id"] for r in rows]
    if sorted(got) != sorted(man.index):
        problems.append(f"the rows do not match the manifest (missing {sorted(set(man.index) - set(got))[:5]}, "
                        f"extra {sorted(set(got) - set(man.index))[:5]}, repeated ids: {len(got) != len(set(got))})")
    cur = con.execute("SELECT cand_id,stem,opt_a,opt_b,opt_c,opt_d,gen_key,backend FROM candidate")
    cols = [d[0] for d in cur.description]
    cands = {r[0]: dict(zip(cols, r)) for r in cur.fetchall()}
    done = S.reviewed_candidate_ids(con)
    items = []
    for r in rows:
        tag = f"row {r['row']} ({r['sheet_id']})"
        if r["sheet_id"] not in man.index:
            continue
        m = man.loc[r["sheet_id"]]
        c = cands.get(m["cand_id"])
        if c is None:
            problems.append(f"{tag}: candidate not in the store")
            continue
        if m["cand_id"] in done:
            problems.append(f"{tag}: already has a decision")
            continue
        if option_text_hash(c) != m["text_hash"]:
            problems.append(f"{tag}: the stored candidate changed since the sheet was made")
            continue
        dec = DECISIONS.get(r["decision"])
        if dec is None:
            problems.append(f"{tag}: decision must be Approve, Edit or Reject")
            continue
        if not r["minutes"] or not 0 < r["minutes"] <= MAX_MINUTES:
            problems.append(f"{tag}: time taken must be between 0.1 and {MAX_MINUTES} minutes")
            continue
        edited = bool(r["edit_stem"]) or any(r["edit_opts"].values())
        bad_codes = [x for x in r["codes"] if x not in REJECT_CODES]
        if bad_codes:
            problems.append(f"{tag}: unknown reject reason {bad_codes}")
        if dec == "reject":
            if not r["codes"]:
                problems.append(f"{tag}: Reject needs a reason")
            if edited:
                problems.append(f"{tag}: Reject, but edited columns are filled in")
        else:
            if r["key"] not in D.LETTERS:
                problems.append(f"{tag}: enter your answer (A-D)")
            if r["codes"]:
                problems.append(f"{tag}: reject reasons given without Reject")
            if dec == "approve" and edited:
                problems.append(f"{tag}: Approve, but edited columns are filled in (choose Edit)")
            if dec == "edit_approve" and not edited:
                problems.append(f"{tag}: Edit, but no edited column is filled in")
        perm = m["perm"] if isinstance(m["perm"], str) else "ABCD"
        shown = {n: c[f"opt_{o.lower()}"] for n, o in zip(D.LETTERS, perm)}
        stem = r["edit_stem"] or c["stem"]
        opts = {k: r["edit_opts"][k] or shown[k] for k in D.LETTERS}
        if dec != "reject" and any(not str(v or "").strip() for v in opts.values()):
            problems.append(f"{tag}: an option is empty")
        prov = (f"AI-generated ({c['backend']}), teacher-reviewed; candidate {m['cand_id']}; "
                f"options {'shuffled ' + perm if m['shuffled'] else 'not shuffled'}")
        items.append({"cand_id": m["cand_id"], "decision": dec, "reject_codes": ";".join(r["codes"]) or None,
                      "confirmed_key": r["key"] if r["key"] in D.LETTERS else None, "reviewer": teacher,
                      "decided_at": date, "seconds_spent": round((r["minutes"] or 0) * 60), "stem": stem,
                      "options": opts, "provenance": prov + ("; edited" if dec == "edit_approve" else ""),
                      "agrees": (r["key"] == m["gen_key_shown"]) if isinstance(m["gen_key_shown"], str) else None})
    if problems:
        raise ValueError("sheet not imported:\n  " + "\n  ".join(problems))
    ids = S.record_candidate_reviews(con, items)
    out = pd.DataFrame(items)
    ag = out[out["agrees"].notna() & (out["decision"] != "reject")]["agrees"]
    return {"imported": len(items), "approved": int((out.decision == "approve").sum()),
            "edited": int((out.decision == "edit_approve").sum()), "rejected": int((out.decision == "reject").sum()),
            "admitted": len(ids), "key_agreement": f"{int(ag.sum())} of {len(ag)}" if len(ag) else "no generator keys to compare"}
