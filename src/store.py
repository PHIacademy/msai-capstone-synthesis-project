"""Governed state for the integrated system: SQLite schema, effective bank, access modes.

Place at src/store.py (Section 3.3 of the notebook).

Decision recorded in the notebook: the official bank stays an immutable CSV; everything that
changes (labels, candidates, gate results, review decisions, admitted items) lives
in one SQLite file, so the governance rules are enforced by constraints rather than by
convention.

What this module provides
- SCHEMA: the Section 3.3 schema, verbatim.
- init_db / connect_rw / connect_ro: create the file; open it for the staff side (writer,
  foreign keys ON) or for the student tutor (mode=ro, query_only).
- add_label: record a label proposal. Official labels are never overwritten.
- approved_labels / build_effective_bank: join the CSV with approved labels and admitted
  items into what the tutor sees. Built once per session; state_fingerprint says when the
  staff side has committed something new.

Also here: the label review log (label_review, created when first needed) and record_label_reviews,
and the insertion of generated candidates (add_candidates, add_candidates_from_jsonl).
Not here yet (later Task 4 steps): review decisions and admitting items. build_effective_bank already
reads admitted_item; gate.write_gate_result and gate.gate_pending store gate results.
"""
from __future__ import annotations

import json
import sqlite3
import warnings
from pathlib import Path

import pandas as pd

from src.data import LETTERS, MODULES, NEW_PAPER_YEAR, format_options

DEFAULT_DB = Path("data/state.sqlite")
APPROVED_SOURCES = ("teacher_blind", "teacher_approved")   # a classifier row is never an approval
PROVENANCE_LABEL = "AI-drafted, teacher-approved, not an HKEAA question"

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE taxonomy (
    qn_type_id TEXT PRIMARY KEY,            -- sub-topic id, e.g. d_cp_02
    module TEXT NOT NULL CHECK (module IN ('A','B','C','D','E')),
    module_name TEXT NOT NULL,              -- official curriculum title
    subtopic_name TEXT NOT NULL,
    UNIQUE (qn_type_id, module)             -- lets other tables check that a sub-topic belongs to its module
);

CREATE TABLE label_proposal (
    id INTEGER PRIMARY KEY,
    qn_id TEXT NOT NULL,
    predicted_module TEXT NOT NULL CHECK (predicted_module IN ('A','B','C','D','E')),  -- one primary module
    predicted_subtopic TEXT,                -- NULL unless confident and well supported
    confidence REAL,
    label_source TEXT NOT NULL CHECK (label_source IN ('classifier','teacher_blind','teacher_approved')),
    approved_by TEXT,
    approved_at TEXT,
    CHECK (label_source = 'classifier' OR (approved_by IS NOT NULL AND approved_at IS NOT NULL)),
    FOREIGN KEY (predicted_subtopic, predicted_module) REFERENCES taxonomy(qn_type_id, module)
);

CREATE TABLE candidate (
    cand_id TEXT PRIMARY KEY,
    backend TEXT NOT NULL,
    target_module TEXT NOT NULL CHECK (target_module IN ('A','B','C','D','E')),
    target_subtopic TEXT,
    target_band TEXT,
    stem TEXT NOT NULL,
    opt_a TEXT, opt_b TEXT, opt_c TEXT, opt_d TEXT,
    gen_key TEXT CHECK (gen_key IN ('A','B','C','D')),
    valid_format INTEGER NOT NULL,
    parse_error TEXT,
    raw TEXT,
    FOREIGN KEY (target_subtopic, target_module) REFERENCES taxonomy(qn_type_id, module)
);

CREATE TABLE gate_result (
    cand_id TEXT PRIMARY KEY REFERENCES candidate(cand_id),
    checks_json TEXT NOT NULL,
    solver_answer TEXT,                     -- unused: reserved for the optional blind-solver extension
    solver_agrees_with_key INTEGER,         -- unused: same
    verdict TEXT NOT NULL CHECK (verdict IN ('reject','flag','pass')),
    reasons_json TEXT
);

CREATE TABLE review_decision (
    id INTEGER PRIMARY KEY,
    cand_id TEXT NOT NULL REFERENCES candidate(cand_id),
    decision TEXT NOT NULL CHECK (decision IN ('approve','edit_approve','reject')),
    reject_codes TEXT,
    confirmed_key TEXT CHECK (confirmed_key IN ('A','B','C','D')),
    reviewer TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    seconds_spent INTEGER,
    CHECK (decision = 'reject' OR confirmed_key IS NOT NULL),
    CHECK (decision != 'reject' OR reject_codes IS NOT NULL)
);

CREATE TABLE admitted_item (
    item_id TEXT PRIMARY KEY,
    cand_id TEXT NOT NULL UNIQUE REFERENCES candidate(cand_id),
    review_id INTEGER NOT NULL REFERENCES review_decision(id),
    stem TEXT NOT NULL,
    options_json TEXT NOT NULL,
    key TEXT NOT NULL CHECK (key IN ('A','B','C','D')),
    module TEXT NOT NULL CHECK (module IN ('A','B','C','D','E')),
    subtopic TEXT,
    cluster INTEGER NOT NULL CHECK (cluster = 0),
    provisional INTEGER NOT NULL CHECK (provisional = 1),
    provenance TEXT NOT NULL,
    FOREIGN KEY (subtopic, module) REFERENCES taxonomy(qn_type_id, module)
);
"""

EFFECTIVE_COLUMNS = [
    "item_id", "source", "qn_id", "year", "question_text", "answer_options", "correct_answer",
    "cohort_pct", "cluster", "cluster_desc", "qn_image_filename", "qn_image_type",
    "ans_image_filename", "tags", "primary_module", "label_level", "label_source", "unit",
    "provisional", "provenance",
]


# ---------------------------------------------------------------- connections
def connect_rw(path: str | Path) -> sqlite3.Connection:
    """Staff-side writer. Foreign keys are a per-connection setting in SQLite, so they are
    switched on here; without this a reopened file would silently stop enforcing them."""
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA foreign_keys = ON")
    return con


def connect_ro(path: str | Path) -> sqlite3.Connection:
    """Student-side reader (decision: the tutor has read-only access to the admitted bank).
    mode=ro stops writes at the file level; query_only is a second guard on the connection."""
    con = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    con.execute("PRAGMA query_only = ON")
    return con


def load_taxonomy_table(con: sqlite3.Connection, tax: pd.DataFrame) -> None:
    cols = ["qn_type_id", "module", "module_name", "subtopic_name"]
    with con:
        con.executemany("INSERT INTO taxonomy VALUES (?,?,?,?)",
                        tax[cols].itertuples(index=False, name=None))


def check_taxonomy(con: sqlite3.Connection, tax: pd.DataFrame) -> None:
    """The taxonomy is static: a change is a new versioned file, never an edit in place."""
    cols = ["qn_type_id", "module", "module_name", "subtopic_name"]
    stored = con.execute("SELECT qn_type_id, module, module_name, subtopic_name FROM taxonomy "
                         "ORDER BY qn_type_id").fetchall()
    wanted = sorted(tax[cols].itertuples(index=False, name=None))
    if stored != wanted:
        raise ValueError("the taxonomy in the database differs from taxonomy.csv; "
                         "a taxonomy change is a new versioned database, not an in-place edit")


def init_db(path: str | Path, tax: pd.DataFrame) -> sqlite3.Connection:
    """Create the governed-state file (schema plus taxonomy) and return a writer connection.

    Safe to call again on an existing file: it checks the taxonomy and changes nothing else."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = connect_rw(path)
    exists = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='taxonomy'").fetchone()
    if exists:
        check_taxonomy(con, tax)
        return con
    con.executescript(SCHEMA)
    load_taxonomy_table(con, tax)
    return con


# --------------------------------------------------------------------- labels
def _insert_label(con: sqlite3.Connection, df: pd.DataFrame, qn_id: str, module: str,
                  subtopic: str | None, label_source: str, confidence: float | None,
                  approved_by: str | None, approved_at: str | None) -> int:
    """Check and insert one label proposal without committing (the caller owns the transaction)."""
    row = df[df["qn_id"] == qn_id]
    if row.empty:
        raise ValueError(f"{qn_id}: not in the bank")
    r = row.iloc[0]
    if r["withdrawn"]:
        raise ValueError(f"{qn_id}: withdrawn question, no label is recorded")
    if not r["unlabelled"]:
        raise ValueError(f"{qn_id}: already has an official label; official labels are never overwritten")
    module = str(module).strip().upper()
    if module not in MODULES:
        raise ValueError(f"{qn_id}: module must be one of {MODULES}, got {module!r}")
    cur = con.execute(
        "INSERT INTO label_proposal(qn_id,predicted_module,predicted_subtopic,confidence,"
        "label_source,approved_by,approved_at) VALUES (?,?,?,?,?,?,?)",
        (qn_id, module, subtopic or None, confidence, label_source, approved_by, approved_at))
    return cur.lastrowid


def add_label(con: sqlite3.Connection, df: pd.DataFrame, qn_id: str, module: str,
              subtopic: str | None = None, *, label_source: str, confidence: float | None = None,
              approved_by: str | None = None, approved_at: str | None = None) -> int:
    """Record one label proposal (a classifier prediction or a teacher label).

    df is the derived bank (data.derive). An official label is never overwritten: only a
    question that is unclassified and not withdrawn can receive a proposal. The database
    still enforces the rest (module in A-E, sub-topic belongs to the module, a teacher label
    needs an approver and a time stamp)."""
    with con:
        return _insert_label(con, df, qn_id, module, subtopic, label_source, confidence,
                             approved_by, approved_at)


def add_labels(con: sqlite3.Connection, df: pd.DataFrame, labels: list) -> list:
    """Record several label proposals in ONE transaction: all of them or none.

    labels is a list of dicts with the keyword names of add_label (qn_id, module, subtopic,
    label_source, confidence, approved_by, approved_at). Used when a teacher returns a sheet."""
    ids = []
    with con:
        for L in labels:
            ids.append(_insert_label(
                con, df, L["qn_id"], L["module"], L.get("subtopic"), L["label_source"],
                L.get("confidence"), L.get("approved_by"), L.get("approved_at")))
    return ids


def approved_labels(con: sqlite3.Connection) -> pd.DataFrame:
    """Teacher labels only (a classifier row is not an approval). If a question has several,
    the most recent one wins."""
    q = ("SELECT id, qn_id, predicted_module AS module, predicted_subtopic AS subtopic, label_source, "
         "approved_by, approved_at FROM label_proposal WHERE label_source IN (?,?) ORDER BY id")
    out = pd.read_sql_query(q, con, params=APPROVED_SOURCES)
    return out.drop_duplicates("qn_id", keep="last").reset_index(drop=True)


# ---------------------------------------------------------- label review log
LABEL_REVIEW_DDL = """
CREATE TABLE IF NOT EXISTS label_review (
    id INTEGER PRIMARY KEY,
    qn_id TEXT NOT NULL UNIQUE,
    part TEXT NOT NULL CHECK (part IN ('with_proposal','blind_no_proposal')),
    proposal_id INTEGER REFERENCES label_proposal(id),     -- the classifier row the teacher was shown
    label_id INTEGER NOT NULL REFERENCES label_proposal(id),  -- the teacher_approved row that resulted
    decision TEXT NOT NULL CHECK (decision IN ('accept','change','label')),
    reviewer TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    seconds_spent INTEGER NOT NULL CHECK (seconds_spent > 0),
    note TEXT,
    CHECK (part = 'blind_no_proposal' OR proposal_id IS NOT NULL)
);
"""


def ensure_label_review(con: sqlite3.Connection) -> None:
    """Create the label_review table if the store was built before it existed (safe to repeat).
    It records, for each question the teacher approved with a classifier proposal on view, what was shown,
    whether it was accepted or changed, and the time taken (Task 5 measures the classifier's help from it)."""
    with con:
        con.executescript(LABEL_REVIEW_DDL)


def record_label_reviews(con: sqlite3.Connection, df: pd.DataFrame, items: list) -> list:
    """Store the teacher's approved labels AND their review log rows in ONE transaction (all or none).

    Each item: qn_id, module, subtopic, part, proposal_id, decision, reviewer, decided_at, seconds_spent, note."""
    ensure_label_review(con)
    ids = []
    with con:
        for it in items:
            lid = _insert_label(con, df, it["qn_id"], it["module"], it.get("subtopic"), "teacher_approved",
                                None, it["reviewer"], it["decided_at"])
            con.execute(
                "INSERT INTO label_review(qn_id,part,proposal_id,label_id,decision,reviewer,decided_at,"
                "seconds_spent,note) VALUES (?,?,?,?,?,?,?,?,?)",
                (it["qn_id"], it["part"], it.get("proposal_id"), lid, it["decision"], it["reviewer"],
                 it["decided_at"], int(it["seconds_spent"]), it.get("note") or None))
            ids.append(lid)
    return ids


# ----------------------------------------------------------------- candidates
def candidate_row(c: dict) -> dict:
    """One candidate as the generator saved it (generator.Candidate as a dict, one line of the bake-off JSONL)
    turned into a row of the candidate table. A key outside A-D is not stored in gen_key (the table only
    allows A-D); the model's own text stays in raw."""
    if not str(c.get("cand_id") or "").strip():
        raise ValueError("a candidate without cand_id cannot be stored")
    t, opts = c.get("target") or {}, c.get("options") or {}
    key = str(c.get("key") or "").strip().upper()
    return {"cand_id": c["cand_id"], "backend": c.get("backend") or "", "target_module": str(t.get("module") or "").upper(),
            "target_subtopic": t.get("subtopic") or None, "target_band": t.get("difficulty_band") or None,
            "stem": c.get("stem") or "", "opt_a": opts.get("A"), "opt_b": opts.get("B"), "opt_c": opts.get("C"),
            "opt_d": opts.get("D"), "gen_key": key if key in LETTERS else None,
            "valid_format": int(bool(c.get("valid_format"))), "parse_error": c.get("parse_error") or None,
            "raw": c.get("raw")}


_CANDIDATE_SQL = ("INSERT INTO candidate(cand_id,backend,target_module,target_subtopic,target_band,stem,opt_a,opt_b,opt_c,"
                  "opt_d,gen_key,valid_format,parse_error,raw) VALUES (:cand_id,:backend,:target_module,:target_subtopic,"
                  ":target_band,:stem,:opt_a,:opt_b,:opt_c,:opt_d,:gen_key,:valid_format,:parse_error,:raw)")


def add_candidates(con: sqlite3.Connection, cands: list) -> int:
    """Store generated candidates in ONE transaction: all of them or none.

    cands is a list of dicts as saved by the generator. Refused (nothing written) if a cand_id is already
    stored or repeated in the list, or if the database rejects a row (module outside A-E, a sub-topic that
    belongs to another module). Candidates are staff-side records only: they never reach the effective bank."""
    rows = [candidate_row(c) for c in cands]
    ids = [r["cand_id"] for r in rows]
    stored = {r[0] for r in con.execute("SELECT cand_id FROM candidate")}
    dup = sorted((stored & set(ids)) | {i for i in ids if ids.count(i) > 1})
    if dup:
        raise ValueError(f"candidates already stored or repeated: {dup[:5]}")
    try:
        with con:
            con.executemany(_CANDIDATE_SQL, rows)
    except sqlite3.IntegrityError as e:
        raise ValueError(f"candidates not stored (nothing was written): {e}") from e
    return len(rows)


def read_candidates_jsonl(path: str | Path) -> list:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def add_candidates_from_jsonl(con: sqlite3.Connection, path: str | Path) -> int:
    """Store the candidates of one bake-off file (one JSON object per line)."""
    return add_candidates(con, read_candidates_jsonl(path))


# ------------------------------------------------------------- candidate review
def reviewed_candidate_ids(con: sqlite3.Connection) -> set:
    return {r[0] for r in con.execute("SELECT cand_id FROM review_decision")}


def record_candidate_reviews(con: sqlite3.Connection, items: list) -> list:
    """Store the teacher's decisions on candidates and the admitted items in ONE transaction (all or none).

    Each item: cand_id, decision ('approve'|'edit_approve'|'reject'), reject_codes (str or None), confirmed_key,
    reviewer, decided_at, seconds_spent, and for admitted ones stem, options (dict A-D, in the order the teacher
    saw), provenance. A candidate that already has a decision is refused. Admitted items are provisional,
    cluster 0, labelled with the candidate's target module and sub-topic. Returns the new item ids."""
    out = []
    try:
        with con:
            n = con.execute("SELECT COUNT(*) FROM admitted_item").fetchone()[0]
            for it in items:
                cur = con.execute(
                    "INSERT INTO review_decision(cand_id,decision,reject_codes,confirmed_key,reviewer,decided_at,seconds_spent)"
                    " VALUES (?,?,?,?,?,?,?)",
                    (it["cand_id"], it["decision"], it.get("reject_codes") or None, it.get("confirmed_key") or None,
                     it["reviewer"], it["decided_at"], int(it["seconds_spent"])))
                if it["decision"] == "reject":
                    continue
                c = con.execute("SELECT target_module,target_subtopic FROM candidate WHERE cand_id=?",
                                (it["cand_id"],)).fetchone()
                if c is None:
                    raise sqlite3.IntegrityError(f"unknown candidate {it['cand_id']}")
                n += 1
                iid = f"GEN-{n:03d}"
                con.execute(
                    "INSERT INTO admitted_item(item_id,cand_id,review_id,stem,options_json,key,module,subtopic,cluster,"
                    "provisional,provenance) VALUES (?,?,?,?,?,?,?,?,0,1,?)",
                    (iid, it["cand_id"], cur.lastrowid, it["stem"], json.dumps(it["options"]), it["confirmed_key"],
                     c[0], c[1], it["provenance"]))
                out.append(iid)
    except sqlite3.IntegrityError as e:
        raise ValueError(f"reviews not stored (nothing was written): {e}") from e
    return out


# -------------------------------------------------------------- effective bank
def _has_value(v) -> bool:
    return v is not None and not pd.isna(v) and str(v).strip() != ""


def build_effective_bank(df: pd.DataFrame, con: sqlite3.Connection) -> pd.DataFrame:
    """What the tutor sees: the CSV joined with approved labels and admitted items.

    Rules (decision 3 and decision 5 of the notebook)
    - Withdrawn questions are never included.
    - The 519 core questions are always included. Of the 38 earlier unclassified ones, those
      with an approved label are labelled; the rest stay unlabelled ('general knowledge').
    - A question of the new paper is included only once a teacher has approved its label.
    - A label with a sub-topic gives tags [sub-topic]; a module-only label gives no tags and
      label_level 'module', so it can credit module-level mastery only, never a tag.
    - An official label is never replaced; a stray proposal for one is ignored with a warning.
    - Admitted generated items are appended, always provisional, standalone, with provenance.

    df is the derived bank (data.derive). Returns a DataFrame with EFFECTIVE_COLUMNS."""
    lab = {r.qn_id: r for r in approved_labels(con).itertuples()}
    known_unlabelled = set(df.loc[df["unlabelled"] & ~df["withdrawn"], "qn_id"])
    stray = sorted(set(lab) - known_unlabelled)
    if stray:
        warnings.warn(f"ignored approved labels for questions that are officially labelled, "
                      f"withdrawn or unknown: {stray[:5]}", stacklevel=2)

    base = df[~df["withdrawn"]].copy()
    tags, prim, level, src = [], [], [], []
    for qn, tg, unl, pm in zip(base["qn_id"], base["tags"], base["unlabelled"], base["primary_module"]):
        if not unl:
            tags.append(tg); prim.append(pm); level.append("official"); src.append("official")
        elif qn in lab:
            l = lab[qn]
            has_sub = _has_value(l.subtopic)          # pandas turns a missing sub-topic into NaN, which is truthy
            tags.append([l.subtopic] if has_sub else [])
            prim.append(l.module)
            level.append("subtopic" if has_sub else "module")
            src.append(l.label_source)
        else:
            tags.append([]); prim.append(""); level.append("none"); src.append("unlabelled")
    base["tags"], base["primary_module"] = tags, prim
    base["label_level"], base["label_source"] = level, src
    base = base[(base["year"] != NEW_PAPER_YEAR) | (base["label_level"] != "none")].copy()

    base["item_id"], base["source"] = base["qn_id"], "hkeaa"
    base["provisional"], base["provenance"] = False, ""
    parts = [base[EFFECTIVE_COLUMNS]]

    gen = pd.read_sql_query(
        "SELECT item_id, stem, options_json, key, module, subtopic, provenance FROM admitted_item ORDER BY item_id", con)
    if len(gen):
        g = pd.DataFrame({
            "item_id": gen["item_id"], "source": "generated", "qn_id": None, "year": pd.NA,
            "question_text": gen["stem"],
            "answer_options": gen["options_json"].apply(lambda j: format_options(json.loads(j))),
            "correct_answer": gen["key"], "cohort_pct": float("nan"), "cluster": 0,
            "cluster_desc": None, "qn_image_filename": None, "qn_image_type": None,
            "ans_image_filename": None,
            "tags": gen["subtopic"].apply(lambda s: [s] if _has_value(s) else []),
            "primary_module": gen["module"],
            "label_level": gen["subtopic"].apply(lambda s: "subtopic" if _has_value(s) else "module"),
            "label_source": "teacher_approved", "unit": gen["item_id"], "provisional": True,
            "provenance": gen["provenance"]
        })
        parts.append(g[EFFECTIVE_COLUMNS])
    return pd.concat(parts, ignore_index=True)


def state_fingerprint(con: sqlite3.Connection) -> tuple:
    """Cheap change marker for the tutor: rebuild the effective bank only when this changes
    (the governed tables are append-only, so counts and highest ids are enough)."""
    one = lambda q: con.execute(q).fetchone()[0]
    return (one("SELECT COALESCE(MAX(id),0) FROM label_proposal"),
            one("SELECT COUNT(*) FROM admitted_item"),
            one("SELECT COALESCE(MAX(id),0) FROM review_decision"))
