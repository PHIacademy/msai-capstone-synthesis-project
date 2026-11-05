"""Single definition of the HKDSE ICT Paper 1A question bank for the integrated system.

Place at src/data.py. The loader, the classifier, the generator and the store all import
their view of the bank from here, so a rule such as "what counts as text-only" or "which
module is primary" exists once (Section 2.8, Section 3.10 of the notebook).

What this module does
- Loads the immutable official bank (final_dataset.csv) and checks it against the schema
  signature (Section 3.3: shape (560, 13)). It never writes the CSV.
- Loads the static taxonomy (data/taxonomy.csv, 26 sub-topics, 5 modules) and checks it.
- Derives per-question columns: year, tags, modules, ONE primary module (the module of the
  first tag), unlabelled / withdrawn flags, image flags, code-only images treated as text,
  and the served unit (a cluster counts once).
- Provides the documented subsets: the 519 core questions and the 78 unlabelled questions
  that need a teacher-approved label.

Version mentions live in the constants below so a newer fields document or a new release
of the bank is a one-line change.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# ------------------------------------------------------------------ constants
SCHEMA_VERSION = "1.7"            # fields_explanation_v1.7.md (names aligned to the curriculum guide)
EXPECTED_SHAPE = (560, 13)        # bank signature (Section 3.3); a new release is a new versioned file
EXPECTED_COLUMNS = (
    "qn_type_id", "qn_id", "paper_year", "question_text", "qn_image_filename", "qn_image_type",
    "qn_image_context", "ans_image_filename", "answer_options", "correct_answer", "cohort_pct",
    "cluster", "cluster_desc",
)
EXPECTED_TAXONOMY_ROWS = 26
NEW_PAPER_YEAR = 2025             # the paper that arrived unclassified; excluded from the core bank

UNCLASSIFIED = "unclassified"     # reserved token in qn_type_id (fields document, 2.6)
WITHDRAWN_KEY = "*"               # reserved token in correct_answer (fields document, 2.13)
IMAGE_SENTINEL = "[IMAGE]"        # reserved token in answer_options (fields document, 2.12)
LETTERS = ("A", "B", "C", "D")
MODULES = ("A", "B", "C", "D", "E")

# Official module titles, Curriculum and Assessment Guide (the only module names anywhere).
# Module C is "Internet and Its Applications" with a capital I in "Its": the C5 adapter was
# trained on that exact string (decision 18).
OFFICIAL_MODULE_NAMES = {
    "A": "Information Processing",
    "B": "Computer System Fundamentals",
    "C": "Internet and Its Applications",
    "D": "Computational Thinking and Programming",
    "E": "Social Implications",
}

_TAG = re.compile(r"^[a-e]_(ip|cs|ia|cp|si)_\d{2}$")
_QN_ID = re.compile(r"^ICT_(\d{4})_paper1a_q\d{2}$")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)[ \t]*\n?")


# --------------------------------------------------------------- small helpers
def split_list(v) -> list:
    """Split a semicolon-delimited field (tags, image lists). Empty or NaN gives []."""
    if v is None or (isinstance(v, float) and v != v) or pd.isna(v) or str(v).strip() == "":
        return []
    return [t.strip() for t in str(v).split(";") if t.strip()]


def tags_of(v) -> list:
    """Real sub-topic tags of a qn_type_id value ('unclassified' is not a tag)."""
    return [t for t in split_list(v) if t.lower() != UNCLASSIFIED]


def nonblank(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].notna() & (df[col].astype(str).str.strip() != "")


def primary_module_of(tags: list) -> str:
    """Module letter of the FIRST tag (d_cp_02 gives D); '' when there is no tag.

    Tags are stored in ascending order, so for the 33 cross-module questions this is an
    alphabetical convention, not a judgement of which topic dominates (Section 3.3)."""
    return tags[0].split("_")[0].upper() if tags else ""


def primary_module(row) -> str:
    """Row version of primary_module_of, same signature as the one in generator.py."""
    getter = row.get if hasattr(row, "get") else (lambda k, d=None: d)
    return primary_module_of(tags_of(getter("qn_type_id")))


# The last question of each paper (Q40) carries the printed paper-end banner inside option D:
# "**END OF PAPER**" (2012-2018) or "**END OF SECTION A**" (2019-2025). It is not part of the option.
_PAPER_END = re.compile(r"\s*\*\*END OF (?:SECTION [A-Z]|PAPER)\*\*\s*$")


def strip_paper_end(text) -> str:
    """Remove the trailing paper-end banner from an options block (the CSV itself is never changed)."""
    return _PAPER_END.sub("", str(text))


def parse_options(text) -> dict:
    """Split an answer_options block ('A. ...  \\nB. ...') into {'A': ..., ...}.

    The paper-end banner is removed first, so it can never reach a few-shot example, a training row
    or a candidate. Returns {} for the '[IMAGE]' sentinel and for anything that is not an A-D list."""
    parts = re.split(r"(?m)^\s*([A-D])\.\s", "\n" + strip_paper_end(text))
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def format_options(options: dict) -> str:
    """Inverse of parse_options, in the CSV's own layout (markdown line breaks)."""
    return "  \n".join(f"{k}. {options[k]}" for k in LETTERS if k in options)


# ----------------------------------------------------------------- taxonomy
def load_taxonomy(path: str | Path = "data/taxonomy.csv") -> pd.DataFrame:
    """Read the static taxonomy and check it. Raises ValueError listing every problem.

    This is the one place where names come from: module_name must equal the official title
    and sub-topic ids must be consistent with their module and category columns."""
    t = pd.read_csv(path, dtype=str)
    need = {"qn_type_id", "module", "cat_id", "module_name", "subcat_id", "subtopic_name"}
    problems = []
    if not need <= set(t.columns):
        raise ValueError(f"taxonomy missing columns: {sorted(need - set(t.columns))}")
    if len(t) != EXPECTED_TAXONOMY_ROWS:
        problems.append(f"expected {EXPECTED_TAXONOMY_ROWS} rows, found {len(t)}")
    if not t["qn_type_id"].is_unique:
        problems.append("duplicate qn_type_id")
    for _, r in t.iterrows():
        i = r["qn_type_id"]
        if not _TAG.match(str(i)):
            problems.append(f"{i}: not of the form <a-e>_<ip|cs|ia|cp|si>_<NN>")
            continue
        if r["module"] != i[0].upper():
            problems.append(f"{i}: module column {r['module']} disagrees with the id")
        if r["cat_id"] != i[2:4]:
            problems.append(f"{i}: cat_id {r['cat_id']} disagrees with the id")
        if r["subcat_id"] != i[-2:]:
            problems.append(f"{i}: subcat_id {r['subcat_id']} disagrees with the id")
        if OFFICIAL_MODULE_NAMES.get(r["module"]) != r["module_name"]:
            problems.append(f"{i}: module_name {r['module_name']!r} is not the official title")
    if problems:
        raise ValueError("taxonomy check failed:\n  " + "\n  ".join(problems))
    return t


# --------------------------------------------------------------------- bank
def validate_bank(bank: pd.DataFrame, tax: pd.DataFrame | None = None,
                  check_shape: bool = True) -> list:
    """Check the bank against the fields document. Returns a list of problems ([] = clean).

    Column and shape problems are returned first; the row-level checks only run when the
    columns are all present."""
    problems = []
    missing = [c for c in EXPECTED_COLUMNS if c not in bank.columns]
    extra = [c for c in bank.columns if c not in EXPECTED_COLUMNS]
    if missing or extra:
        problems.append(f"columns differ from schema v{SCHEMA_VERSION}: missing {missing}, unexpected {extra}")
    if check_shape and bank.shape != EXPECTED_SHAPE:
        problems.append(f"shape {bank.shape} != expected {EXPECTED_SHAPE}")
    if missing:
        return problems

    if not bank["qn_id"].is_unique:
        problems.append("qn_id is not unique (it is the primary key)")
    bad_id = bank.loc[~bank["qn_id"].astype(str).str.match(_QN_ID), "qn_id"].tolist()
    if bad_id:
        problems.append(f"qn_id not of the form ICT_YYYY_paper1a_qNN: {bad_id[:5]}")
    else:
        id_year = bank["qn_id"].str.extract(_QN_ID)[0].astype(int)
        off = bank.loc[id_year != pd.to_numeric(bank["paper_year"], errors="coerce"), "qn_id"].tolist()
        if off:
            problems.append(f"paper_year disagrees with qn_id for {off[:5]}")

    for qid, v in zip(bank["qn_id"], bank["qn_type_id"]):
        items = split_list(v)
        if not items:
            problems.append(f"{qid}: empty qn_type_id (must be 'unclassified' when untagged)")
        elif UNCLASSIFIED in [i.lower() for i in items]:
            if len(items) != 1:
                problems.append(f"{qid}: 'unclassified' mixed with tags")
        else:
            if items != sorted(set(items)):
                problems.append(f"{qid}: tags not ascending and de-duplicated: {items}")
            for tg in items:
                if not _TAG.match(tg):
                    problems.append(f"{qid}: malformed tag {tg!r}")
                elif tax is not None and tg not in set(tax["qn_type_id"]):
                    problems.append(f"{qid}: tag {tg} is not in the taxonomy")

    key = bank["correct_answer"].astype(str).str.strip()
    bad_key = bank.loc[~key.isin(list(LETTERS) + [WITHDRAWN_KEY]), "qn_id"].tolist()
    if bad_key:
        problems.append(f"correct_answer not in A-D or '*': {bad_key[:5]}")

    if not bank["cluster"].isin([0, 1]).all():
        problems.append("cluster must be 0 or 1")
    else:
        has_desc = nonblank(bank, "cluster_desc")
        if (has_desc != (bank["cluster"] == 1)).any():
            problems.append("cluster_desc must be filled exactly when cluster = 1")
        sizes = bank[bank["cluster"] == 1].groupby("cluster_desc").size()
        if (sizes < 2).any():
            problems.append("a cluster has fewer than two member questions")

    nf, nt, nc = (bank[c].apply(lambda v: len(split_list(v)))
                  for c in ("qn_image_filename", "qn_image_type", "qn_image_context"))
    if ((nf != nt) | (nf != nc)).any():
        problems.append("qn_image_filename / qn_image_type / qn_image_context differ in cardinality")
    bad_types = {t for v in bank["qn_image_type"] for t in split_list(v)} - {"table", "figure", "code"}
    if bad_types:
        problems.append(f"unknown qn_image_type values: {sorted(bad_types)}")

    pct = pd.to_numeric(bank["cohort_pct"], errors="coerce")
    if ((pct < 0) | (pct > 100)).any():
        problems.append("cohort_pct outside 0-100")
    return problems


def load_bank(path: str | Path = "data/final_dataset.csv", tax: pd.DataFrame | None = None,
              strict: bool = True) -> pd.DataFrame:
    """Read the official bank (read-only) and check it. strict=True raises on any problem."""
    bank = pd.read_csv(path)
    problems = validate_bank(bank, tax)
    if problems and strict:
        raise ValueError("bank check failed:\n  " + "\n  ".join(problems))
    return bank


def derive(bank: pd.DataFrame) -> pd.DataFrame:
    """Add the derived columns every component uses. Question text is NOT changed here."""
    df = bank.copy()
    df["year"] = pd.to_numeric(df["paper_year"], errors="coerce").astype("Int64")
    df["tags"] = df["qn_type_id"].apply(tags_of)
    df["unlabelled"] = df["tags"].apply(len) == 0
    df["modules"] = df["tags"].apply(lambda L: sorted({t.split("_")[0].upper() for t in L}))
    df["primary_module"] = df["tags"].apply(primary_module_of)
    df["withdrawn"] = df["correct_answer"].astype(str).str.strip() == WITHDRAWN_KEY

    df["has_body_image"] = nonblank(df, "qn_image_filename")
    df["has_option_image"] = nonblank(df, "ans_image_filename")
    types = df["qn_image_type"].apply(lambda v: set(split_list(v)))
    code_only = df["has_body_image"] & types.apply(lambda s: s == {"code"})
    # prepare_bank blanks the image path of code-only rows, so after derive(prepare_bank(x)) the test above
    # sees no image; keep the flag prepare_bank already set instead of overwriting it.
    prior = df["code_as_text"].fillna(False).astype(bool) if "code_as_text" in df else False
    df["code_as_text"] = code_only | prior
    df["figure_dependent"] = df["has_body_image"] & ~df["code_as_text"]   # figure, table or a mix
    df["image_only_options"] = df["answer_options"].astype(str).str.contains(IMAGE_SENTINEL, regex=False)
    df["text_only"] = ~df["figure_dependent"] & ~df["has_option_image"] & ~df["image_only_options"]

    df["unit"] = [
        f"cluster::{d.strip()}" if (c == 1 and isinstance(d, str) and d.strip()) else q
        for q, c, d in zip(df["qn_id"], df["cluster"], df["cluster_desc"])
    ]
    # A real question that may serve as a generation example (never withdrawn, never a
    # cluster member, never figure-dependent or image-only-option; code-only images count as text).
    df["example_eligible"] = (~df["unlabelled"] & ~df["withdrawn"] & (df["cluster"] != 1) & df["text_only"])
    return df


def prepare_bank(bank: pd.DataFrame) -> pd.DataFrame:
    """Copy of the bank in which code-only body images count as text.

    Rows whose qn_image_type is only 'code' were OCR-converted: the code already sits as a
    fenced block in question_text (or in cluster_desc for cluster members). Here the image
    path is blanked and the markdown image link is removed from question_text, so every later
    step sees a text-only item and no prompt carries a dangling picture reference. Figure,
    table and mixed types (code;figure, code;table) are left untouched. The CSV is never
    modified. Same behaviour as prepare_bank in generator.py, defined once here."""
    df = bank.copy()
    if "qn_image_type" in df:
        types = df["qn_image_type"].apply(lambda v: set(split_list(v)))
        is_code = nonblank(df, "qn_image_filename") & types.apply(lambda s: s == {"code"})
        df["question_text"] = df["question_text"].where(
            ~is_code, df["question_text"].astype(str).str.replace(_MD_IMAGE, "", regex=True))
        df["qn_image_filename"] = df["qn_image_filename"].where(~is_code, None)
        df["code_as_text"] = is_code
    df["primary_module"] = df.apply(primary_module, axis=1)
    return df


# ------------------------------------------------------------------ subsets
def core_bank(df: pd.DataFrame) -> pd.DataFrame:
    """The 519 core questions: every paper before the new one, withdrawn items excluded.
    Includes the 38 earlier unclassified questions, which the tutor keeps under 'general
    knowledge' until a teacher labels them (Capstone 6 usable bank)."""
    return df[(df["year"] != NEW_PAPER_YEAR) & ~df["withdrawn"]].copy()


def unlabelled_for_approval(df: pd.DataFrame) -> pd.DataFrame:
    """The 78 questions whose labels a teacher must approve: unclassified and not withdrawn
    (the 40 questions of the new paper plus the 38 earlier ones)."""
    return df[df["unlabelled"] & ~df["withdrawn"]].copy()
