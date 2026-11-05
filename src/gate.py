"""The gate: deterministic checks on a generated candidate (Section 3.5).

Place at src/gate.py. The gate turns each Capstone 5 failure into a check, and every threshold is a named
constant below with the measurement behind it (Section 1.8: 400 real standalone, text-only items, 2012-2024).
They are STARTING VALUES describing the real bank, not teacher judgement.

Checks (numbers as in the Section 3.5 table)
  1 format and length      hard   complete stem; four non-empty options A-D; 99-503 characters; no repeated-phrase
                                  loop; no printed paper-end banner ('END OF PAPER' / 'END OF SECTION A')
  2 figure / dangling ref  hard   '[Figure ...]', image links, and 'the diagram', 'the above', 'the following
                                  algorithm' etc. with no supporting content in the stem
  3 duplicate options      hard   identical options after trimming and lower-casing
    similar options        soft   one option contained in another, or character similarity above 0.95
  4 copy ratio             hard/soft  share of the item's 6-word sequences found in a real question:
                                  >= 0.5 rejects, 0.4 up to 0.5 flags
  5 key length             soft   the keyed option is longer than 1.5 times the mean of the others
  6 stem-option overlap    soft   only the keyed option repeats a 4-word sequence of the stem
  7 distribution tests     batch level, not here (src/stats.py, Task 5)

Storing: write_gate_result stores one result; gate_pending runs the gate on every stored candidate that has none yet.

Verdict: any failed hard check -> 'reject'; otherwise any failed soft check -> 'flag'; otherwise 'pass'. The
gate never says 'approved': both 'flag' and 'pass' go to the teacher. A check that could not run (no corpus,
no key) is listed in result['not_run']. The generator's key is never authoritative: checks 5 and 6 use it only
as a hint, and a missing key is itself a reason to flag.

The gate calls no model, so it is deterministic and testable. It cannot tell whether the key is right or
whether the question fits its module: the teacher decides both at review.
"""
from __future__ import annotations

import difflib
import itertools
import json
import re
import sqlite3
from collections import Counter

import pandas as pd

from src import data as D

# ---- thresholds (Section 1.8) ------------------------------------------------------------------
LENGTH_MIN = 99               # 1st percentile of stem + options characters in real items
LENGTH_MAX = 503              # 99th percentile
LOOP_REPEATS = 8              # a 3-word sequence repeated this often is a loop; real items reach at most 5
DUP_SIM_FLAG = 0.95           # character similarity: flags 1.2% of real items (0.8 would flag 58%)
COPY_NGRAM = 6                # words per sequence
COPY_REJECT = 0.5             # real items: 0.2% reach 0.5 (the Capstone 5 threshold)
COPY_FLAG = 0.4               # real items: 1.0% reach 0.4
KEY_RATIO_FLAG = 1.5          # real items: 3.0% exceed 1.5 (15.5% exceed 1.3)
OVERLAP_NGRAM = 4             # words; only the keyed option sharing one with the stem is flagged (0.3% of real items)
LETTERS = D.LETTERS
BANNER = re.compile(r"\bEND OF (?:PAPER|SECTION [A-Z])\b", re.I)   # the printed banner only, not "the end of the loop"

_REF_PHRASE = re.compile(
    r"\b(the above|the diagram|the figure|the picture|the table (?:below|above|shown)|shown (?:below|above)|as shown|"
    r"the following (?:diagram|figure|table|picture|code|program|flowchart|pseudocode|algorithm|spreadsheet))\b", re.I)
_PLACEHOLDER = re.compile(r"\[(?:figure|image|diagram|table)\b|!\[", re.I)


# ---- text helpers -----------------------------------------------------------------------------
def normalise(text) -> str:
    """Same cleaning as Section 1.8: markdown marks removed, white space collapsed."""
    return re.sub(r"\s+", " ", re.sub(r"[*`#]", "", str(text))).strip()


def words(text: str) -> list:
    return re.findall(r"\w+", text.lower())


def grams(text: str, n: int) -> set:
    w = words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def item_text(stem: str, options: dict) -> str:
    return normalise(stem) + " " + " ".join(normalise(options.get(k, "")) for k in LETTERS)


class RealCorpus:
    """Every 6-word sequence found in a real question (for the copy ratio, check 4)."""

    def __init__(self, texts):
        self.grams = set()
        for t in texts:
            self.grams |= grams(t, COPY_NGRAM)

    @classmethod
    def from_bank(cls, df: pd.DataFrame) -> "RealCorpus":
        """df is the derived bank (D.derive). Every question counts, withdrawn ones too (copying their text is
        still copying), and so does each shared description of a cluster; code-only images count as text."""
        prep = D.prepare_bank(df)
        texts = []
        for stem, opts in zip(prep["question_text"], prep["answer_options"]):
            texts.append(item_text(stem, D.parse_options(opts)))
        if "cluster_desc" in prep:
            texts += [normalise(d) for d in prep["cluster_desc"].dropna().unique() if str(d).strip()]
        return cls(texts)

    def copy_ratio(self, text: str) -> float:
        g = grams(text, COPY_NGRAM)
        return sum(1 for x in g if x in self.grams) / len(g) if g else 0.0


# ---- the checks --------------------------------------------------------------------------------
def _entry(passed, value=None, threshold=None, level=None, note=None) -> dict:
    """level is what a failure means: 'reject' (hard) or 'flag' (soft); None when the check passed or was not run."""
    return {"passed": passed, "value": value, "threshold": threshold,
            "level": None if passed in (True, None) else level, **({"note": note} if note else {})}


def check_format(stem, options, valid_format=True, parse_error=None) -> dict:
    """Check 1. Returns an entry whose value lists every problem found."""
    problems = []
    if not valid_format:
        problems.append(f"invalid format{': ' + parse_error if parse_error else ''}")
    if not normalise(stem or ""):
        problems.append("empty stem")
    if sorted(options) != list(LETTERS) or any(not normalise(options[k]) for k in options):
        problems.append("not exactly four non-empty options A-D")
    text = item_text(stem or "", options)
    n = len(normalise(stem or "")) + sum(len(normalise(v)) for v in options.values())
    if n < LENGTH_MIN or n > LENGTH_MAX:
        problems.append(f"length {n} outside {LENGTH_MIN}-{LENGTH_MAX}")
    w = words(text)
    rep = max(Counter(tuple(w[i:i + 3]) for i in range(len(w) - 2)).values(), default=0)
    if rep >= LOOP_REPEATS:
        problems.append(f"repeated phrase ({rep} times)")
    if BANNER.search(str(stem)) or any(BANNER.search(str(v)) for v in options.values()):
        problems.append("paper-end banner text")
    return _entry(not problems, "; ".join(problems) or n, f"{LENGTH_MIN}-{LENGTH_MAX} characters, no banner", "reject")


def _supported(stem: str) -> bool:
    """Does the stem carry the content a phrase like 'the following algorithm' refers to?"""
    return "```" in stem or bool(re.search(r"^\s*\|", stem, re.M)) or len([l for l in stem.splitlines() if l.strip()]) >= 3


def check_figure(stem) -> dict:
    """Check 2."""
    s = str(stem or "")
    problems = []
    if _PLACEHOLDER.search(s):
        problems.append("figure placeholder or image link")
    m = _REF_PHRASE.search(s)
    if m and not _supported(s):
        problems.append(f"refers to '{m.group(0)}' with no supporting content")
    return _entry(not problems, "; ".join(problems) or None, "no placeholder or unsupported reference", "reject")


def check_duplicates(options) -> tuple:
    """Check 3: (hard entry for exact duplicates, soft entry for containment or similarity above 0.95)."""
    vals = [normalise(options.get(k, "")).lower() for k in LETTERS if normalise(options.get(k, ""))]
    exact = len(set(vals)) < len(vals)
    contained = any(a in b for a, b in itertools.permutations(vals, 2) if a)
    sim = max((difflib.SequenceMatcher(None, a, b).ratio() for a, b in itertools.combinations(vals, 2)), default=0.0)
    hard = _entry(not exact, "identical options" if exact else None, "no identical options", "reject")
    soft = _entry(not (contained or sim > DUP_SIM_FLAG), {"contained": contained, "max_similarity": round(sim, 3)},
                  f"not contained, similarity <= {DUP_SIM_FLAG}", "flag")
    return hard, soft


def check_copy(text, corpus) -> dict:
    """Check 4. Not run (passed None) when no corpus is supplied."""
    if corpus is None:
        return _entry(None, None, f"flag from {COPY_FLAG}, reject from {COPY_REJECT}", note="no corpus supplied")
    r = round(corpus.copy_ratio(text), 3)
    level = "reject" if r >= COPY_REJECT else "flag"
    return _entry(r < COPY_FLAG, r, f"flag from {COPY_FLAG}, reject from {COPY_REJECT}", level)


def check_key_length(options, key) -> dict:
    """Check 5."""
    if key not in options:
        return _entry(None, None, f"<= {KEY_RATIO_FLAG} x the other options", note="no usable key")
    others = [len(normalise(v)) for k, v in options.items() if k != key]
    mean = sum(others) / len(others) if others else 0
    ratio = round(len(normalise(options[key])) / mean, 2) if mean else None
    return _entry(ratio is None or ratio <= KEY_RATIO_FLAG, ratio, f"<= {KEY_RATIO_FLAG} x the other options", "flag")


def check_stem_overlap(stem, options, key) -> dict:
    """Check 6: only the keyed option shares a 4-word sequence with the stem."""
    if key not in options:
        return _entry(None, None, "keyed option alone repeats a stem phrase", note="no usable key")
    s = grams(normalise(stem), OVERLAP_NGRAM)
    shares = {k: bool(grams(normalise(v), OVERLAP_NGRAM) & s) for k, v in options.items()}
    hit = shares[key] and sum(shares.values()) == 1
    return _entry(not hit, "keyed option alone repeats a stem phrase" if hit else None,
                  "keyed option alone repeats a stem phrase", "flag")


# ---- the gate ----------------------------------------------------------------------------------
def evaluate(stem, options, key=None, *, corpus: RealCorpus | None = None, valid_format=True, parse_error=None) -> dict:
    """Run checks 1 to 6 on one candidate and return {'checks', 'verdict', 'reasons', 'not_run'}.

    options is {'A': ..., 'B': ..., 'C': ..., 'D': ...}; key is the generator's letter or None; corpus is the
    RealCorpus for the copy check (without it check 4 is listed in not_run). valid_format and parse_error come
    from the generator's own parser: a candidate it could not parse is rejected by check 1."""
    options = {k: str(v) for k, v in (options or {}).items()}
    key = (key or "").strip().upper() or None
    dup_hard, dup_soft = check_duplicates(options)
    checks = {
        "1_format_length": check_format(stem, options, valid_format, parse_error),
        "2_figure_reference": check_figure(stem),
        "3_duplicate_options": dup_hard,
        "3_similar_options": dup_soft,
        "4_copy_ratio": check_copy(item_text(stem or "", options), corpus),
        "5_key_length": check_key_length(options, key),
        "6_stem_option_overlap": check_stem_overlap(stem or "", options, key),
    }
    levels = [c["level"] for c in checks.values()]
    verdict = "reject" if "reject" in levels else "flag" if "flag" in levels else "pass"
    reasons = [f"{name}: {c['value']}" for name, c in checks.items() if c["level"]]
    if key is None:
        reasons.append("no key from the generator")
        if verdict == "pass":
            verdict = "flag"
    return {"checks": checks, "verdict": verdict, "reasons": reasons,
            "not_run": [n for n, c in checks.items() if c["passed"] is None]}


def _insert_gate_result(con: sqlite3.Connection, cand_id: str, result: dict) -> None:
    con.execute("INSERT INTO gate_result(cand_id,checks_json,verdict,reasons_json) VALUES (?,?,?,?)",
                (cand_id, json.dumps(result["checks"]), result["verdict"], json.dumps(result["reasons"])))


def write_gate_result(con: sqlite3.Connection, cand_id: str, result: dict) -> None:
    """Store a result in gate_result. The candidate row must exist (foreign key); one result per candidate.
    The solver columns of the table are reserved for an optional extension and stay empty."""
    with con:
        _insert_gate_result(con, cand_id, result)


def gate_pending(con: sqlite3.Connection, corpus: RealCorpus | None = None) -> pd.DataFrame:
    """Run the gate on every stored candidate that has no result yet and store the results in ONE transaction.

    Returns one row per candidate checked (cand_id, backend, verdict, reasons); empty when nothing is pending, so
    calling it again is safe. The candidate's own parse result (valid_format, parse_error) goes into check 1."""
    todo = con.execute("SELECT cand_id, backend, stem, opt_a, opt_b, opt_c, opt_d, gen_key, valid_format, parse_error "
                       "FROM candidate WHERE cand_id NOT IN (SELECT cand_id FROM gate_result) ORDER BY cand_id").fetchall()
    done = []
    for cid, backend, stem, *opts, key, valid, perr in todo:
        options = {k: v for k, v in zip(LETTERS, opts) if v is not None}
        done.append((cid, backend, evaluate(stem, options, key, corpus=corpus, valid_format=bool(valid), parse_error=perr)))
    with con:
        for cid, _, r in done:
            _insert_gate_result(con, cid, r)
    return pd.DataFrame([{"cand_id": c, "backend": b, "verdict": r["verdict"], "reasons": "; ".join(r["reasons"])}
                         for c, b, r in done], columns=["cand_id", "backend", "verdict", "reasons"])
