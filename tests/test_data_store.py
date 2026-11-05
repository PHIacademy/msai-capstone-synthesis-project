"""Tests for src/data.py and src/store.py against the real bank.

Run:  python3 -I tests/test_data_store.py        (or pytest, if installed)

Most expected numbers are the ones printed in the notebook (Sections 1.7, 1.8 and 3.3), so a
pass shows the modules reproduce what the notebook reports.
"""
import re
import sqlite3
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import data as D  # noqa: E402
from src import store as S  # noqa: E402

BANK, TAX = ROOT / "data/final_dataset.csv", ROOT / "data/taxonomy.csv"
_cache = {}


def bank_df():
    if "df" not in _cache:
        _cache["tax"] = D.load_taxonomy(TAX)
        _cache["raw"] = D.load_bank(BANK, _cache["tax"])
        _cache["df"] = D.derive(_cache["raw"])
    return _cache["df"]


def tmp_db():
    d = tempfile.mkdtemp()
    return Path(d) / "state.sqlite"


# ------------------------------------------------------------------ data.py
def test_signature_and_schema():
    bank_df()
    assert _cache["raw"].shape == (560, 13)
    assert D.validate_bank(_cache["raw"], _cache["tax"]) == []
    bad = _cache["raw"].drop(columns=["cluster_desc"])
    assert any("columns differ" in p for p in D.validate_bank(bad))
    assert any("shape" in p for p in D.validate_bank(_cache["raw"].iloc[:-1]))


def test_counts_match_notebook_1_7():
    df = bank_df()
    core, unl = D.core_bank(df), D.unlabelled_for_approval(df)
    assert (len(df), int((df.year == 2025).sum()), int(df.withdrawn.sum())) == (560, 40, 1)
    assert len(core) == 519 and core["unit"].nunique() == 506
    cl = core[core["unit"].str.startswith("cluster::")].groupby("unit").size()
    assert (len(cl), int(cl.sum())) == (13, 26)
    assert int(core["unlabelled"].sum()) == 38
    assert int((~df["withdrawn"]).sum()) == 559
    assert len(unl) == 78 and int((unl.year == 2025).sum()) == 40
    mods = df.loc[~df.unlabelled, "primary_module"].value_counts().sort_index().to_dict()
    assert mods == {"A": 154, "B": 84, "C": 119, "D": 74, "E": 50} and int(df.unlabelled.sum()) == 79
    touch = df.explode("modules")["modules"].dropna().value_counts().sort_index().to_dict()
    assert touch == {"A": 154, "B": 93, "C": 141, "D": 75, "E": 53}
    assert int((df.modules.apply(len) > 1).sum()) == 33


def test_images_match_notebook():
    df = bank_df()
    assert round(df.has_body_image.mean() * 100, 1) == 25.9
    assert round(df.has_option_image.mean() * 100, 1) == 0.9
    types = df.loc[df.has_body_image, "qn_image_type"].apply(lambda v: set(D.split_list(v)))
    assert int(df.code_as_text.sum()) == 53
    assert int(types.apply(lambda s: "figure" in s).sum()) == 58
    assert int(types.apply(lambda s: s == {"table"}).sum()) == 32      # notebook: table only 32
    assert int(df.figure_dependent.sum()) == 92 and round(df.figure_dependent.mean() * 100, 1) == 16.4


def test_example_pool_matches_notebook_3_3():
    df = bank_df()
    pool = df[df.example_eligible]
    assert len(pool) == 383
    assert int((pool.primary_module == "D").sum()) == 54
    assert int(pool.tags.apply(lambda L: "d_cp_03" in L).sum()) == 34
    # before code images count as text (341 / 16 / 4 in the notebook)
    pre = df[df.example_eligible & ~df.code_as_text]
    assert (len(pre), int((pre.primary_module == "D").sum()),
            int(pre.tags.apply(lambda L: "d_cp_03" in L).sum())) == (341, 16, 4)


def test_pools_and_taxonomy_match_notebook():
    df = bank_df()
    core = D.core_bank(df)
    pools = core.groupby("unit").tags.apply(lambda s: sorted({t for L in s for t in L})).explode().dropna().value_counts()
    assert pools["d_cp_02"] == 3 and pools["b_cs_05"] == 8 and pools["d_cp_05"] == 9 and pools["d_cp_04"] == 14
    assert sorted(set(_cache["tax"].qn_type_id) - set(core.tags.explode().dropna())) == ["e_si_01"]
    assert int((_cache["tax"].set_index("qn_type_id").reindex(pools.index.union(["e_si_01"])).shape[0])) == 26


def test_taxonomy_check_rejects_wrong_names():
    t = pd.read_csv(TAX, dtype=str)
    t.loc[t.module == "C", "module_name"] = "Internet and its Applications"   # lower-case 'its'
    p = Path(tempfile.mkdtemp()) / "t.csv"
    t.to_csv(p, index=False)
    try:
        D.load_taxonomy(p)
    except ValueError as e:
        assert "official title" in str(e)
    else:
        raise AssertionError("a taxonomy with a wrong module title must be rejected")


# ---- equivalence with generator.py (reference copy of the four definitions it contains)
_G_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)[ \t]*\n?")


def _g_blank(s):
    return s.isna() | (s.astype(str).str.strip() == "")


def _g_tags(v):
    if v is None or (isinstance(v, float) and v != v):
        return []
    return [t.strip() for t in str(v).split(";") if t.strip() and t.strip().lower() != "unclassified"]


def _g_primary_module(row):
    tags = _g_tags(row.get("qn_type_id") if hasattr(row, "get") else None)
    return tags[0][:1].upper() if tags else ""


def _g_prepare_bank(bank):
    df = bank.copy()
    if "qn_image_type" in df:
        is_code = df["qn_image_type"].astype(str).str.strip().str.lower() == "code"
        df["question_text"] = df["question_text"].where(
            ~is_code, df["question_text"].astype(str).str.replace(_G_MD_IMAGE, "", regex=True))
        df["qn_image_filename"] = df["qn_image_filename"].where(~is_code, None)
        df["code_as_text"] = is_code
    df["primary_module"] = df.apply(_g_primary_module, axis=1)
    return df


def test_equivalent_to_generator_definitions():
    bank_df()
    raw = _cache["raw"]
    a, b = D.prepare_bank(raw), _g_prepare_bank(raw)
    assert list(a.columns) == list(b.columns)
    for c in a.columns:
        pd.testing.assert_series_equal(a[c].reset_index(drop=True), b[c].reset_index(drop=True),
                                       check_dtype=False, obj=c)
    assert (raw.apply(D.primary_module, axis=1) == raw.apply(_g_primary_module, axis=1)).all()
    # generator.select_examples filters, applied to the prepared bank, give the same pool
    p = b
    keep = (p["primary_module"] != "") & (p["cluster"] != 1) & _g_blank(p["qn_image_filename"]) \
        & _g_blank(p["ans_image_filename"]) & ~p["answer_options"].astype(str).str.contains("[IMAGE]", regex=False) \
        & (p["correct_answer"].astype(str).str.strip() != "*")
    assert set(p.loc[keep, "qn_id"]) == set(bank_df().loc[bank_df().example_eligible, "qn_id"])
    # the generator compares the whole string with 'code'; check no 'code;code' style row makes them differ
    assert int(b["code_as_text"].sum()) == int(bank_df()["code_as_text"].sum()) == 53
    # official module titles agree with generator.MODULE_NAMES (copied from generator.py)
    assert D.OFFICIAL_MODULE_NAMES == {"A": "Information Processing", "B": "Computer System Fundamentals",
                                       "C": "Internet and Its Applications",
                                       "D": "Computational Thinking and Programming", "E": "Social Implications"}


def test_options_roundtrip():
    df = bank_df()
    ok = df[df.answer_options.apply(lambda o: len(D.parse_options(o)) == 4)]
    assert len(ok) == 535 and (ok.answer_options.apply(lambda o: D.format_options(D.parse_options(o))) ==
                               ok.answer_options.str.strip()).mean() > 0.9
    assert D.parse_options("[IMAGE]") == {}


# ----------------------------------------------------------------- store.py
def test_schema_demo_matches_notebook_3_3():
    """The same 14 attempts as the notebook demo: accepted/rejected pattern must be identical."""
    tax = D.load_taxonomy(TAX)
    con = sqlite3.connect(":memory:")
    con.executescript(S.SCHEMA)
    S.load_taxonomy_table(con, tax)
    out = []

    def attempt(sql, params=()):
        try:
            con.execute(sql, params)
            out.append(True)
        except sqlite3.IntegrityError:
            out.append(False)

    LP = "INSERT INTO label_proposal(qn_id,predicted_module,predicted_subtopic,confidence,label_source) VALUES (?,?,?,?,?)"
    attempt(LP, ("q1", "D", None, 0.81, "classifier"))
    attempt(LP, ("q2", "D", "d_cp_03", 0.93, "classifier"))
    attempt(LP, ("q3", "A", "d_cp_03", 0.90, "classifier"))
    attempt(LP, ("q4", "D", "d_cp_09", 0.90, "classifier"))
    attempt(LP, ("q5", "F", None, 0.50, "classifier"))
    attempt("INSERT INTO label_proposal(qn_id,predicted_module,label_source) VALUES (?,?,?)", ("q1", "D", "teacher_approved"))
    CAND = "INSERT INTO candidate(cand_id,backend,target_module,target_subtopic,stem,valid_format) VALUES (?,?,?,?,?,?)"
    attempt(CAND, ("G1", "demo", "D", None, "s", 1))
    attempt(CAND, ("G2", "demo", "D", "e_si_02", "s", 1))
    RD = "INSERT INTO review_decision(cand_id,decision,confirmed_key,reviewer,decided_at) VALUES (?,?,?,?,?)"
    attempt(RD, ("G1", "approve", None, "t", "x"))
    attempt("INSERT INTO review_decision(cand_id,decision,reviewer,decided_at) VALUES (?,?,?,?)", ("G1", "reject", "t", "x"))
    attempt(RD, ("G1", "approve", "B", "t", "x"))
    ITEM = ("INSERT INTO admitted_item(item_id,cand_id,review_id,stem,options_json,key,module,cluster,provisional,provenance) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)")
    attempt(ITEM, ("I1", "G1", 1, "s", "{}", "B", "D", 1, 1, S.PROVENANCE_LABEL))
    attempt(ITEM, ("I1", "G1", 1, "s", "{}", "B", "D", 0, 0, S.PROVENANCE_LABEL))
    attempt(ITEM, ("I1", "G1", 1, "s", "{}", "B", "D", 0, 1, S.PROVENANCE_LABEL))
    assert out == [True, True, False, False, False, False, True, False, False, False, True, False, False, True]


def test_file_db_enforces_foreign_keys_after_reopen_and_ro_blocks_writes():
    df = bank_df()
    path = tmp_db()
    con = S.init_db(path, _cache["tax"])
    con.close()
    rw = S.connect_rw(path)                     # a fresh connection to an existing file
    try:
        rw.execute("INSERT INTO label_proposal(qn_id,predicted_module,predicted_subtopic,label_source) "
                   "VALUES ('x','A','d_cp_03','classifier')")
    except sqlite3.IntegrityError:
        pass
    else:
        raise AssertionError("foreign key must be enforced on a reopened file")
    ro = S.connect_ro(path)
    for sql in ("INSERT INTO label_proposal(qn_id,predicted_module,label_source) VALUES ('a','A','classifier')",
                "DELETE FROM taxonomy"):
        try:
            ro.execute(sql)
        except sqlite3.OperationalError:
            continue
        raise AssertionError(f"read-only connection accepted: {sql}")
    assert ro.execute("SELECT COUNT(*) FROM taxonomy").fetchone()[0] == 26
    S.init_db(path, _cache["tax"]).close()      # idempotent
    # a changed taxonomy is refused
    t2 = _cache["tax"].copy()
    t2.loc[0, "subtopic_name"] = "Renamed"
    try:
        S.init_db(path, t2)
    except ValueError:
        pass
    else:
        raise AssertionError("a taxonomy that differs from the stored one must be refused")


def test_effective_bank_rules():
    df = bank_df()
    path = tmp_db()
    con = S.init_db(path, _cache["tax"])
    eb = S.build_effective_bank(df, con)
    assert len(eb) == 519 and eb.unit.nunique() == 506 and list(eb.columns) == S.EFFECTIVE_COLUMNS
    assert int((eb.label_level == "none").sum()) == 38 and (eb.source == "hkeaa").all()
    assert not eb.provisional.any() and eb.qn_id.is_unique
    fp0 = S.state_fingerprint(con)

    q25 = df[(df.year == 2025) & df.unlabelled & (df.cluster == 0)].qn_id.iloc[0]
    early = df[(df.year < 2025) & df.unlabelled & ~df.withdrawn].qn_id.iloc[0]
    early2 = df[(df.year < 2025) & df.unlabelled & ~df.withdrawn].qn_id.iloc[1]
    # classifier proposals change nothing
    S.add_label(con, df, q25, "D", "d_cp_03", label_source="classifier", confidence=0.9)
    assert len(S.build_effective_bank(df, con)) == 519
    # teacher labels: module-only and sub-topic
    S.add_label(con, df, q25, "d", None, label_source="teacher_blind", approved_by="T", approved_at="2026-10-24")
    S.add_label(con, df, early, "D", "d_cp_03", label_source="teacher_approved", approved_by="T", approved_at="2026-10-24")
    eb = S.build_effective_bank(df, con).set_index("item_id")
    assert len(eb) == 520
    assert eb.loc[q25, "label_level"] == "module" and eb.loc[q25, "tags"] == [] and eb.loc[q25, "primary_module"] == "D"
    assert eb.loc[early, "label_level"] == "subtopic" and eb.loc[early, "tags"] == ["d_cp_03"]
    assert eb.loc[early2, "label_level"] == "none"
    assert S.state_fingerprint(con) != fp0
    # later teacher label wins
    S.add_label(con, df, q25, "A", None, label_source="teacher_approved", approved_by="T", approved_at="2026-10-25")
    assert S.build_effective_bank(df, con).set_index("item_id").loc[q25, "primary_module"] == "A"

    official = df[~df.unlabelled].qn_id.iloc[0]
    wd = df[df.withdrawn].qn_id.iloc[0]
    for qn, msg in ((official, "never overwritten"), (wd, "withdrawn"), ("nope", "not in the bank")):
        try:
            S.add_label(con, df, qn, "A", label_source="classifier")
        except ValueError as e:
            assert msg in str(e)
        else:
            raise AssertionError(f"label for {qn} must be refused")
    try:
        S.add_label(con, df, early2, "D", label_source="teacher_blind")        # no approver
    except sqlite3.IntegrityError:
        pass
    else:
        raise AssertionError("a teacher label without approver must be refused by the database")
    # a stray approved row for an officially labelled question (bypassing add_label) is ignored with a warning
    with con:
        con.execute("INSERT INTO label_proposal(qn_id,predicted_module,label_source,approved_by,approved_at) "
                    "VALUES (?,?,?,?,?)", (official, "E", "teacher_approved", "T", "2026-10-24"))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        eb2 = S.build_effective_bank(df, con).set_index("item_id")
    assert any("ignored approved labels" in str(x.message) for x in w)
    assert eb2.loc[official, "label_level"] == "official"


def test_admitted_items_enter_as_provisional():
    df = bank_df()
    con = S.init_db(tmp_db(), _cache["tax"])
    with con:
        con.execute("INSERT INTO candidate(cand_id,backend,target_module,target_subtopic,stem,valid_format) "
                    "VALUES ('GEN_x_0001','x','D','d_cp_02','Stem?',1)")
        con.execute("INSERT INTO review_decision(cand_id,decision,confirmed_key,reviewer,decided_at) "
                    "VALUES ('GEN_x_0001','approve','B','T','2026-10-24')")
        con.execute("INSERT INTO admitted_item(item_id,cand_id,review_id,stem,options_json,key,module,subtopic,"
                    "cluster,provisional,provenance) VALUES ('GEN_ITEM_1','GEN_x_0001',1,'Stem?',?,'B','D','d_cp_02',0,1,?)",
                    ('{"A":"a","B":"b","C":"c","D":"d"}', S.PROVENANCE_LABEL))
    eb = S.build_effective_bank(df, con)
    g = eb[eb.source == "generated"].iloc[0]
    assert len(eb) == 520 and g.provisional and g.provenance == S.PROVENANCE_LABEL
    assert g.tags == ["d_cp_02"] and g.cluster == 0 and g.unit == "GEN_ITEM_1"
    assert D.parse_options(g.answer_options) == {"A": "a", "B": "b", "C": "c", "D": "d"}
    # a second item with NO sub-topic next to one with a sub-topic: the missing value must not become a tag
    with con:
        con.execute("INSERT INTO candidate(cand_id,backend,target_module,stem,valid_format) "
                    "VALUES ('GEN_x_0002','x','D','Stem 2?',1)")
        con.execute("INSERT INTO review_decision(cand_id,decision,confirmed_key,reviewer,decided_at) "
                    "VALUES ('GEN_x_0002','approve','C','T','2026-10-24')")
        con.execute("INSERT INTO admitted_item(item_id,cand_id,review_id,stem,options_json,key,module,"
                    "cluster,provisional,provenance) VALUES ('GEN_ITEM_2','GEN_x_0002',2,'Stem 2?',?,'C','D',0,1,?)",
                    ('{"A":"a","B":"b","C":"c","D":"d"}', S.PROVENANCE_LABEL))
    g2 = S.build_effective_bank(df, con).set_index("item_id").loc["GEN_ITEM_2"]
    assert g2.tags == [] and g2.label_level == "module" and g2.primary_module == "D"


def test_code_as_text_survives_prepare_bank_before_derive():
    """The notebook and classifier load with derive(prepare_bank(bank)): the 53 code-only items must still be flagged."""
    bank_df()
    both = D.derive(D.prepare_bank(_cache["raw"]))
    assert int(both.code_as_text.sum()) == 53
    assert (both.text_only == bank_df().text_only).all() and (both.example_eligible == bank_df().example_eligible).all()


def test_paper_end_banner_never_reaches_options():
    """Q40 of each paper carries '**END OF PAPER**' / '**END OF SECTION A**' inside option D (14 rows).
    parse_options removes it, so it cannot reach examples, training rows or candidates; the CSV is unchanged."""
    df = bank_df()
    raw = df.answer_options.astype(str)
    assert raw.str.contains(r"\*\*END OF", regex=True).sum() == 14
    parsed = df.answer_options.apply(D.parse_options)
    assert not parsed.apply(lambda o: any("END OF" in v for v in o.values())).any()
    assert D.parse_options("A. a\nB. b\nC. c\nD. d  \n**END OF PAPER**")["D"] == "d"
    assert D.parse_options("A. a\nB. b\nC. c\nD. d  \n**END OF SECTION A**")["D"] == "d"
    assert D.strip_paper_end("D. an END OF day option") == "D. an END OF day option"


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print("\nall passed" if not failed else f"\n{failed} failed")
    sys.exit(1 if failed else 0)
