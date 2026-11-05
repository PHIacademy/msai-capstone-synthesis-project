"""Tests for the candidate review sheet, the answer-position shuffle and the import (src/candidate_review.py)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
from openpyxl import load_workbook  # noqa: E402

from src import candidate_review as CR  # noqa: E402
from src import data as D  # noqa: E402
from src import gate as G  # noqa: E402
from src import generator as GEN  # noqa: E402
from src import store as S  # noqa: E402

tax = GEN.load_taxonomy(ROOT / "data/taxonomy.csv")
STEMS = ["Which device forwards data packets between two separate networks in a school building?",
         "Which storage unit holds the most data when comparing common units in a computer system?",
         "Which of the following protocols is used to transfer web pages between a browser and a server?",
         "Which data type should be used for storing the average mark of a student in a program?"]


def cand(i, key="A", backend="b1", opts=None):
    o = opts or {"A": f"Router {i}", "B": f"Keyboard {i}", "C": f"Scanner {i}", "D": f"Monitor {i}"}
    return {"cand_id": f"GEN_{backend}_t_{i:04d}", "backend": backend, "stem": STEMS[i % 4] + f" (case {i})", "options": o,
            "key": key, "target": {"module": "C", "subtopic": "c_ia_01", "difficulty_band": "medium"},
            "valid_format": True, "parse_error": ""}


@pytest.fixture
def store(tmp_path):
    con = S.init_db(tmp_path / "s.sqlite", tax)
    cands = [cand(1, "A"), cand(2, "C", "b2"), cand(3, None, "b3"), cand(4, "B")]
    S.add_candidates(con, cands)
    for cid, v in zip([c["cand_id"] for c in cands], ["pass", "flag", "flag", "reject"]):
        con.execute("INSERT INTO gate_result(cand_id,checks_json,verdict,reasons_json) VALUES (?,?,?,?)",
                    (cid, "{}", v, '["no answer key"]' if v == "flag" else "[]"))
    con.commit()
    return con


def fill(path, edits, name="Dr Chan", date="2026-11-02"):
    wb = load_workbook(path)
    ws = wb[CR.SHEET]
    ws["C2"], ws["C3"] = name, date
    for r, vals in edits.items():
        for col, v in vals.items():
            ws[f"{col}{r}"] = v
    wb.save(path)


def test_shuffle_is_deterministic_and_keeps_the_key_on_the_same_text():
    o = {"A": "w", "B": "x", "C": "y", "D": "z"}
    a, b = CR.shuffle_options("c1", o, "B", 42), CR.shuffle_options("c1", o, "B", 42)
    assert a == b
    assert sorted(a["options"].values()) == ["w", "x", "y", "z"]
    assert a["options"][a["key"]] == "x"
    assert "".join(sorted(a["perm"])) == "ABCD"


def test_shuffle_spreads_key_positions():
    o = {"A": "w", "B": "x", "C": "y", "D": "z"}
    keys = [CR.shuffle_options(f"c{i}", o, "A", 42)["key"] for i in range(200)]
    assert {k: keys.count(k) for k in "ABCD"}.values() and min(keys.count(k) for k in "ABCD") > 25


@pytest.mark.parametrize("text", ["All of the above", "None of the above", "Both A and B", "(1) and (2) only", "A and C",
                                  "Only (i)", "(1), (2) and (3)"])
def test_order_dependent_items_are_not_shuffled(text):
    o = {"A": "w", "B": "x", "C": "y", "D": text}
    r = CR.shuffle_options("c1", o, "D", 42)
    assert not r["shuffled"] and r["options"] == o and r["key"] == "D"


def test_every_shuffled_item_actually_moves():
    o = {"A": "w", "B": "x", "C": "y", "D": "z"}
    assert all(CR.shuffle_options(f"c{i}", o, "A", 42)["perm"] != "ABCD" for i in range(300))


def test_incomplete_options_not_shuffled():
    r = CR.shuffle_options("c1", {"A": "w", "B": "x", "C": "", "D": "z"}, "A", 42)
    assert not r["shuffled"]


def test_queue_excludes_rejects_and_decided(store):
    q = CR.candidate_queue(store)
    assert list(q.verdict) == ["pass", "flag", "flag"]


def test_sheet_hides_backend_and_generator_key(store, tmp_path):
    man = CR.export_candidate_sheet(store, tax, tmp_path / "r.xlsx", tmp_path / "m.csv")
    assert len(man) == 3 and set(man.backend) == {"b1", "b2", "b3"}
    ws = load_workbook(tmp_path / "r.xlsx")[CR.SHEET]
    seen = " ".join(str(c.value) for row in ws.iter_rows(min_row=7) for c in row if c.value is not None)
    assert "b1" not in seen and "GEN_" not in seen
    assert [ws.cell(row=7 + i, column=7).value for i in range(3)] == [None] * 3   # key cell empty


def _export(store, tmp_path):
    man = CR.export_candidate_sheet(store, tax, tmp_path / "r.xlsx", tmp_path / "m.csv")
    return man, {r.cand_id: 7 + i for i, r in enumerate(man.itertuples())}


def test_full_import_writes_decisions_and_items(store, tmp_path):
    man, row = _export(store, tmp_path)
    c1, c2, c3 = (row[f"GEN_{b}_t_{i:04d}"] for b, i in (("b1", 1), ("b2", 2), ("b3", 3)))
    fill(tmp_path / "r.xlsx", {c1: {"G": "A", "H": "Approve", "P": 1.5},
                               c2: {"G": "B", "H": "Edit", "I": "Better stem for this question about networks?", "P": 2},
                               c3: {"H": "Reject", "N": "K", "O": "D", "P": 1}})
    res = CR.import_candidate_sheet(tmp_path / "r.xlsx", tmp_path / "m.csv", store)
    assert (res["approved"], res["edited"], res["rejected"], res["admitted"]) == (1, 1, 1, 2)
    d = {r[0]: r for r in store.execute("SELECT cand_id,decision,reject_codes,confirmed_key,seconds_spent FROM review_decision")}
    assert d["GEN_b3_t_0003"][1:3] == ("reject", "K;D") and d["GEN_b1_t_0001"][4] == 90
    adm = store.execute("SELECT item_id,module,subtopic,provisional,cluster,stem,provenance FROM admitted_item ORDER BY item_id").fetchall()
    assert [a[0] for a in adm] == ["GEN-001", "GEN-002"] and all(a[1:5] == ("C", "c_ia_01", 1, 0) for a in adm)
    assert any("Better stem" in a[5] for a in adm) and all("teacher-reviewed" in a[6] for a in adm)
    assert CR.candidate_queue(store).empty


def test_admitted_item_is_saved_in_the_order_the_teacher_saw(store, tmp_path):
    man, row = _export(store, tmp_path)
    m = man[man.cand_id == "GEN_b1_t_0001"].iloc[0]
    fill(tmp_path / "r.xlsx", {row[m.cand_id]: {"G": m.gen_key_shown, "H": "Approve", "P": 1},
                               **{r: {"H": "Reject", "N": "X", "P": 1} for c, r in row.items() if c != m.cand_id}})
    res = CR.import_candidate_sheet(tmp_path / "r.xlsx", tmp_path / "m.csv", store)
    import json
    opts, key = store.execute("SELECT options_json,key FROM admitted_item").fetchone()
    assert json.loads(opts)[key] == "Router 1" and res["key_agreement"] == "1 of 1"


@pytest.mark.parametrize("edits, msg", [
    ({"H": "Approve", "P": 1}, "enter your answer"),
    ({"G": "A", "H": "Reject", "P": 1}, "needs a reason"),
    ({"G": "A", "H": "Edit", "P": 1}, "no edited column"),
    ({"G": "A", "H": "Approve", "I": "x", "P": 1}, "choose Edit"),
    ({"G": "A", "H": "Approve"}, "time taken"),
    ({"H": "Reject", "N": "BOGUS", "P": 1}, "unknown reject reason"),
    ({"G": "A", "H": "Maybe", "P": 1}, "decision must be"),
])
def test_bad_rows_block_the_whole_import(store, tmp_path, edits, msg):
    man, row = _export(store, tmp_path)
    good = {r: {"G": "A", "H": "Approve", "P": 1} for r in row.values()}
    first = row[man.cand_id.iloc[0]]
    good[first] = edits
    fill(tmp_path / "r.xlsx", good)
    with pytest.raises(ValueError, match=msg):
        CR.import_candidate_sheet(tmp_path / "r.xlsx", tmp_path / "m.csv", store)
    assert store.execute("SELECT COUNT(*) FROM review_decision").fetchone()[0] == 0
    assert store.execute("SELECT COUNT(*) FROM admitted_item").fetchone()[0] == 0


def test_missing_rows_name_and_date_and_double_import(store, tmp_path):
    man, row = _export(store, tmp_path)
    all_ok = {r: {"G": "A", "H": "Approve", "P": 1} for r in row.values()}
    fill(tmp_path / "r.xlsx", all_ok, name="", date="02/11/2026")
    with pytest.raises(ValueError) as e:
        CR.import_candidate_sheet(tmp_path / "r.xlsx", tmp_path / "m.csv", store)
    assert "name is empty" in str(e.value) and "YYYY-MM-DD" in str(e.value)
    fill(tmp_path / "r.xlsx", all_ok)
    CR.import_candidate_sheet(tmp_path / "r.xlsx", tmp_path / "m.csv", store)
    with pytest.raises(ValueError, match="already has a decision"):
        CR.import_candidate_sheet(tmp_path / "r.xlsx", tmp_path / "m.csv", store)


def test_changed_candidate_is_refused(store, tmp_path):
    man, row = _export(store, tmp_path)
    store.execute("UPDATE candidate SET opt_a='Something else' WHERE cand_id='GEN_b1_t_0001'")
    store.commit()
    fill(tmp_path / "r.xlsx", {r: {"G": "A", "H": "Approve", "P": 1} for r in row.values()})
    with pytest.raises(ValueError, match="changed since"):
        CR.import_candidate_sheet(tmp_path / "r.xlsx", tmp_path / "m.csv", store)


def test_admitted_items_reach_the_effective_bank_provisional(store, tmp_path):
    man, row = _export(store, tmp_path)
    fill(tmp_path / "r.xlsx", {r: {"G": "A", "H": "Approve", "P": 1} for r in row.values()})
    CR.import_candidate_sheet(tmp_path / "r.xlsx", tmp_path / "m.csv", store)
    df = D.derive(D.prepare_bank(D.load_bank(ROOT / "data/final_dataset.csv", tax)))
    eff = S.build_effective_bank(df, store)
    g = eff[eff.source == "generated"]
    assert len(g) == 3 and g.provisional.all() and (g.cluster == 0).all()
