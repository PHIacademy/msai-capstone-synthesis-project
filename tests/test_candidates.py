"""Tests for storing generated candidates (src/store.py) and running the gate on them (src/gate.py, gate_pending)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from src import data as D  # noqa: E402
from src import gate as G  # noqa: E402
from src import generator as GEN  # noqa: E402
from src import store as S  # noqa: E402

tax = GEN.load_taxonomy(ROOT / "data/taxonomy.csv")
df = D.derive(D.prepare_bank(D.load_bank(ROOT / "data/final_dataset.csv", tax)))
STEM = "A school wants to join the computers of two buildings into one network. Which device forwards data packets between two separate networks?"
GOOD = json.dumps({"stem": STEM, "options": {"A": "Router", "B": "Keyboard", "C": "Scanner", "D": "Monitor"}, "key": "A"})
BANNER = json.dumps({"stem": STEM, "options": {"A": "Router", "B": "Keyboard", "C": "Scanner", "D": "Monitor  **END OF SECTION A**"}, "key": "A"})
THREE = "Which device forwards packets?\nA. Router\nB. Keyboard\nC. Scanner"
BADKEY = json.dumps({"stem": STEM, "options": {"A": "Router", "B": "Keyboard", "C": "Scanner", "D": "Monitor"}, "key": "E"})
NOKEY = json.dumps({"stem": STEM, "options": {"A": "Router", "B": "Keyboard", "C": "Scanner", "D": "Monitor"}})


def bake(tmp_path, raws, subtopic="c_ia_01", run_id=None, name="demo.jsonl"):
    cache = tmp_path / f"cache_{name}"
    cache.write_text("\n".join(json.dumps({"module": "C", "subtopic": subtopic, "raw": r}) for r in raws), encoding="utf-8")
    cfg = GEN.GeneratorConfig(name="offline_cache", kind="cache", cache_path=str(cache))
    out = tmp_path / f"out_{name}"
    GEN.bake_off({"offline_cache": cfg}, ["offline_cache"], [GEN.Target("C", subtopic, n=len(raws))], lambda t: GEN.Grounding(), out, run_id=run_id)
    return out / "offline_cache.jsonl"


def store(tmp_path):
    return S.init_db(tmp_path / "s.sqlite", tax)


def test_a_bake_off_file_is_stored_with_every_field(tmp_path):
    con = store(tmp_path)
    assert S.add_candidates_from_jsonl(con, bake(tmp_path, [GOOD, THREE])) == 2
    rows = con.execute("SELECT cand_id, backend, target_module, target_subtopic, target_band, opt_a, opt_d, gen_key, valid_format, parse_error "
                       "FROM candidate ORDER BY cand_id").fetchall()
    good, bad = rows
    assert good[1:] == ("offline_cache", "C", "c_ia_01", "medium", "Router", "Monitor", "A", 1, None)
    assert bad[7:9] == (None, 0) and "missing" in bad[9] and bad[6] is None            # three options: no option D, no key, invalid


def test_storing_the_same_file_twice_is_refused_and_writes_nothing(tmp_path):
    con = store(tmp_path)
    f = bake(tmp_path, [GOOD, BANNER])
    S.add_candidates_from_jsonl(con, f)
    with pytest.raises(ValueError, match="already stored"):
        S.add_candidates_from_jsonl(con, f)
    assert con.execute("SELECT COUNT(*) FROM candidate").fetchone()[0] == 2


def test_two_runs_of_one_backend_are_stored_together(tmp_path):
    con = store(tmp_path)
    S.add_candidates_from_jsonl(con, bake(tmp_path, [GOOD], name="a.jsonl"))
    S.add_candidates_from_jsonl(con, bake(tmp_path, [GOOD], name="b.jsonl"))
    assert con.execute("SELECT COUNT(*) FROM candidate").fetchone()[0] == 2


def test_a_key_outside_a_to_d_is_not_stored_but_the_raw_text_is_kept(tmp_path):
    con = store(tmp_path)
    S.add_candidates_from_jsonl(con, bake(tmp_path, [BADKEY]))
    key, raw, valid = con.execute("SELECT gen_key, raw, valid_format FROM candidate").fetchone()
    assert key is None and '"key": "E"' in raw and valid == 0


def test_the_database_refuses_a_subtopic_of_another_module_and_nothing_is_written(tmp_path):
    con = store(tmp_path)
    ok, bad = json.loads(bake(tmp_path, [GOOD], name="ok.jsonl").read_text()), json.loads(bake(tmp_path, [GOOD], name="bad.jsonl").read_text())
    bad["target"]["subtopic"] = "e_si_02"                                      # module C target with a Module E sub-topic
    with pytest.raises(ValueError, match="nothing was written"):
        S.add_candidates(con, [ok, bad])
    assert con.execute("SELECT COUNT(*) FROM candidate").fetchone()[0] == 0
    with pytest.raises(ValueError, match="without cand_id"):
        S.add_candidates(con, [{"backend": "x"}])


def test_the_gate_runs_once_on_every_stored_candidate(tmp_path):
    con = store(tmp_path)
    S.add_candidates_from_jsonl(con, bake(tmp_path, [GOOD, BANNER, THREE, NOKEY]))
    corpus = G.RealCorpus.from_bank(df)
    res = G.gate_pending(con, corpus)
    assert res.verdict.tolist() == ["pass", "reject", "reject", "flag"]
    assert "paper-end banner" in con.execute("SELECT checks_json FROM gate_result WHERE verdict='reject' ORDER BY cand_id").fetchone()[0]
    assert "no key from the generator" in res.reasons.iloc[3] and "1_format_length" in res.reasons.iloc[2]
    assert con.execute("SELECT COUNT(*) FROM gate_result").fetchone()[0] == 4
    assert len(G.gate_pending(con, corpus)) == 0                                # nothing pending: safe to repeat
    S.add_candidates_from_jsonl(con, bake(tmp_path, [GOOD], name="later.jsonl"))
    assert len(G.gate_pending(con, corpus)) == 1                                # only the new candidate is checked


def test_stored_candidates_never_reach_the_tutor(tmp_path):
    con = store(tmp_path)
    before = S.build_effective_bank(df, con)
    S.add_candidates_from_jsonl(con, bake(tmp_path, [GOOD]))
    G.gate_pending(con, None)
    after = S.build_effective_bank(df, con)
    assert len(before) == len(after) and not (after.source == "generated").any()
