"""Integration test against the real store (skipped when data/state.sqlite or the bank is absent)."""
import sqlite3
from pathlib import Path

import pytest

DB = Path("data/state.sqlite")
pytestmark = pytest.mark.skipif(not (DB.exists() and Path("data/final_dataset.csv").exists() and Path("data/taxonomy.csv").exists()),
                                reason="needs data/state.sqlite, data/final_dataset.csv and data/taxonomy.csv")


def test_tutor_from_store_matches_the_effective_bank_and_cannot_write():
    from src import data as D, store as S
    from src.tutor import Tutor
    tax = D.load_taxonomy("data/taxonomy.csv")
    qdf = D.derive(D.prepare_bank(D.load_bank("data/final_dataset.csv", tax)))
    t = Tutor.from_store(qdf, DB, tax)
    con = S.connect_ro(DB)
    try:
        eb = S.build_effective_bank(qdf, con)
        n_adm = con.execute("SELECT COUNT(*) FROM admitted_item").fetchone()[0]
        n_ok = con.execute("SELECT COUNT(*) FROM review_decision WHERE decision != 'reject'").fetchone()[0]
        assert len(t.eb) == len(eb)
        assert int((t.eb["source"] == "generated").sum()) == n_adm == n_ok     # only approved candidates arrive
        with pytest.raises(sqlite3.OperationalError):
            con.execute("INSERT INTO taxonomy VALUES ('x','A','x','x')")        # the tutor's connection type cannot write
    finally:
        con.close()
    assert not any(hasattr(t, a) for a in ("con", "conn", "connection"))        # the tutor keeps no connection


def test_render_hides_the_banner_and_shows_option_pictures_on_the_real_bank():
    from src import data as D
    from src.tutor import Tutor
    tax = D.load_taxonomy("data/taxonomy.csv")
    qdf = D.derive(D.prepare_bank(D.load_bank("data/final_dataset.csv", tax)))
    t = Tutor.from_store(qdf, DB, tax)
    banner = [u for u, m in t.units.items() if any("END OF" in str(t.by_id.loc[i]["answer_options"]) for i in m["items"])]
    assert banner and all("END OF" not in t.render(u) for u in banner)
    pics = t.eb[t.eb.ans_image_filename.notna() & (t.eb.ans_image_filename.astype(str).str.strip() != "")]
    assert len(pics)
    for _, r in pics.iterrows():
        txt = t.render(r.unit).replace("\\", "/")
        assert "[IMAGE]" not in txt and all(f.strip() in txt for f in str(r.ans_image_filename).split(";"))
