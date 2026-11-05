"""Tests for src/stats.py: gap detection (checked against the numbers in Section 1.8) and the batch-level
tests of Task 5 (Welch's t-test, Hedges' g, the Wilson interval)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from scipy import stats as sps  # noqa: E402

from src import data as D  # noqa: E402
from src import stats as ST  # noqa: E402

tax = D.load_taxonomy(ROOT / "data/taxonomy.csv")
df = D.derive(D.load_bank(ROOT / "data/final_dataset.csv", tax))
core = D.core_bank(df)


def mini(rows):
    """rows: (item_id, unit, tags, provisional)."""
    return pd.DataFrame(rows, columns=["item_id", "unit", "tags", "provisional"])


def test_core_bank_matches_section_1_8():
    pools = ST.pool_sizes(core, tax)
    assert len(pools) == 26
    assert int(pools.units.median()) == 20
    assert int((pools.units < 15).sum()) == 9
    assert set(pools[pools.units < 5].subtopic) == {"d_cp_02", "e_si_01"}
    assert pools.set_index("subtopic").units["d_cp_02"] == 3
    gaps = ST.detect_gaps(core, tax)
    assert len(gaps) == 9 and set(gaps[gaps.thin].subtopic) == {"d_cp_02", "e_si_01"}
    assert "e_si_01" in set(gaps.subtopic)                 # a sub-topic with no question is still seen
    assert gaps.loc[gaps.subtopic == "d_cp_02", "shortfall"].item() == 12
    assert (gaps.reason == "below 15 units").all()


def test_a_cluster_counts_once_and_a_unit_counts_once_per_subtopic():
    b = mini([("q1", "cluster::x", ["a_ip_01"], False), ("q2", "cluster::x", ["a_ip_01"], False),
              ("q3", "q3", ["a_ip_01", "a_ip_02"], False), ("q4", "q4", [], False)])
    p = ST.pool_sizes(b, tax).set_index("subtopic")
    assert p.units["a_ip_01"] == 2 and p.units["a_ip_02"] == 1       # q1+q2 one unit; q3 once in each
    assert p.units.sum() == 3                                          # an untagged question fills no cell


def test_provisional_items_do_not_close_a_gap_unless_asked():
    real = [(f"q{i}", f"q{i}", ["d_cp_02"], False) for i in range(3)]
    gen = [(f"g{i}", f"g{i}", ["d_cp_02"], True) for i in range(20)]
    b = mini(real + gen)
    g = ST.detect_gaps(b, tax).set_index("subtopic")
    assert g.loc["d_cp_02", "units"] == 3 and g.loc["d_cp_02", "generated_units"] == 20 and g.loc["d_cp_02", "thin"]
    assert "d_cp_02" not in ST.detect_gaps(b, tax, include_provisional=True).subtopic.tolist()


def test_no_priority_score_and_stable_order():
    g = ST.detect_gaps(core, tax)
    assert list(g.columns) == ST.GAP_COLUMNS
    assert g.sort_values(["module", "subtopic"]).index.tolist() == list(range(len(g)))


# ------------------------------------------------------------------ Welch's t-test (check 7)
A = [210, 250, 300, 280, 330, 410, 390, 260, 305, 275]
B = [200, 220, 240, 255, 230, 260, 245, 235, 250, 225, 215, 265, 238, 242, 228, 252]


def test_welch_matches_scipy_for_unequal_sizes_and_variances():
    r = ST.welch_t(A, B)
    ref = sps.ttest_ind(A, B, equal_var=False)
    assert r["t"] == pytest.approx(ref.statistic, rel=1e-9)
    assert r["p"] == pytest.approx(ref.pvalue, rel=1e-9)
    assert r["df"] == pytest.approx(ref.df, rel=1e-9)
    assert r["diff"] == pytest.approx(np.mean(A) - np.mean(B))
    assert (r["n_a"], r["n_b"]) == (10, 16)


def test_welch_sign_follows_the_first_sample_and_swapping_flips_it():
    up, down = ST.welch_t(A, B), ST.welch_t(B, A)
    assert up["t"] > 0 > down["t"]
    assert up["t"] == pytest.approx(-down["t"]) and up["p"] == pytest.approx(down["p"])
    assert up["df"] == pytest.approx(down["df"])


def test_welch_identical_samples_give_zero_t_and_p_one():
    r = ST.welch_t([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert r["t"] == 0 and r["p"] == pytest.approx(1.0)


def test_welch_one_sample_without_variation_still_works():
    r = ST.welch_t([5, 5, 5, 5], [4, 6, 5, 7, 3])
    ref = sps.ttest_ind([5, 5, 5, 5], [4, 6, 5, 7, 3], equal_var=False)
    assert r["t"] == pytest.approx(ref.statistic) and r["p"] == pytest.approx(ref.pvalue)


def test_welch_refuses_what_it_cannot_test():
    with pytest.raises(ValueError):
        ST.welch_t([1.0], [1, 2, 3])                      # one value is not a sample
    with pytest.raises(ValueError):
        ST.welch_t([1, 2, np.nan], [1, 2, 3])             # missing values are not dropped silently
    with pytest.raises(ValueError):
        ST.welch_t([4, 4, 4], [4, 4, 4, 4])               # no variation in either sample


def test_hedges_g_matches_a_hand_calculation_and_has_the_right_sign():
    a, b = [1, 2, 3, 4, 5], [3, 4, 5, 6, 7]
    # d = (3 - 5) / sqrt(2.5) = -1.2649; J = 1 - 3 / (4 * 10 - 9) = 0.90323; g = -1.1425
    assert ST.effect_size(a, b) == pytest.approx(-1.2649110640673518 * (1 - 3 / 31))
    assert ST.effect_size(b, a) == pytest.approx(-ST.effect_size(a, b))
    assert ST.effect_size(a, a) == 0


def test_hedges_g_refuses_zero_pooled_variance():
    with pytest.raises(ValueError):
        ST.effect_size([2, 2, 2], [3, 3, 3])


def test_compare_groups_gives_one_row_per_group_in_order():
    t = ST.compare_groups(B, {"x": A, "y": [230, 240, 250, 235]})
    assert list(t.group) == ["x", "y"] and list(t.n) == [10, 4]
    assert t.loc[0, "g"] == pytest.approx(ST.effect_size(A, B))
    assert t.loc[0, "p"] == pytest.approx(ST.welch_t(A, B)["p"])
    assert t.loc[0, "reference_mean"] == pytest.approx(np.mean(B))


def test_a_shifted_batch_is_detected_and_a_same_distribution_batch_is_not():
    rng = np.random.default_rng(7)
    ref = rng.normal(260, 70, 400)
    same = rng.normal(260, 70, 18)
    longer = rng.normal(400, 70, 18)
    assert ST.welch_t(same, ref)["p"] > 0.05
    assert ST.welch_t(longer, ref)["p"] < 0.001 and ST.effect_size(longer, ref) > 1.5


# ------------------------------------------------------------------ Wilson interval
def test_wilson_known_values():
    lo, hi = ST.wilson_interval(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-4)
    lo, hi = ST.wilson_interval(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.7225, abs=1e-4)
    lo, hi = ST.wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)


def test_wilson_stays_inside_zero_and_one_and_contains_the_share():
    for k, n in [(0, 1), (1, 1), (15, 15), (17, 17), (10, 14), (42, 46), (3, 100)]:
        lo, hi = ST.wilson_interval(k, n)
        assert 0 <= lo <= k / n <= hi <= 1


def test_wilson_narrows_as_n_grows_and_refuses_bad_input():
    w = lambda k, n: np.subtract(*ST.wilson_interval(k, n)[::-1])
    assert w(50, 100) < w(5, 10)
    for k, n in [(1, 0), (-1, 5), (6, 5)]:
        with pytest.raises(ValueError):
            ST.wilson_interval(k, n)
