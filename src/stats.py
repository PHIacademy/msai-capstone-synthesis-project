"""Statistics of the integrated system: gap detection (Section 3.2, step 3) and the batch-level
distribution test (check 7, Task 5).

Place at src/stats.py. The module has two parts.

Part 1: gap detection - which (module, sub-topic) cells need more questions
The rule (Section 3.2, decision 19; thresholds measured in Section 1.8)
- The pool of a sub-topic is counted in SERVED UNITS: a pair of questions that share a diagram or
  description counts once, and a unit counts once for each sub-topic it carries.
- A cell is a GAP when its pool is below MIN_POOL_UNITS (15).
- A pool below THIN_POOL_UNITS (5) is marked 'thin'.
- The loop runs over the taxonomy (26 sub-topics), not over the tags found in the data, so a sub-topic
  with no question at all (e_si_01) is still seen.
- The output is a plain list. There is no priority score or ranking: rows are ordered by module and
  sub-topic id only.

What counts in the pool
- Only questions with a sub-topic label: a module-level label gives no tag, so it cannot fill a cell.
- Admitted generated items are provisional and are counted separately ('generated_units'). By default
  they do NOT close a gap, because the gap list is the target list for generation and the real pool is
  what the generator is grounded on. Pass include_provisional=True to count them as well.

Part 2: batch-level tests (check 7 of the gate design, Section 3.5) and a confidence interval
- welch_t: Welch's t-test (unequal variances, unequal sizes) for two samples of a measurable property,
  such as the length of the stem. Returns the t statistic, the degrees of freedom (Welch-Satterthwaite),
  the two-sided p-value and the means.
- effect_size: Hedges' g, the standardised difference of the two means (pooled standard deviation with the
  small-sample correction). Because the real reference set is much larger than a batch of candidates, the
  pooled standard deviation is close to the reference's own, so g reads as "how many real-question
  standard deviations apart the batch is".
- compare_groups: one row per group against a reference sample (the table the notebook prints).
- wilson_interval: the Wilson score interval for a share k of n (for example the share of reviewed
  candidates that a teacher approved without edits).
The p-value needs scipy (imported when welch_t runs, so gap detection does not depend on it). These are
descriptive batch checks, not a gate: they never accept or reject an item.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

MIN_POOL_UNITS = 15       # Section 1.8: median sub-topic has 20 units; 9 of 26 fall below 15
THIN_POOL_UNITS = 5       # Section 1.8: below 5 is 'thin' (d_cp_02 and e_si_01 on the core bank)

GAP_COLUMNS = ["module", "subtopic", "subtopic_name", "units", "generated_units", "shortfall", "thin", "reason"]


def _unit_tags(bank: pd.DataFrame, provisional: bool) -> pd.DataFrame:
    """One row per (unit, sub-topic) for the real (provisional=False) or generated questions."""
    prov = bank["provisional"].astype(bool) if "provisional" in bank else pd.Series(False, index=bank.index)
    sub = bank[prov == provisional]
    rows = sub[["unit", "tags"]].explode("tags").dropna(subset=["tags"])
    rows = rows[rows["tags"].astype(str).str.strip() != ""]
    return rows.rename(columns={"tags": "subtopic"}).drop_duplicates()


def pool_sizes(bank: pd.DataFrame, tax: pd.DataFrame, include_provisional: bool = False) -> pd.DataFrame:
    """Pool size in served units for every sub-topic of the taxonomy (zero if none).

    bank is the effective bank (or the core bank): it needs `unit` and `tags`, and optionally
    `provisional`. Returns one row per taxonomy sub-topic with units, generated_units and the total used."""
    real = _unit_tags(bank, False).groupby("subtopic")["unit"].nunique()
    gen = _unit_tags(bank, True).groupby("subtopic")["unit"].nunique()
    out = tax[["qn_type_id", "module", "subtopic_name"]].rename(columns={"qn_type_id": "subtopic"}).copy()
    out["units"] = out["subtopic"].map(real).fillna(0).astype(int)
    out["generated_units"] = out["subtopic"].map(gen).fillna(0).astype(int)
    out["counted"] = out["units"] + (out["generated_units"] if include_provisional else 0)
    return out.sort_values(["module", "subtopic"]).reset_index(drop=True)


def detect_gaps(bank: pd.DataFrame, tax: pd.DataFrame, min_units: int = MIN_POOL_UNITS,
                thin_units: int = THIN_POOL_UNITS, include_provisional: bool = False) -> pd.DataFrame:
    """The generation target list: every sub-topic whose pool is below min_units.

    Columns: module, subtopic, subtopic_name, units (real), generated_units, shortfall (units still
    needed to reach min_units, never negative), thin, reason ('below 15 units').
    Ordered by module and sub-topic id; there is no priority score."""
    pools = pool_sizes(bank, tax, include_provisional)
    pools["thin"] = pools["counted"] < thin_units
    below = pools["counted"] < min_units
    pools["shortfall"] = (min_units - pools["counted"]).clip(lower=0)
    pools["reason"] = f"below {min_units} units"
    gaps = pools[below].copy()
    return gaps[GAP_COLUMNS].reset_index(drop=True)


# ------------------------------------------------------------------ Part 2: batch-level tests
def _sample(x, name: str) -> np.ndarray:
    """A sample as a float array. Fewer than two values, or any missing value, is an error: a test that
    silently drops values or runs on one number would report a figure that means nothing."""
    a = np.asarray(list(x), dtype=float)
    if a.size < 2:
        raise ValueError(f"{name}: at least two values are needed, got {a.size}")
    if np.isnan(a).any():
        raise ValueError(f"{name}: contains missing values")
    return a


def welch_t(a, b) -> dict:
    """Welch's t-test for the difference of two means, without assuming equal variances or sizes.

    a is the sample under test (for example the stem lengths of one backend's candidates) and b the
    reference (the real questions). The difference is mean(a) - mean(b), so a positive t means the
    sample is longer. Returns n_a, n_b, mean_a, mean_b, sd_a, sd_b, diff, t, df (Welch-Satterthwaite) and
    p (two-sided). Raises ValueError when a sample has fewer than two values or both samples have no
    variation at all."""
    from scipy import stats as sps          # imported here so that gap detection does not need scipy

    x, y = _sample(a, "a"), _sample(b, "b")
    va, vb = x.var(ddof=1), y.var(ddof=1)
    na, nb = x.size, y.size
    se2 = va / na + vb / nb
    if se2 == 0:
        raise ValueError("both samples have no variation, so the test is undefined")
    diff = float(x.mean() - y.mean())
    t = diff / math.sqrt(se2)
    # Welch-Satterthwaite degrees of freedom; a sample with no variation contributes a zero term
    df = se2 ** 2 / (((va / na) ** 2 / (na - 1)) + ((vb / nb) ** 2 / (nb - 1)))
    p = float(2 * sps.t.sf(abs(t), df))
    return {"n_a": na, "n_b": nb, "mean_a": float(x.mean()), "mean_b": float(y.mean()),
            "sd_a": float(math.sqrt(va)), "sd_b": float(math.sqrt(vb)),
            "diff": diff, "t": float(t), "df": float(df), "p": p}


def effect_size(a, b) -> float:
    """Hedges' g for mean(a) - mean(b): the difference in pooled standard deviations, with the
    small-sample correction J = 1 - 3 / (4 * (n_a + n_b) - 9). Positive when a is larger.

    Raises ValueError for a sample with fewer than two values or when the pooled variance is zero."""
    x, y = _sample(a, "a"), _sample(b, "b")
    na, nb = x.size, y.size
    pooled = ((na - 1) * x.var(ddof=1) + (nb - 1) * y.var(ddof=1)) / (na + nb - 2)
    if pooled == 0:
        raise ValueError("the pooled variance is zero, so the effect size is undefined")
    d = (x.mean() - y.mean()) / math.sqrt(pooled)
    return float(d * (1 - 3 / (4 * (na + nb) - 9)))


def compare_groups(reference, groups: dict) -> pd.DataFrame:
    """Welch's t-test and Hedges' g of every group against one reference sample.

    reference is a list of numbers (the real questions); groups maps a name to a list of numbers (one
    backend's candidates). One row per group, in the order given: n, mean, reference_mean, diff, t, df, p, g."""
    rows = []
    for name, vals in groups.items():
        w = welch_t(vals, reference)
        rows.append({"group": name, "n": w["n_a"], "mean": w["mean_a"], "reference_mean": w["mean_b"],
                     "diff": w["diff"], "t": w["t"], "df": w["df"], "p": w["p"],
                     "g": effect_size(vals, reference)})
    return pd.DataFrame(rows)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval (default 95%) for a share k of n. Returns (low, high) as shares between 0 and 1.

    Unlike the plain normal interval it stays inside 0 and 1 and behaves for shares near 0 or 1 and for
    small n, which is the case for 15 of 15 approved. Raises ValueError unless 0 <= k <= n and n >= 1."""
    if n < 1 or not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n and n >= 1, got k={k}, n={n}")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
