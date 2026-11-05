"""Module (and, only where safe, sub-topic) classifier for unlabelled questions (Capstone 3 method).

Place at src/classifier.py (Sections 2.7, 3.2 and 3.4 of the notebook).

What it does
- Learns the module of a question from its text (`question_text` plus `cluster_desc`) using the
  labelled questions of the bank. Three model families from Capstone 3 (logistic regression, linear
  SVC, decision tree) on TF-IDF features, tuned by grid search over folds grouped by served unit
  (a cluster never straddles folds). The best family by cross-validated macro-F1 is kept.
- Abstains when the module confidence is below a threshold set from out-of-fold predictions on the
  training years (the lowest confidence at which answered questions reach TARGET_ACC; MODULE_CONF is
  the fallback). A sub-topic is predicted only when the
  module was predicted, the sub-topic has at least SUBTOPIC_MIN_TRAIN training questions
  (2012-2022, any tag, Section 1.8) and its confidence is at least SUBTOPIC_CONF.
- Evaluation: temporal hold-out (train 2012-2022, test 2023-2024) against the Capstone 5
  nearest-centroid proxy on the same split, scored strictly (first-tag module) and leniently (any
  module of the question); and against the teacher's blind labels (the ground truth for the
  questions that have no official label), with the option to leave out the old-syllabus questions.
- Output: `propose_labels` writes CLASSIFIER rows only (`label_source = 'classifier'`, no approver).
  A teacher's approval is a different row, written by `src/review.py`. A question that already has a
  teacher label, or that already has a classifier row, is skipped, and abstentions write nothing.

Run it as a module from the project root (python -m src.classifier evaluate), because it imports src.data.

Everything below the constants is deterministic (fixed seed). The constants are starting values;
Task 5 reports sensitivity.
"""
from __future__ import annotations

import argparse
import re
import warnings
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src import data as D
from src import store as S

# ------------------------------------------------------------------ constants (named, Section 1.8)
SEED = 42                      # the seed used throughout Capstone 5
N_FOLDS = 5                    # folds for the grouped grid search
TRAIN_LAST_YEAR = 2022         # temporal split: train 2012-2022 ...
TEST_YEARS = (2023, 2024)      # ... test 2023-2024
SUBTOPIC_MIN_TRAIN = 15        # a sub-topic is predicted only with this many training questions (17 of 26)
TARGET_ACC = 0.90              # the abstention threshold is the lowest confidence at which the answered
                               # questions are this accurate in cross-validation (training years only)
MODULE_CONF = 0.50             # fallback threshold if no confidence level reaches TARGET_ACC
THRESHOLD_GRID = tuple(round(x, 2) for x in np.arange(0.20, 0.96, 0.05))
SUBTOPIC_CONF = 0.70           # sub-topic confidence, renormalised within the predicted module (starting value)
OLD_SYLLABUS_NOTE = "old syllabus"

_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


# ------------------------------------------------------------------ text
def classifier_text(question_text, cluster_desc=None) -> str:
    """The text the classifier sees: the shared preamble (if any) followed by the question.
    Image links and markdown marks are removed. The options are not used (Section 3.2, step 1)."""
    parts = []
    if S._has_value(cluster_desc):
        parts.append(str(cluster_desc))
    parts.append(str(question_text))
    t = _MD_IMAGE.sub(" ", "\n".join(parts)).replace("**", " ").replace("`", " ")
    return re.sub(r"\s+", " ", t).strip()


def frame_texts(frame: pd.DataFrame) -> list:
    return [classifier_text(q, c) for q, c in zip(frame["question_text"], frame["cluster_desc"])]


def training_frame(df: pd.DataFrame, up_to_year: int = TRAIN_LAST_YEAR) -> pd.DataFrame:
    """Labelled, non-withdrawn questions up to a year. Unclassified questions are never training data."""
    return df[~df["unlabelled"] & ~df["withdrawn"] & (df["year"] <= up_to_year)].copy()


def test_frame(df: pd.DataFrame, years=TEST_YEARS) -> pd.DataFrame:
    return df[~df["unlabelled"] & ~df["withdrawn"] & df["year"].isin(list(years))].copy()


def cv_splits(train: pd.DataFrame, n_folds: int = N_FOLDS) -> list:
    """Folds grouped by served unit: the members of a cluster always fall in the same fold."""
    groups = train["unit"].to_numpy()
    k = min(n_folds, len(set(groups)))
    return list(GroupKFold(n_splits=k).split(np.zeros(len(train)), train["primary_module"], groups))


# ------------------------------------------------------------------ module model
def _pipeline(estimator) -> Pipeline:
    return Pipeline([("tfidf", TfidfVectorizer(lowercase=True, sublinear_tf=True, strip_accents="unicode")),
                     ("clf", estimator)])


def _families(fast: bool) -> dict:
    ngrams = [(1, 1)] if fast else [(1, 1), (1, 2)]
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"clf__C": [1, 8] if fast else [0.5, 2, 8, 32], "tfidf__ngram_range": ngrams}),
        "svc_linear": (
            SVC(kernel="linear", probability=True, class_weight="balanced", random_state=SEED),
            {"clf__C": [1] if fast else [0.5, 2, 8], "tfidf__ngram_range": ngrams}),
        "decision_tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
            {"clf__max_depth": [10] if fast else [None, 10, 20],
             "clf__min_samples_leaf": [1] if fast else [1, 2], "tfidf__ngram_range": ngrams}),
    }


@dataclass
class ModuleModel:
    name: str
    estimator: Pipeline
    cv_table: pd.DataFrame     # one row per family: best parameters and cross-validated macro-F1
    threshold: float = MODULE_CONF       # abstain below this confidence (set from out-of-fold predictions)
    oof_table: pd.DataFrame | None = None  # threshold, coverage and accuracy among answered, out of fold

    @property
    def classes(self) -> list:
        return list(self.estimator.classes_)


def fit_module_model(train: pd.DataFrame, fast: bool = False) -> ModuleModel:
    """Grid search per family over cluster-grouped folds; keep the family with the best macro-F1."""
    X, y = frame_texts(train), train["primary_module"].to_numpy()
    cv = cv_splits(train)
    rows, best = [], None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, (est, grid) in _families(fast).items():
            gs = GridSearchCV(_pipeline(est), grid, scoring="f1_macro", cv=cv, refit=True, n_jobs=1)
            gs.fit(X, y)
            rows.append({"family": name, "cv_macro_f1": round(float(gs.best_score_), 4),
                         "best_params": {k: v for k, v in gs.best_params_.items()}})
            if best is None or gs.best_score_ > best[1] + 1e-12:
                best = (name, gs.best_score_, gs.best_estimator_)
    oof = oof_threshold_table(best[2], X, y, cv)
    return ModuleModel(best[0], best[2], pd.DataFrame(rows), choose_threshold(oof), oof)


def oof_threshold_table(estimator, X, y, cv) -> pd.DataFrame:
    """Out-of-fold confidence of the chosen model on the training years: for each threshold, the share
    of questions answered and the strict accuracy among them. The test years are never used here."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba = cross_val_predict(clone(estimator), X, y, cv=cv, method="predict_proba")
    classes = np.array(sorted(set(y)))
    pred, conf, y = classes[proba.argmax(axis=1)], proba.max(axis=1), np.asarray(y)
    rows = []
    for t in THRESHOLD_GRID:
        ans = conf >= t
        rows.append({"threshold": t, "coverage": round(float(ans.mean()), 3),
                     "accuracy_answered": round(float((pred[ans] == y[ans]).mean()), 3) if ans.any() else float("nan")})
    return pd.DataFrame(rows)


def choose_threshold(oof: pd.DataFrame, target: float = TARGET_ACC) -> float:
    """Lowest threshold (most questions answered) whose answered questions reach the target accuracy."""
    ok = oof[(oof["accuracy_answered"] >= target) & (oof["coverage"] > 0)]
    return float(ok["threshold"].min()) if len(ok) else MODULE_CONF


def predict_texts(model: ModuleModel, texts: list) -> pd.DataFrame:
    """Module, confidence and abstain flag for each text."""
    proba = model.estimator.predict_proba(texts)
    classes = np.array(model.classes)
    idx = proba.argmax(axis=1)
    conf = proba[np.arange(len(texts)), idx]
    return pd.DataFrame({"module": classes[idx], "confidence": conf, "abstain": conf < model.threshold})


# ------------------------------------------------------------------ sub-topic model
def eligible_subtopics(df: pd.DataFrame, minimum: int = SUBTOPIC_MIN_TRAIN) -> list:
    """Sub-topics with at least `minimum` training questions in 2012-2022 (any tag; Section 1.8).

    The count always uses 2012-2022, also for a model trained on later years, so the eligible list is the
    same everywhere (17 of 26) and does not depend on which model is being fitted."""
    train = training_frame(df, TRAIN_LAST_YEAR)
    counts = Counter(t for tags in train["tags"] for t in tags if t.lower() != D.UNCLASSIFIED)
    return sorted(t for t, n in counts.items() if n >= minimum)


@dataclass
class SubtopicModel:
    estimator: Pipeline | None
    eligible: list

    @property
    def classes(self) -> list:
        return list(self.estimator.classes_) if self.estimator is not None else []


def fit_subtopic_model(train: pd.DataFrame, eligible: list) -> SubtopicModel:
    """One logistic regression over the eligible sub-topics. A question is a training row for the
    first of its tags that is eligible and belongs to its primary module."""
    X, y = [], []
    for text, tags, mod in zip(frame_texts(train), train["tags"], train["primary_module"]):
        pick = next((t for t in tags if t in eligible and t[0].upper() == mod), None)
        if pick:
            X.append(text)
            y.append(pick)
    if len(set(y)) < 2:
        return SubtopicModel(None, eligible)
    est = _pipeline(LogisticRegression(max_iter=2000, C=8, class_weight="balanced"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(X, y)
    return SubtopicModel(est, eligible)


def predict_subtopics(sub: SubtopicModel, texts: list, modules: list, abstain: list) -> list:
    """(sub-topic or None, confidence) per text. The probability is renormalised over the eligible
    sub-topics of the predicted module, so the sub-topic always belongs to that module. A module with
    fewer than two eligible sub-topics, or an abstained module, gets no sub-topic."""
    out = [(None, None)] * len(texts)
    if sub.estimator is None or not texts:
        return out
    proba, classes = sub.estimator.predict_proba(texts), np.array(sub.classes)
    res = []
    for i, (m, ab) in enumerate(zip(modules, abstain)):
        mask = np.array([c[0].upper() == m for c in classes])
        if ab or mask.sum() < 2:
            res.append((None, None))
            continue
        p = proba[i] * mask
        p = p / p.sum()
        j = int(p.argmax())
        res.append((str(classes[j]), float(p[j])) if p[j] >= SUBTOPIC_CONF else (None, float(p[j])))
    return res


# ------------------------------------------------------------------ the whole classifier
@dataclass
class Classifier:
    module: ModuleModel
    subtopic: SubtopicModel
    trained_up_to: int
    n_train: int


def fit_classifier(df: pd.DataFrame, up_to_year: int = TRAIN_LAST_YEAR, fast: bool = False) -> Classifier:
    train = training_frame(df, up_to_year)
    return Classifier(fit_module_model(train, fast), fit_subtopic_model(train, eligible_subtopics(df)),
                      up_to_year, len(train))


def predict_frame(clf: Classifier, frame: pd.DataFrame) -> pd.DataFrame:
    """module, confidence, abstain, subtopic, subtopic_conf for each row (index = qn_id)."""
    texts = frame_texts(frame)
    p = predict_texts(clf.module, texts)
    sub = predict_subtopics(clf.subtopic, texts, p["module"].tolist(), p["abstain"].tolist())
    p["subtopic"], p["subtopic_conf"] = [s for s, _ in sub], [c for _, c in sub]
    p.index = frame["qn_id"].to_numpy()
    return p


# ------------------------------------------------------------------ baseline (Capstone 5 proxy)
def fit_centroid_baseline(train: pd.DataFrame) -> Pipeline:
    """TF-IDF nearest centroid: the Capstone 5 module proxy, re-run on the same split."""
    est = _pipeline(NearestCentroid())
    est.fit(frame_texts(train), train["primary_module"].to_numpy())
    return est


# ------------------------------------------------------------------ scoring
def score_modules(truth_primary, truth_modules, pred_module, abstain) -> dict:
    """Strict = first-tag module; lenient = any module of the question. Accuracy over answered
    questions, accuracy counting an abstention as wrong, and macro-F1 over answered questions (over the
    modules that occur among them, so a small subset is not penalised for a module it does not contain)."""
    tp, pm, ab = np.asarray(truth_primary), np.asarray(pred_module), np.asarray(abstain, dtype=bool)
    n, ans = len(tp), ~ab
    strict = (tp == pm)
    lenient = np.array([p in set(m) for p, m in zip(pm, truth_modules)])
    k = int(ans.sum())
    f1 = (f1_score(tp[ans], pm[ans], labels=sorted(set(tp[ans]) | set(pm[ans])), average="macro", zero_division=0) if k else float("nan"))
    return {"n": n, "answered": k, "abstention_rate": round(1 - k / n, 3) if n else float("nan"),
            "strict_acc_answered": round(float(strict[ans].mean()), 3) if k else float("nan"),
            "lenient_acc_answered": round(float(lenient[ans].mean()), 3) if k else float("nan"),
            "strict_acc_all": round(float((strict & ans).sum() / n), 3) if n else float("nan"),
            "macro_f1_answered": round(float(f1), 3)}


def error_table(frame: pd.DataFrame, pred: pd.DataFrame, by: str) -> pd.DataFrame:
    """Per-module or per-year table of questions, abstentions and strict accuracy among answered."""
    t = pd.DataFrame({by: (frame["primary_module"] if by == "module" else frame["year"]).to_numpy(),
                      "correct": (frame["primary_module"].to_numpy() == pred["module"].to_numpy()),
                      "abstain": pred["abstain"].to_numpy()})
    g = t.groupby(by)
    out = pd.DataFrame({"n": g.size(), "abstained": g["abstain"].sum()})
    out["accuracy_answered"] = g.apply(lambda x: round(float(x.loc[~x.abstain, "correct"].mean()), 3)
                                       if (~x.abstain).any() else float("nan"), include_groups=False)
    return out


def _subtopic_score(truth_tags, pred_sub) -> dict:
    """Coverage and lenient accuracy of the predicted sub-topics (right if it is among the question's tags)."""
    made = [(t, p) for t, p in zip(truth_tags, pred_sub) if isinstance(p, str)]
    return {"n": len(truth_tags), "predicted": len(made),
            "coverage": round(len(made) / len(truth_tags), 3) if truth_tags else float("nan"),
            "accuracy_predicted": round(float(np.mean([p in t for t, p in made])), 3) if made else float("nan")}


def temporal_evaluation(df: pd.DataFrame, fast: bool = False) -> dict:
    """Train 2012-2022, test 2023-2024. Classifier against the Capstone 5 proxy on the same split."""
    clf = fit_classifier(df, TRAIN_LAST_YEAR, fast)
    test = test_frame(df)
    pred = predict_frame(clf, test)
    base = fit_centroid_baseline(training_frame(df, TRAIN_LAST_YEAR)).predict(frame_texts(test))
    return {
        "model": clf.module.name, "cv_table": clf.module.cv_table, "threshold": clf.module.threshold,
        "oof_table": clf.module.oof_table, "n_train": clf.n_train, "n_test": len(test),
        "classifier": score_modules(test["primary_module"], test["modules"], pred["module"], pred["abstain"]),
        "baseline_proxy": score_modules(test["primary_module"], test["modules"], base, np.zeros(len(test), bool)),
        "per_module": error_table(test, pred, "module"), "per_year": error_table(test, pred, "year"),
        "subtopic": _subtopic_score(test["tags"].tolist(), pred["subtopic"].tolist()),
        "eligible_subtopics": len(eligible_subtopics(df)),
    }


def old_syllabus_ids(xlsx_path) -> set:
    """Question IDs the teacher marked 'Old syllabus' in the Notes column of the returned sheet."""
    from src.review import read_blind_sheet
    _, _, rows = read_blind_sheet(xlsx_path)
    return {r["qn_id"] for r in rows if OLD_SYLLABUS_NOTE in r["notes"].lower()}


def evaluate_blind(df: pd.DataFrame, con, exclude_ids=(), fast: bool = False, clf: Classifier | None = None) -> dict:
    """Score the classifier against the teacher's blind labels. The model is trained on every
    labelled question (the blind questions have no official label, so none is in the training set).
    Reported for all blind labels and for the set without `exclude_ids` (the old-syllabus questions)."""
    lab = S.approved_labels(con)
    lab = lab[lab["label_source"] == "teacher_blind"]
    sample = df[df["qn_id"].isin(lab["qn_id"])].merge(lab[["qn_id", "module", "subtopic"]], on="qn_id",
                                                     suffixes=("", "_teacher"))
    clf = clf or fit_classifier(df, int(df.loc[~df["unlabelled"], "year"].max()), fast)
    pred = predict_frame(clf, sample)
    sample["_pm"], sample["_ab"] = pred["module"].to_numpy(), pred["abstain"].to_numpy()
    sample["_ps"] = pred["subtopic"].to_numpy()

    def block(s: pd.DataFrame) -> dict:
        truth_sub = [[t] if isinstance(t, str) and t else [] for t in s["subtopic"]]
        has_truth = [i for i, t in enumerate(truth_sub) if t]
        return {"module": score_modules(s["module"], [[m] for m in s["module"]], s["_pm"], s["_ab"]),
                "subtopic": _subtopic_score([truth_sub[i] for i in has_truth], [s["_ps"].iloc[i] for i in has_truth])}

    keep = ~sample["qn_id"].isin(set(exclude_ids))
    return {"n": len(sample), "all": block(sample), "without_excluded": block(sample[keep]),
            "n_excluded": int((~keep).sum()),
            "from_2025": block(sample[sample["year"] == D.NEW_PAPER_YEAR]),
            "earlier_years": block(sample[sample["year"] != D.NEW_PAPER_YEAR]),
            "rows": sample[["qn_id", "year", "module", "subtopic", "_pm", "_ab", "_ps"]]
            .rename(columns={"module": "teacher_module", "subtopic": "teacher_subtopic",
                             "_pm": "pred_module", "_ab": "abstained", "_ps": "pred_subtopic"})}


# ------------------------------------------------------------------ writing proposals
def propose_labels(con, df: pd.DataFrame, clf: Classifier) -> pd.DataFrame:
    """Write classifier proposals for the questions that await approval.

    Skips questions that already have a teacher label or a classifier row, and writes nothing for an
    abstention (a module is required). Rows carry label_source 'classifier' and no approver; the whole
    batch is one transaction. Returns every question considered, with a `written` flag."""
    pool = D.unlabelled_for_approval(df)
    have = set(S.approved_labels(con)["qn_id"])
    have |= set(pd.read_sql_query("SELECT qn_id FROM label_proposal WHERE label_source='classifier'", con)["qn_id"])
    todo = pool[~pool["qn_id"].isin(have)]
    if todo.empty:
        return pd.DataFrame(columns=["qn_id", "module", "confidence", "abstain", "subtopic", "subtopic_conf", "written"])
    pred = predict_frame(clf, todo)
    labels = [{"qn_id": q, "module": r["module"], "subtopic": r["subtopic"] if isinstance(r["subtopic"], str) else None,
               "label_source": "classifier",
               "confidence": float(r["confidence"])} for q, r in pred.iterrows() if not r["abstain"]]
    S.add_labels(con, df, labels)
    out = pred.reset_index(names="qn_id")
    out["written"] = ~out["abstain"]
    return out[["qn_id", "module", "confidence", "abstain", "subtopic", "subtopic_conf", "written"]]


# ------------------------------------------------------------------ command line
def _load(bank: str, taxonomy: str) -> pd.DataFrame:
    tax = D.load_taxonomy(taxonomy)
    return D.derive(D.prepare_bank(D.load_bank(bank, tax)))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Module classifier: evaluate, or write proposals.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("evaluate", "propose"):
        p = sub.add_parser(name)
        p.add_argument("--bank", default="data/final_dataset.csv")
        p.add_argument("--taxonomy", default="data/taxonomy.csv")
        p.add_argument("--db", default=str(S.DEFAULT_DB))
        p.add_argument("--fast", action="store_true", help="small grid, for a quick look")
    sub.choices["evaluate"].add_argument("--sheet", default="outputs/blind_label_sheet_filled.xlsx",
                                         help="returned sheet, used to find the old-syllabus notes")
    a = ap.parse_args(argv)
    df = _load(a.bank, a.taxonomy)
    pd.set_option("display.width", 160)
    if a.cmd == "evaluate":
        ev = temporal_evaluation(df, a.fast)
        print(f"Temporal hold-out: train {ev['n_train']} (2012-{TRAIN_LAST_YEAR}), test {ev['n_test']} "
              f"({TEST_YEARS[0]}-{TEST_YEARS[1]}); chosen family: {ev['model']}; "
              f"abstention threshold {ev['threshold']:.2f} (from out-of-fold predictions)")
        print(ev["cv_table"].to_string(index=False))
        print("classifier :", ev["classifier"])
        print("C5 proxy   :", ev["baseline_proxy"])
        print("sub-topics :", ev["subtopic"], f"| eligible: {ev['eligible_subtopics']} of 26")
        print(ev["per_module"], "\n", ev["per_year"], sep="")
        con = S.connect_ro(a.db)
        try:
            from pathlib import Path
            old = old_syllabus_ids(a.sheet) if Path(a.sheet).exists() else set()
            bl = evaluate_blind(df, con, old, a.fast)
        finally:
            con.close()
        print(f"\nBlind labels: n = {bl['n']} ({bl['n_excluded']} old-syllabus questions left out in 'without_excluded')")
        for k in ("all", "without_excluded", "from_2025", "earlier_years"):
            print(f"{k:17s}", bl[k])
    else:
        clf = fit_classifier(df, int(df.loc[~df["unlabelled"], "year"].max()), a.fast)
        con = S.connect_rw(a.db)
        try:
            out = propose_labels(con, df, clf)
        finally:
            con.close()
        print(f"considered {len(out)}, written {int(out['written'].sum())}, abstained {int(out['abstain'].sum())}")


if __name__ == "__main__":
    main()
