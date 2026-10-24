
#IMPORTS
import time
import warnings
import json
from pathlib import Path
import os
import pickle
import tempfile
import multiprocessing as mp
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

#METRICS/BOUNDS
timeLimit = 52.0            # requested budget; checks occur between fits
selectionFraction = 0.45    # stop trying new candidates after this share of the limit
CLSmetric = "f1_macro"      # swap to the official metric once the rules are published
RegMetric = "r2"            # e.g. "neg_root_mean_squared_error"
cvRows = 6000               # subsample size used for model selection
rdSeed = 0

#ANALYZE COLUMNS
"""
_coerce_numeric function:


*   changes date and time into numbers of time units
*   true/false columns become 1 0
*   every number that is stored as text is now stored as a number
*   empty values such as "?" and "n/a" are now NaN
"""
def _coerce_numeric(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        out = s.astype("int64").astype("float64") / 1e9
        out[s.isna()] = np.nan
        return out
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    out = pd.to_numeric(s, errors="coerce").astype("float64")
    return out.replace([np.inf, -np.inf], np.nan)

"""
analyze_columns function:

Classify every column as:

*   numeric
*   categorical
*   constant
*   duplicate
*   id_like
*   empty

"""
def analyze_columns(X):
    n = len(X)
    info, seen = {}, {}
    for c in X.columns:
        s = X[c]
        miss = float(s.isna().mean()) if n else 0.0
        entry = {"dtype": str(s.dtype), "missing_frac": round(miss, 4),
                 "n_unique": int(s.nunique(dropna=True)), "kind": None, "dropped": False,
                 "reason": None}
        numeric_like = pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s)
        if not numeric_like and s.notna().any():
            conv = pd.to_numeric(s, errors="coerce")
            if conv.notna().sum() >= 0.95 * s.notna().sum():
                numeric_like = True
        if entry["n_unique"] <= 1:
            entry.update(kind="empty" if miss == 1 else "constant", dropped=True,
                         reason="no information")
        else:
            key = (pd.util.hash_pandas_object(s.astype(str), index=False).sum(), entry["n_unique"])
            if key in seen and X[seen[key]].astype(str).equals(s.astype(str)):
                entry.update(kind="duplicate", dropped=True, reason=f"copy of {seen[key]}")
            else:
                seen[key] = c
                if numeric_like:
                    entry["kind"] = "numeric"
                else:
                    ratio = entry["n_unique"] / max(n, 1)
                    if n > 50 and ratio > 0.9:
                        entry.update(kind="id_like", dropped=True, reason="unique per row")
                    else:
                        entry["kind"] = "categorical"
        if entry["kind"] == "numeric" and not pd.api.types.is_datetime64_any_dtype(s):
            conv = _coerce_numeric(s)
            # unique integers spanning about n values behave like a row id (even if shuffled)
            if (n > 50 and entry["n_unique"] == n and conv.notna().all()
                    and np.allclose(conv, np.round(conv))
                    and (conv.max() - conv.min() + 1) <= 1.5 * n):
                entry.update(kind="id_like", dropped=True, reason="unique integer per row")
        info[c] = entry
    return info

"""
Cleaner class learns from the training data which columns to keep, then outputs only those, as floats or text. It also does the same for the test data

fit function runs once on the training data by calling "analyze_columns", then saves 2 lists:

1.   the numeric columns to keep
2.   the categorical columns to keep




transform function runs on both, the training data the test data. It picks out the saved columns in the same order:

*   converts the numeric ones to floats using "_coerce_numeric"
*   converts the categorical ones into text with missing values labelled as "missing"

"""
class Cleaner(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.info_ = analyze_columns(X)
        self.num_ = [c for c, i in self.info_.items() if i["kind"] == "numeric"]
        self.cat_ = [c for c, i in self.info_.items() if i["kind"] == "categorical"]

        if not self.num_ and not self.cat_:
            self.use_fallback_ = True
        else:
            self.use_fallback_ = False

        return self

    def transform(self, X):
        if self.use_fallback_:
            return pd.DataFrame(
                {"fallback_feature": np.zeros(len(X))},
                index=X.index
            )
        X = X.reindex(columns=self.num_ + self.cat_)
        out = pd.DataFrame(index=X.index)
        for c in self.num_:
            out[c] = _coerce_numeric(X[c])
        for c in self.cat_:
            s = X[c]
            out[c] = s.astype(str).where(s.notna(), "missing").astype(object)
        return out

"""
make_prep function builds the full preparation pipeline:

1.   runs Cleaner
2.   fills missing numbers with the median and flags them
3.   scales numeric columns
4.   one-hot encodes categorical columns
"""

def make_prep(scale=True):
    num = [("imp", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=False))]
    if scale:
        num.append(("sc", StandardScaler()))
    cat = [("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=25,
                                sparse_output=False))]
    ct = ColumnTransformer(
        [("num", Pipeline(num), make_column_selector(dtype_include=np.number)),
         ("cat", Pipeline(cat), make_column_selector(dtype_include=object))],
        remainder="drop", verbose_feature_names_out=False)
    return Pipeline([("clean", Cleaner()), ("ct", ct)])

"""
*build_profile* function assembles the structured data profile:

*   row and column counts
*   missing values
*   column types
*   constants
*   duplicates
*   ID columns
*   class balance/skew
"""

def build_profile(X, task, y=None, n_clusters=None):
    info = analyze_columns(X)
    prof = {
        "task": task,
        "n_rows": int(len(X)),
        "n_columns": int(X.shape[1]),
        "n_features_used": int(sum(not i["dropped"] for i in info.values())),
        "missing_cells": int(X.isna().sum().sum()),
        "columns_with_missing": [c for c, i in info.items() if i["missing_frac"] > 0],
        "numeric_columns": [c for c, i in info.items() if i["kind"] == "numeric"],
        "categorical_columns": [c for c, i in info.items() if i["kind"] == "categorical"],
        "constant_columns": [c for c, i in info.items() if i["kind"] in ("constant", "empty")],
        "duplicate_columns": [c for c, i in info.items() if i["kind"] == "duplicate"],
        "id_like_columns": [c for c, i in info.items() if i["kind"] == "id_like"],
        "duplicate_rows": int(X.duplicated().sum()),
        "columns": info,
    }
    if y is not None and task == "classification":
        vc = pd.Series(y).value_counts(normalize=True)
        prof["n_classes"] = int(len(vc))
        prof["class_balance"] = {str(k): round(float(v), 4) for k, v in vc.items()}
        prof["imbalance_ratio"] = round(float(vc.max() / vc.min()), 3)
    if y is not None and task == "regression":
        yy = pd.to_numeric(pd.Series(y), errors="coerce")
        prof["target_mean"], prof["target_std"] = float(yy.mean()), float(yy.std())
        prof["target_skew"] = float(yy.skew())
    if task == "clustering":
        prof["n_clusters"] = int(n_clusters) if n_clusters else None
    return prof

"""
*infer_task* function guesses the task if it is not specified.

A given cluster count means clustering, a numeric target with many distinct

*   if given a clust count -> clustering
*   a numeric target with many distinct values -> linear regression
*   else, it chooses classification
"""
def infer_task(y=None, n_clusters=None):
    if n_clusters is not None or y is None:
        return "clustering"
    y = pd.Series(y)
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20 and y.nunique() / len(y) > 0.05:
        return "regression"
    return "classification"

#CANDIDATE MODELS

"""
_candidates function lists the models to try for the task.

*   fastest model comes first
*   use class weighting for imbalanced data
*   log-target variant for skewed data
"""
def _candidates(task, n_rows, imbalance, skew, rdSeed):
    cw = "balanced" if imbalance and imbalance > 3 else None
    jobs = 1
    if task == "classification":
        candidates = []

        # Compare regularization strengths and class-weight choices.
        # Default arguments freeze each lambda's settings correctly.
        settings = [
            (1.0, None),
            (1.0, "balanced"),
            (0.1, None),
            (0.1, "balanced"),
            (10.0, None),
            (10.0, "balanced"),
        ]

        for strength, weight in settings:
            label = "balanced" if weight else "unweighted"
            name = f"logreg_C{strength}_{label}"

            candidates.append((
                name,
                lambda strength=strength, weight=weight: Pipeline([
                    ("p", make_prep(True)),
                    ("m", LogisticRegression(
                        C=strength,
                        class_weight=weight,
                        max_iter=1000,
                        random_state=rdSeed,
                    )),
                ]),
            ))

        # Preserve the original boosting candidate.
        candidates.append((
            "hgb",
            lambda: Pipeline([
                ("p", make_prep(False)),
                ("m", HistGradientBoostingClassifier(
                    class_weight=cw,
                    random_state=rdSeed,
                )),
            ]),
        ))

        # Compare a more regularized boosting model.
        candidates.append((
            "hgb_regularized",
            lambda: Pipeline([
                ("p", make_prep(False)),
                ("m", HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=200,
                    max_leaf_nodes=15,
                    min_samples_leaf=25,
                    l2_regularization=5.0,
                    early_stopping=True,
                    random_state=rdSeed,
                )),
            ]),
        ))

        candidates.append((
            "rf",
            lambda: Pipeline([
                ("p", make_prep(False)),
                ("m", RandomForestClassifier(
                    n_estimators=200,
                    n_jobs=jobs,
                    class_weight=cw,
                    min_samples_leaf=1,
                    random_state=rdSeed,
                )),
            ]),
        ))
        candidates.append((
            "extra_trees",
            lambda: Pipeline([
                ("p", make_prep(False)),
                ("m", ExtraTreesClassifier(
                    n_estimators=150,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight=cw,
                    n_jobs=1,
                    random_state=rdSeed,
                )),
            ]),
        ))

        # Test a diverse set of models earlier.
        priority = {
            "logreg_C1.0_unweighted": 0,
            "hgb": 1,
            "extra_trees": 2,
            "logreg_C1.0_balanced": 3,
            "hgb_regularized": 4,
            "rf": 5,
        }

        candidates.sort(
            key=lambda item: priority.get(item[0], 10)
        )
        return candidates
    cands = [
        ("ridge", lambda: Pipeline([("p", make_prep(True)), ("m", RidgeCV(alphas=np.logspace(-3, 3, 13)))])),
        ("hgb", lambda: Pipeline([("p", make_prep(False)),
                                  ("m", HistGradientBoostingRegressor(random_state=rdSeed))])),
        ("rf", lambda: Pipeline([("p", make_prep(False)),
                                 ("m", RandomForestRegressor(200, n_jobs=jobs, random_state=rdSeed))])),
    ]
    if skew is not None and abs(skew) > 1:
        cands.insert(2, ("hgb_log", lambda: TransformedTargetRegressor(
            Pipeline([("p", make_prep(False)), ("m", HistGradientBoostingRegressor(random_state=rdSeed))]),
            func=np.log1p, inverse_func=np.expm1, check_inverse=False)))
    return cands

"""
fit_supervised function cross-validates each candidate on a sample
within the time budget, then refits the best one on all the data
"""

def _default(value, name):
      return globals()[name] if value is None else value


class RegressionBlend:
    """Refitted estimators with weights selected using out-of-fold predictions."""
    def __init__(self, models, weights):
        self.models = models
        self.weights = np.asarray(weights, dtype=float)

    def predict(self, X):
        return sum(w * m.predict(X) for w, m in zip(self.weights, self.models))


def regression_candidates(X, y, seed):
    """Small, diverse search; all preprocessing stays inside each CV fold."""
    def pipe(model, scale=False):
        return Pipeline([("p", make_prep(scale)), ("m", model)])

    candidates = [
        ("ridge", pipe(RidgeCV(alphas=np.logspace(-3, 4, 16)), True)),
        ("extra_trees", pipe(ExtraTreesRegressor(
            n_estimators=200, n_jobs=1, random_state=seed))),
        ("hgb", pipe(HistGradientBoostingRegressor(random_state=seed))),
        ("rf", pipe(RandomForestRegressor(
            n_estimators=200, n_jobs=1, random_state=seed))),
        ("extra_trees_smooth", pipe(ExtraTreesRegressor(
            n_estimators=200, max_features=0.8, min_samples_leaf=2,
            n_jobs=1, random_state=seed))),
        ("hgb_extended", pipe(HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
            min_samples_leaf=10, l2_regularization=1.0,
            early_stopping=True, random_state=seed))),
        ("hgb_small_leaves", pipe(HistGradientBoostingRegressor(
            learning_rate=0.07, max_iter=300, max_leaf_nodes=15,
            min_samples_leaf=5, l2_regularization=2.0,
            early_stopping=True, random_state=seed))),
    ]
    # Quadratic scaling makes SVR inappropriate for a large unknown table.
    if len(X) <= 3000 and X.shape[1] <= 60:
        for c in (3.0, 20.0):
            candidates.append((f"svr_C{c:g}", TransformedTargetRegressor(
                regressor=pipe(SVR(C=c, epsilon=0.1, cache_size=128), True),
                transformer=StandardScaler())))
    if np.min(y) >= 0 and pd.Series(y).skew() > 1:
        candidates.append(("hgb_log", TransformedTargetRegressor(
            regressor=pipe(HistGradientBoostingRegressor(
                learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                l2_regularization=1.0, early_stopping=True,
                random_state=seed)),
            func=np.log1p, inverse_func=np.expm1)))
    return candidates


def fit_regression(X, y, deadline, seed, log):
    """OOF model/blend selection. Deadline is checked between fits, not inside fits."""
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    y = np.asarray(y, dtype=float)
    if len(y) < 4 or not np.isfinite(y).all():
        raise ValueError("Regression needs at least four finite labeled rows")
    if RegMetric not in ("r2", "neg_root_mean_squared_error", "neg_mean_squared_error", "neg_mean_absolute_error"):
        raise ValueError("Unsupported regression selection metric")

    def score_predictions(actual, predicted):
        if RegMetric == "r2":
            return float(r2_score(actual, predicted))
        if RegMetric == "neg_mean_absolute_error":
            return -float(mean_absolute_error(actual, predicted))
        error = float(mean_squared_error(actual, predicted))
        return -float(np.sqrt(error) if RegMetric == "neg_root_mean_squared_error" else error)

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), min(len(X), cvRows), replace=False)
    # Keep original order when the whole table fits the selection budget.
    if len(idx) == len(X):
        idx = np.arange(len(X))
    Xs, ys = X.iloc[idx], y[idx]
    folds = list(KFold(3, shuffle=True, random_state=seed).split(Xs))
    candidates = regression_candidates(Xs, ys, seed)
    records, completed = [], []
    for name, estimator in candidates:
        if completed and time.time() >= deadline:
            log(f"  skipping {name}: selection budget used")
            break
        began = time.time()
        oof = np.full(len(ys), np.nan)
        fold_scores = []
        try:
            for train_idx, valid_idx in folds:
                if completed and time.time() >= deadline:
                    raise TimeoutError("selection budget used during CV")
                fitted = clone(estimator).fit(Xs.iloc[train_idx], ys[train_idx])
                prediction = fitted.predict(Xs.iloc[valid_idx])
                if not np.isfinite(prediction).all():
                    raise ValueError("non-finite predictions")
                oof[valid_idx] = prediction
                fold_scores.append(score_predictions(ys[valid_idx], prediction))
            cv_score = float(np.mean(fold_scores))
            if not np.isfinite(cv_score):
                raise ValueError("non-finite validation score")
            row = dict(model=name, cv_score=cv_score,
                       cv_std=float(np.std(fold_scores)),
                       cv_seconds=round(time.time()-began, 2))
            records.append(row)
            completed.append((cv_score, name, estimator, oof, row))
            if cv_score >= max(entry[0] for entry in completed):
                _checkpoint_estimator(estimator, X, y, row, records)
            log(f"  {name}: {RegMetric}={cv_score:.4f} ({row['cv_seconds']:.2f}s)")
        except Exception as error:
            log(f"  {name} skipped: {type(error).__name__}: {error}")
    if not completed:
        model = Pipeline([("m", DummyRegressor(strategy="mean"))]).fit(X, y)
        chosen = dict(model="dummy_mean", cv_score=None, fallback=True)
        return model, chosen, records

    completed.sort(key=lambda entry: entry[0], reverse=True)
    winner = completed[0]
    selected, weights = [winner], [1.0]
    # Blend only strong, distinct model families, with the same folds as singles.
    def family(name):
        return name.split("_")[0]
    for other in completed[1:4]:
        if family(other[1]) == family(winner[1]):
            continue
        for weight in (0.25, 0.5, 0.75):
            prediction = weight*winner[3] + (1-weight)*other[3]
            score = float(np.mean([
                score_predictions(ys[v], prediction[v]) for _, v in folds]))
            row = dict(model=f"blend({winner[1]},{other[1]})",
                       weights=[weight, 1-weight], cv_score=score,
                       cv_seconds=0.0)
            records.append(row)
            # Small minimum gain in R2 avoids adopting near-tie blends.
            gain = 0.002 if RegMetric == "r2" else 0.0
            current = (winner[0] if len(selected)==1 else blend_score)
            if score > current + gain:
                selected, weights, blend_score = [winner, other], [weight, 1-weight], score

    # Fit the best individual model first so a usable result exists.
    model = clone(winner[2]).fit(X, y)
    chosen = dict(winner[4])

    # Only attempt a blend if sufficient time remains.
    remaining = max(
    0.0,
    deadline + (1 - selectionFraction) * timeLimit - time.time()
)

    if len(selected) > 1 and remaining >= 8.0:
        try:
            log("Attempting regression blend")

            second_model = clone(selected[1][2]).fit(X, y)

            model = RegressionBlend(
                [model, second_model],
                weights
            )

            chosen = {
                "model": "blend",
                "components": [r[1] for r in selected],
                "weights": weights,
                "cv_score": blend_score
            }

        except Exception as error:
            log(f"Blend failed, using best individual model: {error}")
    chosen["selection_metric"] = RegMetric
    chosen["selection_rows"] = len(ys)
    chosen["cv_folds"] = 3
    chosen["deadline_enforcement"] = "checks_between_fits"
    return model, chosen, records


def fit_supervised(X, y, task, deadline_select, rdSeed=None, log=print):
  rdSeed = _default(rdSeed, "rdSeed")
  if task == "regression":
      return fit_regression(X, y, deadline_select, rdSeed, log)
  n = len(X)
  prof_y = pd.Series(y)
  imb = None
  skew = None
  if task == "classification":
      vc = prof_y.value_counts()
      imb = float(vc.max() / vc.min())
  else:
      skew = float(prof_y.skew())
      if prof_y.min() <= 0:  # log target only valid for positive values
          skew = None
  cands = _candidates(task, n, imb, skew, rdSeed)

  # subsample for selection
  Xs, ys = X, y
  if n > cvRows:
      strat = y if task == "classification" and pd.Series(y).value_counts().min() >= 2 else None
      Xs, _, ys, _ = train_test_split(X, y, train_size=cvRows, random_state=rdSeed, stratify=strat)
  if task == "classification":
      min_class = pd.Series(ys).value_counts().min()

      if min_class >= 2:
          cv = StratifiedKFold(
              n_splits=min(3, int(min_class)),
              shuffle=True,
              random_state=rdSeed
          )
      else:
          cv = KFold(
              n_splits=min(3, len(ys)),
              shuffle=True,
              random_state=rdSeed
          )

      metric = CLSmetric
  else:
      cv = KFold(3, shuffle=True, random_state=rdSeed)
      metric = RegMetric

  results = []
  for name, factory in cands:
      if results and time.time() > deadline_select:
          log(f"  skipping {name}: selection budget used")
          continue
      t0 = time.time()
      try:
          fold_scores = cross_val_score(
              factory(),
              Xs,
              ys,
              cv=cv,
              scoring=metric,
              n_jobs=1,
              error_score="raise",
          )

          score = float(np.mean(fold_scores))

          if not np.isfinite(score):
              raise ValueError("Cross-validation returned a non-finite score")
      except Exception as e:  # a failing candidate must never kill the run
          log(f"  {name} failed: {type(e).__name__}: {e}")
          continue
      results.append({"model": name, "cv_score": round(score, 4), "cv_seconds": round(time.time() - t0, 2)})
      if score >= max(r["cv_score"] for r in results):
          _checkpoint_estimator(factory(), X, y, results[-1], results)
      log(f"  {name}: {metric}={score:.4f} ({time.time() - t0:.1f}s)")
  if not results:
    from sklearn.dummy import DummyClassifier

    model = DummyClassifier(strategy="most_frequent")
    model.fit(X, y)

    chosen = {
        "model": "dummy_majority",
        "cv_score": None,
        "fallback": True
    }

    return model, chosen, []
    # Select the model with the highest cross-validation score
  winner = max(results, key=lambda r: r["cv_score"])

  # Find the corresponding model factory
  winner_name = winner["model"]
  winner_factory = next(
      factory for name, factory in cands
      if name == winner_name
  )

  # Train the selected model on the complete training dataset
  model = winner_factory()
  model.fit(X, y)

  chosen = dict(winner)
  chosen["selection_metric"] = metric
  chosen["selection_rows"] = len(ys)

  return model, chosen, results

"""
*fit_cluster* function prepares the data, tries k-means, two Gaussian mixture variants
and Ward linkage with the required k, and keeps the one with the best silhouette score
"""
def fit_cluster(X, k, deadline, rdSeed=None, log=print):
    from itertools import combinations
    from sklearn.metrics import pairwise_distances

    rdSeed = _default(rdSeed, "rdSeed")
    n = len(X)
    k = int(k)

    if not 1 <= k <= n:
        raise ValueError(
            "Cluster count must be between 1 and number of rows"
        )

    if k == 1 or k == n:
        labels = (
            np.zeros(n, dtype=int)
            if k == 1 else np.arange(n)
        )
        return labels, {
            "model": "trivial",
            "silhouette": None,
        }, []

    rng = np.random.RandomState(rdSeed)

    # Limit the rows used for candidate fitting and comparison.
    fit_idx = rng.choice(
        n, min(n, max(3000, k)), replace=False
    )
    score_idx = rng.choice(
        n, min(n, max(600, k + 1)), replace=False
    )

    Xfit = X.iloc[fit_idx]

    # Learn which columns need transformation from the fitting sample.
    initial_info = analyze_columns(Xfit)
    log_columns = []

    for c, details in initial_info.items():
        if details["dropped"] or details["kind"] != "numeric":
            continue

        values = _coerce_numeric(Xfit[c]).dropna()

        if (
            len(values) >= 20
            and values.nunique() > 10
            and values.min() >= 0
            and values.skew() > 2.0
        ):
            log_columns.append(c)

    # Apply the same transformation to fitting, scoring and final rows.
    # Preserve raw data for the common evaluation space.
    Xlog = X.copy()

    for c in log_columns:
        values = _coerce_numeric(X[c])
        Xlog[c] = np.log1p(values.clip(lower=0))

    log("Columns eligible for log transformation:", log_columns)

    Xfit = X.iloc[fit_idx]
    Xscore = X.iloc[score_idx]

    info = analyze_columns(Xfit)
    kept = [
        c for c, v in info.items()
        if not v["dropped"]
    ]
    nums = [
        c for c in kept
        if info[c]["kind"] == "numeric"
    ]

    if not kept:
        raise ValueError(
            "No usable features remain after cleaning"
        )

    # Compare all candidates using the same reference distances.
    reference_prep = make_prep(True)
    reference_fit = reference_prep.fit_transform(Xfit[kept])
    reference_score = reference_prep.transform(Xscore[kept])

    distances = pairwise_distances(reference_score)
    np.fill_diagonal(distances, 0.0)

    views = [("all_features", kept)]

    if len(nums) >= 2 and nums != kept:
        views.append(("numeric_features", nums))

    # Avoid exploring binary indicators as continuous feature pairs.
    continuous = [
        c for c in nums
        if info[c]["n_unique"] > max(5, k)
    ]

    if 2 <= len(continuous) <= 6:
        pairs = list(combinations(continuous, 2))
        rng.shuffle(pairs)
        pairs = pairs[:12]

    elif len(continuous) > 6:
        pairs = []
        seen = set()

        for _ in range(100):
            pair = tuple(sorted(
                rng.choice(
                    len(continuous), 2, replace=False
                )
            ))

            if pair not in seen:
                seen.add(pair)
                pairs.append(tuple(
                    continuous[j] for j in pair
                ))

            if len(pairs) == 12:
                break

    else:
        pairs = []

    seen_views = {
        tuple(columns) for _, columns in views
    }

    for pair in pairs:
        if tuple(pair) not in seen_views:
            views.append(("pair", list(pair)))
            seen_views.add(tuple(pair))

    candidates = []

    for view_name, columns in views:
        candidates.append(("raw", view_name, columns))

        if any(c in log_columns for c in columns):
            candidates.append(("log", view_name, columns))

    best = None
    results = []

    for mode, view_name, columns in candidates:
        if best is not None and time.time() >= deadline:
            break

        try:
            source = X if mode == "raw" else Xlog

            if mode == "raw" and columns == kept:
                prep = reference_prep
                A = reference_fit
                B = reference_score
            else:
                prep = make_prep(True)
                A = prep.fit_transform(
                    source.iloc[fit_idx][columns]
                )
                B = prep.transform(
                    source.iloc[score_idx][columns]
                )

            models = [
                ("kmeans", KMeans(
                    n_clusters=k,
                    n_init=10,
                    random_state=rdSeed,
                ))
            ]

            # Additional algorithms on the complete representation.
            if (
                view_name == "all_features"
                and k <= 20
                and A.shape[1] <= 40
            ):
                models.extend([
                    ("gmm_full", GaussianMixture(
                        n_components=k,
                        covariance_type="full",
                        n_init=1,
                        random_state=rdSeed,
                    )),
                    ("gmm_diag", GaussianMixture(
                        n_components=k,
                        covariance_type="diag",
                        n_init=1,
                        random_state=rdSeed,
                    )),
                ])

            for name, model in models:
                if (
                    best is not None
                    and time.time() >= deadline
                ):
                    break

                started = time.time()
                model.fit(A)
                labels = model.predict(B)

                if not 1 < len(np.unique(labels)) < len(labels):
                    continue

                # Used to choose the winner.
                score = float(silhouette_score(
                    distances,
                    labels,
                    metric="precomputed",
                ))

                # Diagnostic only: separation in selected features.
                own_score = float(
                    silhouette_score(B, labels)
                )

                row = {
                    "model": name,
                    "preprocessing": mode,
                    "view": view_name,
                    "selected_columns": list(columns),
                    "silhouette": round(score, 4),
                    "selected_space_silhouette": round(
                        own_score, 4
                    ),
                    "fit_rows": len(fit_idx),
                    "seconds": round(
                        time.time() - started, 2
                    ),
                }

                results.append(row)

                log(
                    f"  {mode} {name} {list(columns)}: "
                    f"common silhouette={score:.4f}, "
                    f"selected-space={own_score:.4f}"
                )

                # Choose using the common-space silhouette.
                if best is None or score > best[0]:
                    best = (
                        score, model, prep,
                        list(columns), row,
                    )
                    if _WORKER_CONTEXT is not None:
                        full_labels = np.concatenate([
                            model.predict(prep.transform(source.iloc[s:s+512][columns]))
                            for s in range(0, n, 512)
                        ])
                        _publish_predictions(full_labels, row, results)

        except (
            ValueError,
            TypeError,
            MemoryError,
            np.linalg.LinAlgError,
        ) as error:
            log(f"  {view_name} failed: {error}")

    if best is None:
        raise ValueError(
            "No clustering candidate succeeded"
        )

    _, model, prep, columns, chosen = best

    if chosen["preprocessing"] == "log":
        X = Xlog

    log("Selected preprocessing:", chosen["preprocessing"])

    # Return predictions for every row, using small batches.
    labels = np.concatenate([
        model.predict(
            prep.transform(
                X.iloc[start:start + 512][columns]
            )
        )
        for start in range(0, n, 512)
    ])

    # Ensure the required number of nonempty clusters.
    unique_labels = np.unique(labels)

    if len(unique_labels) != k:
        log(f"Expected {k} clusters, got {len(unique_labels)}. Using fallback.")

        fallback_prep = make_prep(True)
        fallback_data = fallback_prep.fit_transform(X)

        fallback = KMeans(
            n_clusters=k,
            n_init=3,
            random_state=rdSeed
        )

        labels = fallback.fit_predict(fallback_data)

        # KMeans can still return fewer groups when rows are identical.
        if len(np.unique(labels)) != k:
            labels = np.asarray(labels).copy()

            # Reserve one row for each required cluster.
            labels[:k] = np.arange(k)

            # Map remaining labels into the required range.
            labels[k:] = np.asarray(labels[k:], dtype=int) % k

        chosen = {
            "model": "kmeans_fallback",
            "requested_clusters": k,
            "returned_clusters": len(np.unique(labels))
        }

    log("Selected columns:", columns)

    return labels, chosen, results


#TESTING THE DATA

"""
run function detects the task, profiles the data, trains or clusters, and returns the predictions, the profile and a record of which models it tried

*   train: DataFrame (with "target" column for supervised tasks)
*   test: DataFrame of rows to predict (supervised)
"""
def _run_impl(train, test=None, task=None, target=None, n_clusters=None, timeLimit=timeLimit, rdSeed=rdSeed, verbose=True):
    t_start = time.time()
    log = print if verbose else (lambda *a, **k: None)
    deadline_select = t_start + selectionFraction * timeLimit

    y = train[target] if target is not None and target in train.columns else None
    task = task or infer_task(y, n_clusters)
    log(f"task={task}")

    if task == "clustering":
        data = test if test is not None else train
        data = data.drop(columns=[target], errors="ignore")
        labels, best, results = fit_cluster(data, n_clusters, deadline_select, rdSeed, log)
        prof = build_profile(data, task, n_clusters=n_clusters)
        info = {"chosen": best, "tried": results}
        preds = labels
    else:
        if task == "regression":
            y = pd.to_numeric(y, errors="coerce")
            keep = pd.Series(
                np.isfinite(y.to_numpy(dtype=float)),
                index=y.index
            )
        else:
            keep = y.notna()
        X = train.drop(columns=[target]).loc[keep].reset_index(drop=True)
        y = y.loc[keep].reset_index(drop=True)
        if task == "classification":
            le = LabelEncoder()
            y_fit = le.fit_transform(y.astype(str) if y.dtype == object else y)
        else:
            y_fit = pd.to_numeric(y, errors="coerce").to_numpy()
        prof = build_profile(X, task, y=y)
        if task == "classification" and len(np.unique(y_fit)) == 1:
            from sklearn.dummy import DummyClassifier

            model = DummyClassifier(strategy="most_frequent")
            model.fit(X, y_fit)

            best = {
                "model": "dummy_single_class",
                "cv_score": None
            }
            results = []
        else:
            model, best, results = fit_supervised(
                X, y_fit, task, deadline_select, rdSeed, log
            )
        X_new = (test.drop(columns=[target], errors="ignore") if test is not None else X)
        preds = model.predict(X_new)
        if task == "classification":
            preds = le.inverse_transform(np.asarray(preds).astype(int))
        info = {"chosen": best, "tried": results}

    info["seconds"] = round(time.time() - t_start, 2)
    log(f"done in {info['seconds']}s")
    return {"predictions": np.asarray(preds), "profile": prof, "model_info": info}

_WORKER_CONTEXT = None


def _atomic_save(path, value):
    temporary = str(path) + ".writing"
    with open(temporary, "wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _publish_predictions(predictions, chosen, tried):
    context = _WORKER_CONTEXT
    if context is None:
        return
    _atomic_save(context["checkpoint"], {
        "predictions": np.asarray(predictions),
        "profile": context["profile"],
        "model_info": {"chosen": dict(chosen), "tried": list(tried),
                       "checkpoint": True},
    })


def _checkpoint_estimator(estimator, X, y, chosen, tried):
    """A completed CV winner is refitted and checkpointed before more searching."""
    context = _WORKER_CONTEXT
    if context is None:
        return
    model = clone(estimator).fit(X, y)
    predictions = model.predict(context["X_new"])
    if context.get("encoder") is not None:
        predictions = context["encoder"].inverse_transform(
            np.asarray(predictions).astype(int))
    _publish_predictions(predictions, chosen, tried)


def _fallback_result(train, test, task, target, k, profile=None):
    """Explicit emergency predictions, used only until a trained checkpoint exists."""
    if task == "clustering":
        data = test if test is not None else train
        if k is None or not 1 <= int(k) <= len(data):
            raise ValueError("A valid cluster count is required")
        predictions = np.arange(len(data)) % int(k)
        model = "emergency_row_partition"
    else:
        y = train[target].dropna()
        if task == "regression":
            y = pd.to_numeric(y, errors="coerce")
            y = y[np.isfinite(y)]
        if not len(y):
            raise ValueError("No usable labels")
        count = len(test) if test is not None else len(y)
        value = float(y.mean()) if task == "regression" else y.mode().iloc[0]
        predictions = np.full(count, value)
        model = "emergency_mean" if task == "regression" else "emergency_majority"
    return {"predictions": predictions,
            "profile": profile if profile is not None else {
                "task": task, "n_rows": len(train),
                "profiling_complete": False},
            "model_info": {"chosen": {"model": model, "fallback": True}, "tried": []}}


def _budget_worker(train, test, task, target, k, budget, seed, verbose, checkpoint, error_path):
    global _WORKER_CONTEXT
    try:
        if task == "clustering":
            data = (test if test is not None else train).drop(columns=[target], errors="ignore")
            profile = build_profile(data, task, n_clusters=k)
            encoder = None
            X_new = None
        else:
            y = train[target]
            keep = y.notna()
            X = train.drop(columns=[target]).loc[keep].reset_index(drop=True)
            y = y.loc[keep].reset_index(drop=True)
            profile = build_profile(X, task, y=y)
            encoder = None
            if task == "classification":
                encoder = LabelEncoder().fit(y.astype(str) if y.dtype == object else y)
            X_new = test.drop(columns=[target], errors="ignore") if test is not None else X
        _WORKER_CONTEXT = dict(checkpoint=checkpoint, profile=profile,
                               encoder=encoder, X_new=X_new)
        _atomic_save(checkpoint, _fallback_result(train, test, task, target, k, profile))
        result = _run_impl(train, test, task, target, k, budget, seed, verbose)
        result["model_info"]["checkpoint"] = False
        _atomic_save(checkpoint, result)
    except BaseException as error:
        _atomic_save(error_path, f"{type(error).__name__}: {error}")


def run(train, test=None, task=None, target=None, n_clusters=None,
        timeLimit=timeLimit, rdSeed=rdSeed, verbose=True):
    """Supervise all training in a terminable process (Windows spawn compatible).

    timeLimit covers this call's wall time, not CSV loading or Python imports.
    A return/cleanup margin is reserved. OS scheduling and serialization prevent
    an absolute real-time guarantee; use 55 seconds under a 60-second runner.
    Call from a .py file with the usual if __name__ == '__main__' guard.
    """
    began = time.monotonic()
    budget = float(timeLimit)
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("timeLimit must be finite and positive")
    y = train[target] if target is not None and target in train.columns else None
    task = task or infer_task(y, n_clusters)
    if task not in ("classification", "regression", "clustering"):
        raise ValueError("Unknown task")
    if task != "clustering" and y is None:
        raise ValueError("A valid target column is required")
    # Reserve time for terminating, loading the completed checkpoint and return.
    worker_deadline = began + max(0.0, budget - min(2.0, budget * 0.15))
    context = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="automl_budget_") as folder:
        checkpoint = str(Path(folder) / "completed.pkl")
        error_path = str(Path(folder) / "error.pkl")
        process = context.Process(target=_budget_worker, args=(
            train, test, task, target, n_clusters, budget, rdSeed, verbose,
            checkpoint, error_path))
        # Bound numerical-library threads in the child; use only standard library.
        names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        previous = {name: os.environ.get(name) for name in names}
        try:
            for name in names:
                os.environ[name] = "1"
            process.start()
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        timed_out = False
        try:
            process.join(max(0.0, worker_deadline-time.monotonic()))
            if process.is_alive():
                timed_out = True
                process.terminate()
                process.join(timeout=0.5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=0.5)
            if Path(error_path).exists() and not timed_out:
                with open(error_path, "rb") as handle:
                    raise RuntimeError(pickle.load(handle))
            if Path(checkpoint).exists():
                with open(checkpoint, "rb") as handle:
                    result = pickle.load(handle)
            elif not timed_out:
                raise RuntimeError(f"Training worker exited with code {process.exitcode}")
            else:
                result = _fallback_result(train, test, task, target, n_clusters)
        finally:
            if process.is_alive():
                process.kill()
                process.join(timeout=0.5)
            process.close()
    info = result["model_info"]
    info.update(timed_out=timed_out, seconds=round(time.monotonic()-began, 2),
                time_budget_seconds=budget, deadline_enforcement="process_supervisor")
    info["chosen"]["deadline_enforcement"] = "process_supervisor"
    if verbose and timed_out:
        print("Time limit reached; returning latest completed predictions.", flush=True)
    if verbose and info["chosen"].get("fallback"):
        print("WARNING: no trained model completed; emergency predictions returned.", flush=True)
    return result


if __name__ == "__main__":
    mp.freeze_support()
    import argparse
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        mean_squared_error,
        mean_absolute_error,
        r2_score,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--time-limit", type=float, default=52.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", help="Optional JSON predictions, profile and model details")
    parser.add_argument(
        "--task",
        required=True,
        choices=["classification", "regression", "clustering"],
    )
    parser.add_argument("--target")
    parser.add_argument("--clusters", type=int)
    parser.add_argument(
        "--features",
        nargs="+",
        help="Optional feature columns to use",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.csv, sep=None, engine="python")
    if args.features:
        selected = list(args.features)

        # Keep the target for supervised training and evaluation.
        if args.task != "clustering" and args.target:
            selected.append(args.target)

        selected = list(dict.fromkeys(selected))
        missing = [c for c in selected if c not in data.columns]

        if missing:
            parser.error(f"Columns not found: {missing}")

        data = data[selected].copy()

    if args.task == "clustering":
        if args.clusters is None:
            parser.error("--clusters is required for clustering")

        started = time.perf_counter()

        result = run(
            data,
            task="clustering",
            target=args.target,
            n_clusters=args.clusters,
            timeLimit=args.time_limit,
            rdSeed=args.seed,
        )

        chosen = result["model_info"]["chosen"]

        print("\nCLUSTERING RESULTS")
        print("Chosen model:", chosen["model"])
        print("Selected features:", chosen.get("selected_columns", []))

        labels, counts = np.unique(
            result["predictions"], return_counts=True
        )

        print("\nCluster sizes:")
        for label, count in zip(labels, counts):
            percentage = 100 * count / counts.sum()
            print(
                f"  Cluster {label}: "
                f"{count} customers ({percentage:.1f}%)"
            )

        print(
            f"Clusters returned: {len(labels)} "
            f"(requested: {args.clusters})"
        )
        # Supports either version of the score names.
        common = chosen.get(
            "common_space_silhouette",
            chosen.get("silhouette"),
        )
        selected = chosen.get(
            "selected_space_silhouette",
            chosen.get("silhouette"),
        )

        if common is not None:
            print(f"All-feature silhouette:      {common:.4f}")
        if selected is not None:
            print(f"Selected-feature silhouette: {selected:.4f}")

    else:
        if not args.target or args.target not in data.columns:
            parser.error("--target must name a column in your CSV")

        # Remove rows without usable labels.
        data = data.loc[data[args.target].notna()].copy()

        if args.task == "regression":
            data[args.target] = pd.to_numeric(
                data[args.target], errors="coerce"
            )
            data = data.loc[
                np.isfinite(data[args.target])
            ].copy()

        if len(data) < 5:
            parser.error("At least five labeled rows are required")

        # Use stratification when each class can appear in both sets.
        stratify = None

        if args.task == "classification":
            counts = data[args.target].value_counts()
            n_test = int(np.ceil(len(data) * 0.2))

            if (
                counts.min() >= 2
                and n_test >= len(counts)
                and len(data) - n_test >= len(counts)
            ):
                stratify = data[args.target]

        train, test = train_test_split(
            data,
            test_size=0.2,
            random_state=args.seed,
            stratify=stratify,
        )

        print(f"Training rows: {len(train)}")
        print(f"Testing rows: {len(test)}")

        started = time.perf_counter()

        result = run(
            train,
            test.drop(columns=[args.target]),
            task=args.task,
            target=args.target,
            timeLimit=args.time_limit,
            rdSeed=args.seed,
        )

        actual = test[args.target].to_numpy()
        predicted = result["predictions"]

        print("\nHOLDOUT RESULTS")

        if args.task == "classification":
            actual = actual.astype(str)
            predicted = predicted.astype(str)

            print(f"Accuracy: {accuracy_score(actual, predicted):.4%}")
            print(f"Balanced accuracy: {balanced_accuracy_score(actual, predicted):.4%}")
            print(f"Macro F1: {f1_score(actual, predicted, average='macro', zero_division=0):.4%}")

            # Majority and minority accuracy (binary classification)
            classes, counts = np.unique(actual, return_counts=True)

            if len(classes) == 2:
                majority_class = classes[np.argmax(counts)]
                minority_class = classes[np.argmin(counts)]

                majority_accuracy = np.mean(
                    predicted[actual == majority_class] == majority_class
                )
                minority_accuracy = np.mean(
                    predicted[actual == minority_class] == minority_class
                )

                print(f"Majority-class recall ({majority_class}): {majority_accuracy:.4%}")
                print(f"Minority-class recall ({minority_class}): {minority_accuracy:.4%}")

        else:
            print(f"R2: {r2_score(actual, predicted):.4f}")
            print(
                f"RMSE: "
                f"{np.sqrt(mean_squared_error(actual, predicted)):.4f}"
            )
            print(
                f"MAE: "
                f"{mean_absolute_error(actual, predicted):.4f}"
            )

        print("Chosen model:", result["model_info"]["chosen"])

    print(
        f"Runtime: {time.perf_counter() - started:.2f} seconds"
    )
    if args.output:
        def json_value(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(f"Cannot encode {type(value).__name__}")
        Path(args.output).write_text(
            json.dumps(result, indent=2, default=json_value), encoding="utf-8")
        print("Saved:", args.output)
