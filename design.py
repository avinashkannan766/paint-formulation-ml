"""Inverse design: find recipes that meet target properties at minimum cost.

Method (random search - simple, transparent, and robust):
  1. Draw many candidate recipes from the SAME distribution the models were trained on,
     so predictions stay inside the range where the models are reliable.
  2. Predict all four properties for every candidate.
  3. Keep candidates that meet every target. A "safety margin" shrinks the acceptable region
     by a fraction of each model's 90% error range, so recipes that only pass by luck are dropped.
  4. Rank the survivors by ingredient cost.

Try it from the command line:
    python design.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from generate_data import BINDERS, GLOSSY_SHARE, sample_base
from paint_science import COMPONENTS, build_features, derive, finish_from_gloss

BASE = Path(__file__).resolve().parent
PROPS = ["viscosity_ku", "gloss_60", "contrast_ratio", "touch_dry_min"]


def load_models():
    meta = json.loads((BASE / "models" / "meta.json").read_text())
    models = {t: joblib.load(BASE / "models" / f"{t}.joblib") for t in meta["targets"]}
    return meta, models


def draw_candidates(n, fixed, binder_type=None, seed=0):
    """Random recipes + process settings. `fixed` pins things you don't want searched
    (e.g. temp_C, rh_pct, wet_film_um - conditions of use rather than recipe choices)."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        glossy = rng.random() < GLOSSY_SHARE
        row = dict(sample_base(rng, glossy))
        row["water"] = 100.0 - sum(row.values())
        if binder_type:
            bt = binder_type
        elif glossy:
            bt = str(rng.choice(list(BINDERS), p=[0.7, 0.2, 0.1]))
        else:
            bt = str(rng.choice(list(BINDERS)))
        row.update(
            binder_type=bt,
            binder_tg=rng.uniform(*BINDERS[bt]["tg"]),
            extender_psize_um=rng.uniform(1.2, 6.0),
            dispersion_rpm=rng.uniform(800, 2500),
            disp_time_min=rng.uniform(10, 40),
            ph=float(np.clip(rng.normal(8.7, 0.35), 7.8, 9.5)),
            temp_C=rng.uniform(18, 35),
            rh_pct=rng.uniform(35, 80),
            wet_film_um=rng.uniform(80, 150),
        )
        row.update(fixed)
        rows.append(row)
    return pd.DataFrame(rows)


def predict_all(df, meta, models):
    """Point predictions plus the 90% half-width (q) for every property."""
    X = build_features(df)
    preds, q, is_log = {}, {}, {}
    for t, cfg in meta["targets"].items():
        raw = models[t].predict(X)
        preds[t] = np.exp(raw) if cfg["log"] else raw
        q[t] = meta["intervals"][t]["q90"]
        is_log[t] = cfg["log"]
    return pd.DataFrame(preds, index=df.index), q, is_log


def _passes(pred, q, log, lo, hi, margin):
    """True where the prediction meets [lo, hi] even after moving `margin` x q against us."""
    if log:
        lower, upper = pred * np.exp(-margin * q), pred * np.exp(margin * q)
    else:
        lower, upper = pred - margin * q, pred + margin * q
    ok = np.ones(len(pred), dtype=bool)
    if lo is not None:
        ok &= lower >= lo
    if hi is not None:
        ok &= upper <= hi
    return ok


def prepare(fixed, binder_type=None, n_candidates=30000, seed=0, meta=None, models=None):
    """The slow part: draw candidates and predict them. Do it once, then call select() as
    often as you like with different targets."""
    if meta is None:
        meta, models = load_models()
    cand = draw_candidates(n_candidates, fixed, binder_type, seed)
    pred, q, is_log = predict_all(cand, meta, models)
    return dict(cand=cand, pred=pred, q=q, is_log=is_log, d=derive(cand))


def select(pool, targets, margin=0.5, max_voc=None, max_cost=None, top_k=10):
    """The fast part. targets: {property: (lo, hi)} with None for 'no limit'.
    margin: 0 = trust point predictions, 1 = require the whole 90% range to pass.
    Returns (top_k cheapest recipes as a DataFrame, report explaining what limited the search)."""
    cand, pred, q, is_log, d = (pool[k] for k in ("cand", "pred", "q", "is_log", "d"))

    checks = {}
    for t in PROPS:
        lo, hi = targets.get(t, (None, None))
        if lo is not None or hi is not None:
            checks[t] = _passes(pred[t].values, q[t], is_log[t], lo, hi, margin)
    if max_voc is not None:
        checks["voc_gpl"] = (d["voc_gpl"] <= max_voc).values
    if max_cost is not None:
        checks["cost_per_L"] = (d["cost_per_L"] <= max_cost).values

    ok = np.ones(len(cand), dtype=bool)
    for m in checks.values():
        ok &= m
    report = dict(
        n_candidates=len(cand),
        n_feasible=int(ok.sum()),
        pass_rate={k: float(v.mean()) for k, v in checks.items()},
    )

    out = pd.concat([
        cand[COMPONENTS + ["binder_type"]].round(2),
        cand[["binder_tg", "extender_psize_um", "dispersion_rpm", "disp_time_min", "ph",
              "temp_C", "rh_pct", "wet_film_um"]].round(1),
        pred.round(3),
        d[["pvc", "pvc_cpvc", "voc_gpl", "cost_per_L"]].round(3),
    ], axis=1)
    out["finish"] = finish_from_gloss(pred["gloss_60"].values)

    if report["n_feasible"] == 0:
        return out.iloc[0:0], report
    return out[ok].sort_values("cost_per_L").head(top_k).reset_index(drop=True), report


def suggest(targets, fixed, binder_type=None, margin=0.5, max_voc=None, max_cost=None,
            n_candidates=30000, top_k=10, seed=0):
    pool = prepare(fixed, binder_type, n_candidates, seed)
    return select(pool, targets, margin, max_voc, max_cost, top_k)


if __name__ == "__main__":
    targets = {
        "gloss_60": (60, None),
        "contrast_ratio": (0.90, None),
        "viscosity_ku": (80, 95),
        "touch_dry_min": (None, 45),
    }
    fixed = dict(temp_C=27.0, rh_pct=55.0, wet_film_um=110.0)
    res, rep = suggest(targets, fixed, max_voc=30)
    print(f"{rep['n_feasible']} of {rep['n_candidates']} candidates met every target")
    print("Share of candidates passing each requirement on its own:")
    for k, v in rep["pass_rate"].items():
        print(f"  {k:16s} {v * 100:5.1f} %")
    if len(res):
        pd.set_option("display.width", 200, "display.max_columns", 30)
        print(res[["binder", "tio2", "caco3", "talc", "hec", "hase", "binder_type",
                   "viscosity_ku", "gloss_60", "contrast_ratio", "touch_dry_min",
                   "finish", "pvc_cpvc", "cost_per_L"]].head(5))
