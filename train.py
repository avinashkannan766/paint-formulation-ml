"""Train and evaluate property-prediction models for the paint-formulation data.

Run:  python train.py            (after: python generate_data.py)

Outputs
-------
models/<target>.joblib, models/meta.json   - final LightGBM models + metadata for the app
reports/metrics.csv                        - grouped-CV metrics for every model/target
reports/*.png                              - EDA, predicted-vs-actual, confusion, SHAP charts
"""
import json
import os
import warnings

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, r2_score)
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from paint_science import (COMPONENTS, FEATURE_COLUMNS, FINISH_CLASSES, PROCESS_COLS,
                           build_features, derive, finish_from_gloss)

warnings.filterwarnings("ignore")

DATA = "data/paint_formulations.csv"
TARGETS = {
    "viscosity_ku": dict(label="Viscosity", unit="KU", log=False),
    "gloss_60": dict(label="Gloss 60°", unit="GU", log=False),
    "contrast_ratio": dict(label="Contrast ratio", unit="", log=False),
    "touch_dry_min": dict(label="Touch-dry time", unit="min", log=True),
}
# Physics-based monotonic constraints (+1 increasing, -1 decreasing)
CONSTRAINTS = {
    "viscosity_ku": {"hec": 1, "hase": 1},
    "gloss_60": {"pvc_cpvc": -1},
    "contrast_ratio": {"tio2": 1, "wet_film_um": 1},
    "touch_dry_min": {"wet_film_um": 1, "temp_C": -1},
}

# ---- chart style (neutral ink + validated categorical slots 1-3)
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BINDER_STYLE = {"pure_acrylic": (BLUE, "o"), "styrene_acrylic": (ORANGE, "s"), "vinyl_acrylic": (AQUA, "^")}
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#2a78d6", "#184f95"])


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)


def new_fig(nrows, ncols, w, h):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))
    fig.patch.set_facecolor(SURFACE)
    for ax in np.atleast_1d(axes).ravel():
        style_ax(ax)
    return fig, axes


def title(ax, text):
    ax.set_title(text, loc="left", fontsize=10, color=INK, fontweight="bold")


# ---------------------------------------------------------------- models
def model_factories(mc):
    return {
        "ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 20))),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=0),
        "lightgbm": lambda: lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=10,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            monotone_constraints=mc, random_state=0, verbose=-1, n_jobs=-1),
    }


def oof_predict(factory, X, y, groups, splitter):
    oof = np.zeros(len(y))
    for tr, te in splitter.split(X, y, groups):
        m = factory()
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict(X.iloc[te])
    return oof


def fwd(y, log):
    return np.log(y) if log else y


def inv(y, log):
    return np.exp(y) if log else y


# ---------------------------------------------------------------- charts
def eda_plot(d):
    fig, axes = new_fig(2, 2, 10, 7)
    ax = axes[0, 0]
    for bt, (c, mk) in BINDER_STYLE.items():
        s = d[d.binder_type == bt]
        ax.scatter(s.pvc_cpvc, s.gloss_60, s=14, color=c, marker=mk, alpha=0.75,
                   linewidths=0, label=bt.replace("_", " "))
    ax.axvline(1.0, color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("PVC / CPVC")
    ax.set_ylabel("Gloss 60° (GU)")
    title(ax, "Gloss collapses around CPVC")
    leg = ax.legend(frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color(INK2)

    ax = axes[0, 1]
    ax.scatter(d.tio2_vf_dry, d.contrast_ratio, s=14, color=BLUE, alpha=0.7, linewidths=0)
    ax.set_xlabel("TiO2 volume fraction of dry film")
    ax.set_ylabel("Contrast ratio")
    title(ax, "Opacity saturates with TiO2")

    ax = axes[1, 0]
    ax.scatter(d.thickener_total, d.viscosity_ku, s=14, color=BLUE, alpha=0.7, linewidths=0)
    ax.set_xlabel("Total thickener (kg / 100 kg)")
    ax.set_ylabel("Viscosity (KU)")
    title(ax, "Viscosity rises with thickener, with diminishing returns")

    ax = axes[1, 1]
    ax.scatter(d.water_total, d.touch_dry_min, s=14, color=BLUE, alpha=0.7, linewidths=0)
    ax.set_xlabel("Total volatile water (kg / 100 kg)")
    ax.set_ylabel("Touch-dry time (min)")
    title(ax, "Drying slows with more water")
    fig.tight_layout()
    fig.savefig("reports/eda.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def pred_vs_actual_plot(results):
    fig, axes = new_fig(2, 2, 10, 8)
    for ax, (t, (y, p, r2, mae)) in zip(axes.ravel(), results.items()):
        cfg = TARGETS[t]
        lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.9, ls="--")
        ax.scatter(y, p, s=14, color=BLUE, alpha=0.7, linewidths=0)
        unit = f" ({cfg['unit']})" if cfg["unit"] else ""
        ax.set_xlabel(f"Measured{unit}")
        ax.set_ylabel(f"Predicted{unit}")
        title(ax, cfg["label"])
        ax.text(0.04, 0.95, f"R² {r2:.2f}   MAE {mae:.2f}", transform=ax.transAxes,
                va="top", fontsize=9, color=INK2)
    fig.suptitle("Out-of-fold predictions (grouped by base recipe)", x=0.02, ha="left",
                 fontsize=11, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig("reports/predicted_vs_actual.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def confusion_plot(cms):
    fig, axes = new_fig(1, 2, 10, 4.3)
    for ax, (name, cm, acc, f1) in zip(axes, cms):
        ax.grid(False)
        ax.imshow(cm, cmap=BLUES)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > cm.max() * 0.5 else INK)
        ax.set_xticks(range(4), FINISH_CLASSES, rotation=20)
        ax.set_yticks(range(4), FINISH_CLASSES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        title(ax, f"{name}\nacc {acc:.2f}   macro-F1 {f1:.2f}")
    fig.tight_layout()
    fig.savefig("reports/finish_confusion.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def shap_plot(model, X, t):
    sv = shap.TreeExplainer(model).shap_values(X)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False)
    top = imp.head(12)[::-1]
    fig, ax = new_fig(1, 1, 6.5, 4.6)
    ax.barh(top.index, top.values, color=BLUE, height=0.55)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean |SHAP value|")
    title(ax, f"What drives {TARGETS[t]['label'].lower()}")
    fig.tight_layout()
    fig.savefig(f"reports/shap_{t}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return imp


# ---------------------------------------------------------------- main
def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    df = pd.read_csv(DATA)
    X = build_features(df)
    groups = df["base_id"].values
    d = derive(df)
    eda_plot(d)

    gkf = GroupKFold(n_splits=5)
    rkf = KFold(n_splits=5, shuffle=True, random_state=0)

    rows, plot_data, intervals, importances = [], {}, {}, {}
    oof_gloss = None
    for t, cfg in TARGETS.items():
        y = df[t].values.astype(float)
        yt = fwd(y, cfg["log"])
        mc = [CONSTRAINTS[t].get(c, 0) for c in FEATURE_COLUMNS]
        facts = model_factories(mc)

        for name, f in facts.items():
            p = inv(oof_predict(f, X, yt, groups, gkf), cfg["log"])
            rows.append(dict(target=t, model=name, cv="grouped",
                             MAE=mean_absolute_error(y, p), R2=r2_score(y, p)))
            if name == "lightgbm":
                plot_data[t] = (y, p, r2_score(y, p), mean_absolute_error(y, p))
                if cfg["log"]:
                    intervals[t] = dict(type="multiplicative",
                                        q90=float(np.quantile(np.abs(np.log(p) - np.log(y)), 0.9)))
                else:
                    intervals[t] = dict(type="additive", q90=float(np.quantile(np.abs(p - y), 0.9)))
                if t == "gloss_60":
                    oof_gloss = p

        # leakage demo: same model, random split instead of grouped split
        p_rand = inv(oof_predict(facts["lightgbm"], X, yt, groups, rkf), cfg["log"])
        rows.append(dict(target=t, model="lightgbm", cv="random (leaky)",
                         MAE=mean_absolute_error(y, p_rand), R2=r2_score(y, p_rand)))

        # final model on all data
        final = facts["lightgbm"]()
        final.fit(X, yt)
        joblib.dump(final, f"models/{t}.joblib")
        importances[t] = shap_plot(final, X, t)

    pred_vs_actual_plot(plot_data)

    # ---- finish class: direct classifier vs thresholding predicted gloss
    y_cls = pd.Categorical(df["finish"], categories=FINISH_CLASSES).codes
    oof_cls = np.zeros(len(df), dtype=int)
    for tr, te in gkf.split(X, y_cls, groups):
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                 min_child_samples=5, random_state=0, verbose=-1, n_jobs=-1)
        clf.fit(X.iloc[tr], y_cls[tr])
        oof_cls[te] = clf.predict(X.iloc[te])
    thr_cls = pd.Categorical(finish_from_gloss(oof_gloss), categories=FINISH_CLASSES).codes

    cms, finish_scores = [], {}
    for name, pred in [("Direct classifier", oof_cls), ("Thresholded gloss prediction", thr_cls)]:
        acc = accuracy_score(y_cls, pred)
        f1 = f1_score(y_cls, pred, average="macro", labels=[0, 1, 2, 3], zero_division=0)
        cms.append((name, confusion_matrix(y_cls, pred, labels=[0, 1, 2, 3]), acc, f1))
        finish_scores[name] = dict(accuracy=float(acc), macro_f1=float(f1))
    confusion_plot(cms)

    # ---- save reports and metadata
    metrics = pd.DataFrame(rows)
    metrics.to_csv("reports/metrics.csv", index=False)

    raw_cols = COMPONENTS + PROCESS_COLS
    meta = dict(
        features=FEATURE_COLUMNS,
        targets=TARGETS,
        intervals=intervals,
        train_ranges={c: [float(df[c].min()), float(df[c].max())] for c in raw_cols},
        finish_scores=finish_scores,
    )
    with open("models/meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    # ---- console summary
    pd.set_option("display.width", 140)
    print("\n=== Grouped 5-fold CV (original units) ===")
    g = metrics[metrics.cv == "grouped"].pivot(index="target", columns="model", values=["R2", "MAE"])
    print(g.round(3))
    print("\n=== LightGBM: grouped vs random split (R2) ===")
    lg = metrics[metrics.model == "lightgbm"].pivot(index="target", columns="cv", values="R2")
    print(lg.round(3))
    print("\n=== Finish class ===")
    for k, v in finish_scores.items():
        print(f"{k:32s} accuracy {v['accuracy']:.3f}  macro-F1 {v['macro_f1']:.3f}")
    print("\n=== 90% interval half-widths (from out-of-fold residuals) ===")
    for t, v in intervals.items():
        print(f"{t:16s} {v['type']:14s} {v['q90']:.3f}")
    print("\n=== Top SHAP features ===")
    for t, imp in importances.items():
        print(f"{t:16s}", ", ".join(imp.head(5).index))


if __name__ == "__main__":
    main()
