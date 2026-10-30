"""Generate a SYNTHETIC paint-formulation dataset.

Properties are simulated from simplified paint-science relationships plus noise:

* Viscosity (KU)  - thickener dose (saturating), pH-dependent HASE efficiency, volume solids,
                    PVC, dispersion quality, temperature.
* Gloss (60 deg)  - sigmoid collapse around CPVC (PVC/CPVC), binder type, coarse extenders,
                    matting silica, dispersion quality.
* Contrast ratio  - TiO2 volume fraction with crowding, extender spacing, dispersion quality,
                    dry-film thickness, air-void hiding above CPVC.
* Touch-dry time  - volatile content, wet film thickness, temperature, humidity, coalescent, Tg.
* Finish class    - thresholds on 60 deg gloss (Matt / Satin / Semi-gloss / Gloss).

The data is built as `n_bases` base recipes x `n_variants` lab variants, so rows from the same
base share a `base_id` - use it for grouped cross-validation.

THIS IS NOT REAL LAB DATA. Use it to build and test the pipeline, then swap in real records
with the same columns.
"""
import argparse
import os

import numpy as np
import pandas as pd

from paint_science import COMPONENTS, EXTENDERS, PROCESS_COLS, derive, finish_from_gloss

BINDERS = {
    "pure_acrylic": dict(tg=(5, 25), gmax=90.0),
    "styrene_acrylic": dict(tg=(10, 35), gmax=84.0),
    "vinyl_acrylic": dict(tg=(5, 20), gmax=74.0),
}
WATER_RANGE = (18.0, 50.0)
VARIANT_SIGMA = {
    "hec": 0.25, "hase": 0.25, "dispersant": 0.20, "defoamer": 0.20,
    "biocide": 0.20, "coalescent": 0.20,
}


def _water_ok(recipe):
    water = 100.0 - sum(recipe.values())
    return WATER_RANGE[0] <= water <= WATER_RANGE[1]


GLOSSY_SHARE = 0.30  # fraction of base recipes drawn from the high-gloss region


def sample_base(rng, glossy=False):
    """glossy=True draws a binder-rich, low-pigment recipe (low PVC/CPVC -> high gloss)."""
    while True:
        r = {}
        if glossy:
            r["binder"] = rng.uniform(35, 52)
            r["tio2"] = rng.uniform(4, 14)
            s = 0.25  # shrink extender ranges so PVC stays low
        else:
            r["binder"] = rng.uniform(22, 50)
            r["tio2"] = rng.uniform(5, 22)
            s = 1.0
        r["caco3"] = rng.uniform(0, 18 * s) if rng.random() < 0.75 else 0.0
        r["talc"] = rng.uniform(1, 1 + 5 * s) if rng.random() < 0.45 else 0.0
        r["clay"] = rng.uniform(1, 1 + 4 * s) if rng.random() < 0.35 else 0.0
        r["silica"] = rng.uniform(0.3, 0.3 + 2.2 * s) if rng.random() < 0.25 else 0.0
        r["hec"] = rng.uniform(0.05, 0.5)
        r["hase"] = rng.uniform(0.0, 0.8)
        ext = r["caco3"] + r["talc"] + r["clay"] + r["silica"]
        ratio = float(np.clip(rng.normal(0.03, 0.012), 0.008, 0.07))  # dosed on pigment
        r["dispersant"] = ratio * (r["tio2"] + 0.4 * ext)
        r["defoamer"] = rng.uniform(0.1, 0.5)
        r["biocide"] = rng.uniform(0.05, 0.25)
        r["coalescent"] = rng.uniform(0.3, 3.0)
        if _water_ok(r):
            return r


def make_variant(rng, base):
    for _ in range(100):
        cand = {
            k: (v * np.exp(rng.normal(0, VARIANT_SIGMA.get(k, 0.10))) if v > 0 else 0.0)
            for k, v in base.items()
        }
        if _water_ok(cand):
            return cand
    return dict(base)


def simulate(d, rng):
    """Vectorised property simulation on a derived-feature table."""
    n = len(d)
    x = d["pvc"] / (d["cpvc"] * d["_cpvc_lot"])  # true PVC/CPVC incl. hidden lot variation

    # dispersion quality 0..1 (energy input x dispersant dose near its optimum)
    dq = 1 - np.exp(-(d["dispersion_rpm"] / 1000.0) * d["disp_time_min"] / 12.0)
    de = np.exp(-(((d["dispersant_ratio"] - 0.035) / 0.02) ** 2))
    q = dq * (0.55 + 0.45 * de)

    # ---- viscosity (KU)
    ph_eff = 1 / (1 + np.exp(-(d["ph"] - 8.0) / 0.35))
    ku = (
        52
        + 36 * d["hec"] / (d["hec"] + 0.18)
        + 22 * d["hase"] / (d["hase"] + 0.25) * ph_eff
        + 70 * (d["vol_solids"] - 0.33)
        + 28 * (d["pvc"] - 0.40)
        + 14 * (1 - q)
        - 0.45 * (d["temp_C"] - 25)
        + rng.normal(0, 2.0, n)
    )
    ku = np.clip(ku, 40, 150)

    # ---- gloss 60 deg
    gmax = d["binder_type"].map({k: v["gmax"] for k, v in BINDERS.items()})
    sig = 1 / (1 + np.exp((x - 0.55) / 0.12))
    gloss = sig * (
        gmax * (0.7 + 0.3 * q)
        - 2.0 * (d["extender_psize_um"] - 2.0)
        - 2.5 * d["silica"]
        + 1.2 * (d["coalescent"] - 1.5)
        - 4.0 * np.maximum(0, d["defoamer"] - 0.25)
        - 0.15 * (d["binder_tg"] - 15)
    )
    gloss = gloss * np.exp(rng.normal(0, 0.06, n)) + rng.normal(0, 0.8, n)
    gloss = np.clip(gloss, 0.5, 95)

    # ---- contrast ratio (opacity)
    vf = d["tio2_vf_dry"]
    eff = vf * np.exp(-2.0 * vf)  # crowding
    boost = 1 + 0.3 * d["ext_share"] * np.clip(1 - (d["extender_psize_um"] - 1.2) / 6, 0, 1)
    eff = eff * boost * (0.6 + 0.4 * q) + 0.12 * np.maximum(0, x - 1.0)
    dft = d["wet_film_um"] * d["vol_solids"]
    cr = 1 - np.exp(-0.6 * eff * dft)
    cr = np.clip(cr + rng.normal(0, 0.007, n), 0.35, 0.995)

    # ---- touch-dry time (min)
    t = (
        24
        * (d["wet_film_um"] / 100) ** 1.3
        * (d["water_total"] / 50) ** 1.4
        * np.exp(-0.04 * (d["temp_C"] - 25))
        * (1 + 0.010 * (d["rh_pct"] - 50))
        * (1 + 0.10 * (d["coalescent"] - 1.5))
        * (1 - 0.006 * (d["binder_tg"] - 15))
        * (1.1 - 0.2 * np.clip(x, 0, 1.2))
        * np.exp(rng.normal(0, 0.08, n))
    )

    return pd.DataFrame({
        "viscosity_ku": ku.round(1),
        "gloss_60": gloss.round(1),
        "contrast_ratio": cr.round(3),
        "touch_dry_min": t.round(1),
        "finish": finish_from_gloss(gloss),
    })


def build_dataset(n_bases, n_variants, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_bases):
        glossy = rng.random() < GLOSSY_SHARE
        if glossy:  # pure acrylic has the highest gloss ceiling
            btype = str(rng.choice(list(BINDERS), p=[0.7, 0.2, 0.1]))
        else:
            btype = str(rng.choice(list(BINDERS)))
        tg = float(rng.uniform(*BINDERS[btype]["tg"]))
        psize = float(rng.uniform(1.2, 6.0))
        base = sample_base(rng, glossy)
        for v in range(n_variants):
            recipe = base if v == 0 else make_variant(rng, base)
            row = {k: round(val, 3) for k, val in recipe.items()}
            row["water"] = round(100.0 - sum(row.values()), 3)
            row.update(
                base_id=b,
                batch_id=f"B{b:03d}-{v}",
                binder_type=btype,
                binder_tg=round(tg, 1),
                extender_psize_um=round(psize * float(np.exp(rng.normal(0, 0.05))), 2),
                dispersion_rpm=int(rng.uniform(800, 2500)),
                disp_time_min=int(rng.uniform(10, 40)),
                ph=round(float(np.clip(rng.normal(8.7, 0.35), 7.8, 9.5)), 2),
                temp_C=round(float(rng.uniform(18, 35)), 1),
                rh_pct=int(rng.uniform(35, 80)),
                wet_film_um=int(rng.uniform(80, 150)),
                _cpvc_lot=float(np.exp(rng.normal(0, 0.02))),  # hidden, dropped before saving
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    d = derive(df)
    targets = simulate(d, rng)

    cols = ["base_id", "batch_id"] + COMPONENTS + ["binder_type"] + PROCESS_COLS
    return pd.concat([df[cols], targets], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_bases", type=int, default=80)
    ap.add_argument("--n_variants", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/paint_formulations.csv")
    a = ap.parse_args()

    df = build_dataset(a.n_bases, a.n_variants, a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)

    print(f"Saved {len(df)} rows ({a.n_bases} base recipes x {a.n_variants} variants) -> {a.out}")
    print(df[["viscosity_ku", "gloss_60", "contrast_ratio", "touch_dry_min"]].describe().round(2))
    print("\nFinish classes:\n", df["finish"].value_counts())


if __name__ == "__main__":
    main()
