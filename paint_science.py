"""Shared paint-science constants and feature engineering.

Conventions
-----------
* Recipes are expressed as kg of each material per 100 kg of paint (water is the balance).
* Prices are ILLUSTRATIVE (INR/kg) - replace with your own costs.
* This module is used by the data generator, the training script, the predictor and the app,
  so the same formulas are applied everywhere (no train/serve skew).
"""
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- materials
COMPONENTS = [
    "water", "binder", "tio2", "caco3", "talc", "clay", "silica",
    "hec", "hase", "dispersant", "defoamer", "biocide", "coalescent",
]
PIGMENTS = ["tio2", "caco3", "talc", "clay", "silica"]
EXTENDERS = ["caco3", "talc", "clay", "silica"]

LABELS = {
    "water": "Water",
    "binder": "Acrylic emulsion binder (50% solids)",
    "tio2": "TiO2 (rutile)",
    "caco3": "Calcium carbonate",
    "talc": "Talc",
    "clay": "China clay / kaolin",
    "silica": "Matting silica",
    "hec": "Thickener - HEC",
    "hase": "Thickener - HASE",
    "dispersant": "Dispersant",
    "defoamer": "Defoamer",
    "biocide": "Biocide / preservative",
    "coalescent": "Coalescing agent",
}

DENSITY = {  # kg/L
    "water": 1.00, "binder": 1.06, "tio2": 4.10, "caco3": 2.70, "talc": 2.80,
    "clay": 2.60, "silica": 2.00, "hec": 1.00, "hase": 1.05, "dispersant": 1.10,
    "defoamer": 0.95, "biocide": 1.05, "coalescent": 0.95,
}
OIL_ABSORPTION = {"tio2": 18, "caco3": 15, "talc": 35, "clay": 45, "silica": 120}  # g oil / 100 g
PRICE_INR_PER_KG = {  # illustrative
    "water": 0.1, "binder": 110, "tio2": 280, "caco3": 12, "talc": 18, "clay": 25,
    "silica": 350, "hec": 900, "hase": 450, "dispersant": 220, "defoamer": 400,
    "biocide": 600, "coalescent": 300,
}

BINDER_SOLIDS = 0.50       # emulsion solids fraction
POLYMER_DENSITY = 1.08     # kg/L
OIL_DENSITY = 0.93         # kg/L (linseed oil, for oil-absorption CPVC)
LATEX_CPVC_FACTOR = 0.85   # latex CPVC is lower than the oil-absorption estimate

BINDER_TYPES = ["pure_acrylic", "styrene_acrylic", "vinyl_acrylic"]
PROCESS_COLS = [
    "binder_tg", "extender_psize_um", "dispersion_rpm", "disp_time_min",
    "ph", "temp_C", "rh_pct", "wet_film_um",
]
DERIVED_COLS = [
    "density_kgL", "pvc", "cpvc", "pvc_cpvc", "vol_solids", "wt_solids",
    "tio2_vf_dry", "ext_share", "ext_to_tio2", "thickener_total",
    "dispersant_ratio", "water_total",
]
FEATURE_COLUMNS = (
    COMPONENTS + PROCESS_COLS + [f"binder_{b}" for b in BINDER_TYPES] + DERIVED_COLS
)

# --------------------------------------------------------------------------- finish classes
FINISH_CLASSES = ["Matt", "Satin", "Semi-gloss", "Gloss"]
GLOSS_BINS = [8, 30, 65]  # 60-degree gloss units


def finish_from_gloss(gloss):
    idx = np.digitize(np.asarray(gloss, dtype=float), GLOSS_BINS)
    return np.array(FINISH_CLASSES)[idx]


# --------------------------------------------------------------------------- derived features
def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Add formulation-science features (PVC, CPVC, volume solids, VOC, cost, ...)."""
    out = df.copy()
    eps = 1e-9

    vol = {c: out[c] / DENSITY[c] for c in COMPONENTS}
    total_vol = sum(vol.values())
    pig_vol = sum(vol[c] for c in PIGMENTS)
    ext_vol = sum(vol[c] for c in EXTENDERS)
    poly_vol = out["binder"] * BINDER_SOLIDS / POLYMER_DENSITY
    oil_vol = sum(out[c] * OIL_ABSORPTION[c] / 100.0 for c in PIGMENTS) / OIL_DENSITY
    ext_mass = sum(out[c] for c in EXTENDERS)
    pig_mass = ext_mass + out["tio2"]
    batch = out[COMPONENTS].sum(axis=1)

    out["total_vol_L"] = total_vol
    out["pig_vol"] = pig_vol
    out["ext_vol"] = ext_vol
    out["tio2_vol"] = vol["tio2"]
    out["poly_vol"] = poly_vol

    out["density_kgL"] = batch / total_vol
    out["pvc"] = pig_vol / (pig_vol + poly_vol + eps)
    out["cpvc"] = LATEX_CPVC_FACTOR * pig_vol / (pig_vol + oil_vol + eps)
    out["pvc_cpvc"] = out["pvc"] / out["cpvc"]
    out["vol_solids"] = (pig_vol + poly_vol) / total_vol
    out["wt_solids"] = (out["binder"] * BINDER_SOLIDS + pig_mass) / batch
    out["tio2_vf_dry"] = vol["tio2"] / (pig_vol + poly_vol + eps)
    out["ext_share"] = ext_vol / (pig_vol + eps)
    out["ext_to_tio2"] = ext_mass / (out["tio2"] + eps)
    out["thickener_total"] = out["hec"] + out["hase"]
    out["dispersant_ratio"] = out["dispersant"] / (out["tio2"] + 0.4 * ext_mass + eps)
    out["water_total"] = out["water"] + (1 - BINDER_SOLIDS) * out["binder"]

    # business metrics (not used as model inputs)
    out["voc_gpl"] = out["coalescent"] / total_vol * 1000.0
    cost = sum(out[c] * PRICE_INR_PER_KG[c] for c in COMPONENTS)
    out["cost_per_kg"] = cost / batch
    out["cost_per_L"] = out["cost_per_kg"] * out["density_kgL"]
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw recipe/process table -> model feature matrix (fixed column order)."""
    d = derive(df)
    for b in BINDER_TYPES:
        d[f"binder_{b}"] = (d["binder_type"] == b).astype(int)
    return d[FEATURE_COLUMNS]
