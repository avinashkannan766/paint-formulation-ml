"""Streamlit app: predict paint properties from a formulation.

Run (from the project folder, after `python train.py`):
    streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from ensure_models import ensure_models
from paint_science import (BINDER_TYPES, COMPONENTS, LABELS, build_features, derive,
                           finish_from_gloss)

BASE = Path(__file__).resolve().parent

st.set_page_config(page_title="Paint property predictor", layout="wide")


@st.cache_resource(show_spinner="First start: training the models. This takes a minute or two...")
def prepare_models():
    return ensure_models()


prepare_models()  # trains the models once if models/ is missing (e.g. on a fresh deployment)


@st.cache_resource
def load_models():
    meta_path = BASE / "models" / "meta.json"
    if not meta_path.exists():
        return None, None
    meta = json.loads(meta_path.read_text())
    models = {t: joblib.load(BASE / "models" / f"{t}.joblib") for t in meta["targets"]}
    return meta, models


meta, models = load_models()

st.title("Paint property predictor")
st.caption(
    "Predicts viscosity, gloss, opacity, drying time and finish from a formulation. "
    "Trained on SYNTHETIC data, so use it to learn the workflow, not to make real "
    "formulation decisions."
)

if meta is None:
    st.error("No trained models found. Run `python generate_data.py` and `python train.py` first.")
    st.stop()

# --------------------------------------------------------------------------- inputs
DEFAULTS = {
    "binder": 30.0, "tio2": 12.0, "caco3": 8.0, "talc": 2.0, "clay": 0.0, "silica": 0.0,
    "hec": 0.25, "hase": 0.30, "dispersant": 0.48, "defoamer": 0.30, "biocide": 0.15,
    "coalescent": 1.5,
}
STEPS = {"hec": 0.01, "hase": 0.01, "dispersant": 0.01, "defoamer": 0.01, "biocide": 0.01,
         "coalescent": 0.1}

st.sidebar.header("Formulation (kg per 100 kg paint)")
recipe = {}
for c in COMPONENTS:
    if c == "water":
        continue
    recipe[c] = st.sidebar.number_input(
        LABELS[c], min_value=0.0, max_value=100.0, value=DEFAULTS[c],
        step=STEPS.get(c, 0.5), format="%.2f", key=c)

water = 100.0 - sum(recipe.values())
st.sidebar.metric("Water (balance to 100 kg)", f"{water:.2f} kg")

st.sidebar.header("Binder and process")
binder_type = st.sidebar.selectbox("Binder type", BINDER_TYPES, key="binder_type")
process = {
    "binder_tg": st.sidebar.number_input("Binder Tg (°C)", value=15.0, step=1.0, key="binder_tg"),
    "extender_psize_um": st.sidebar.number_input("Extender particle size (µm)", value=3.0, step=0.1, key="psize"),
    "dispersion_rpm": st.sidebar.number_input("Dispersion speed (rpm)", value=1500, step=100, key="rpm"),
    "disp_time_min": st.sidebar.number_input("Dispersion time (min)", value=25, step=1, key="dtime"),
    "ph": st.sidebar.number_input("pH", value=8.7, step=0.1, key="ph"),
    "temp_C": st.sidebar.number_input("Application temperature (°C)", value=27.0, step=1.0, key="temp"),
    "rh_pct": st.sidebar.number_input("Relative humidity (%)", value=55, step=1, key="rh"),
    "wet_film_um": st.sidebar.number_input("Wet film thickness (µm)", value=110, step=5, key="film"),
}

if water < 0:
    st.error(f"The ingredients add up to {100 - water:.1f} kg, more than 100 kg. Reduce some amounts.")
    st.stop()

# --------------------------------------------------------------------------- features
row = {"water": water, **recipe, **process, "binder_type": binder_type}
df = pd.DataFrame([row])
X = build_features(df)
dv = derive(df).iloc[0]

# --------------------------------------------------------------------------- predictions
CLIP = {"gloss_60": (0.0, 100.0), "contrast_ratio": (0.0, 1.0)}
preds = {}
for t, cfg in meta["targets"].items():
    raw = float(models[t].predict(X)[0])
    q = meta["intervals"][t]["q90"]
    if cfg["log"]:
        p, lo, hi = np.exp(raw), np.exp(raw - q), np.exp(raw + q)
    else:
        p, lo, hi = raw, raw - q, raw + q
    if t in CLIP:
        p, lo, hi = (float(np.clip(v, *CLIP[t])) for v in (p, lo, hi))
    preds[t] = (p, lo, hi)

# --------------------------------------------------------------------------- warnings
outside = []
for col, (lo_r, hi_r) in meta["train_ranges"].items():
    v = row[col]
    if v < lo_r - 1e-9 or v > hi_r + 1e-9:
        outside.append(f"{LABELS.get(col, col)}: {v:g} (training range {lo_r:g} to {hi_r:g})")
if outside:
    st.warning(
        "Some inputs are outside the range the model was trained on, so the predictions "
        "may be unreliable:\n\n" + "\n".join(f"- {o}" for o in outside)
    )

# --------------------------------------------------------------------------- results
st.subheader("Predicted properties")
fmt = {
    "viscosity_ku": ("Viscosity", "{:.0f}", "KU"),
    "gloss_60": ("Gloss 60°", "{:.0f}", "GU"),
    "contrast_ratio": ("Contrast ratio (opacity)", "{:.3f}", ""),
    "touch_dry_min": ("Touch-dry time", "{:.0f}", "min"),
}
cols = st.columns(5)
for col, (t, (label, f, unit)) in zip(cols, fmt.items()):
    p, lo, hi = preds[t]
    col.metric(label, f"{f.format(p)} {unit}".strip())
    col.caption(f"90% range: {f.format(lo)} to {f.format(hi)}")

gloss_p, gloss_lo, gloss_hi = preds["gloss_60"]
finish_mid = str(finish_from_gloss(gloss_p))
finish_set = {str(finish_from_gloss(gloss_lo)), str(finish_from_gloss(gloss_hi)), finish_mid}
cols[4].metric("Finish", finish_mid)
if len(finish_set) > 1:
    cols[4].caption("Could also be: " + ", ".join(sorted(finish_set - {finish_mid})))
else:
    cols[4].caption("Within the 90% range")

st.subheader("Formulation science")
m = st.columns(6)
m[0].metric("PVC", f"{dv['pvc']:.3f}")
m[1].metric("CPVC (estimated)", f"{dv['cpvc']:.3f}")
m[2].metric("PVC / CPVC", f"{dv['pvc_cpvc']:.2f}")
m[3].metric("Volume solids", f"{dv['vol_solids'] * 100:.1f} %")
m[4].metric("VOC (coalescent)", f"{dv['voc_gpl']:.1f} g/L")
m[5].metric("Cost (illustrative)", f"₹{dv['cost_per_L']:.0f} /L")

with st.expander("Recipe sheet"):
    sheet = pd.DataFrame({"Ingredient": [LABELS[c] for c in COMPONENTS],
                          "kg / 100 kg": [round(row[c], 3) for c in COMPONENTS]})
    st.dataframe(sheet, hide_index=True)

with st.expander("How to read these numbers"):
    st.markdown(
        "- **90% range:** in cross-validation about 90% of predictions landed within this "
        "distance of the measured value. It is a rough guide with a fixed width, not a "
        "guarantee.\n"
        "- **PVC / CPVC:** how close the paint is to running out of binder. Near or above 1 "
        "the film turns matt and porous, so gloss collapses.\n"
        "- **Finish:** taken from the predicted 60° gloss (Matt < 8, Satin 8-30, "
        "Semi-gloss 30-65, Gloss > 65 GU). 'Could also be' appears when the range crosses a boundary.\n"
        "- **Data:** the models were trained on simulated data. Cost and VOC use illustrative values."
    )
