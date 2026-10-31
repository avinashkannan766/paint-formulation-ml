"""Streamlit page: suggest recipes that meet target properties at the lowest cost."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

import design
from ensure_models import ensure_models
from paint_science import BINDER_TYPES, COMPONENTS, LABELS

st.set_page_config(page_title="Suggest a recipe", layout="wide")
st.title("Suggest a recipe")
st.caption(
    "Give target properties and limits; the app searches thousands of candidate recipes with "
    "the trained models and shows the cheapest ones that qualify. Models are trained on "
    "SYNTHETIC data, so treat this as a learning tool."
)

@st.cache_resource(show_spinner="First start: training the models. This takes a minute or two...")
def prepare_models():
    return ensure_models()


prepare_models()  # trains the models once if models/ is missing (e.g. on a fresh deployment)

if not (Path(design.BASE) / "models" / "meta.json").exists():
    st.error("No trained models found. Run `python train.py` first.")
    st.stop()


@st.cache_data(show_spinner="Drawing and predicting candidate recipes...")
def get_pool(fixed_items, binder_type, n_candidates, seed):
    return design.prepare(dict(fixed_items), binder_type, n_candidates, seed)


PROPERTY_ROWS = [
    # key, label, unit, default lo, default hi, default on, step, format
    ("gloss_60", "Gloss 60°", "GU", 50.0, 0.0, True, 1.0, "%.0f"),
    ("contrast_ratio", "Contrast ratio", "", 0.90, 0.0, True, 0.01, "%.2f"),
    ("viscosity_ku", "Viscosity", "KU", 80.0, 95.0, True, 1.0, "%.0f"),
    ("touch_dry_min", "Touch-dry time", "min", 0.0, 45.0, True, 1.0, "%.0f"),
]

with st.form("targets"):
    st.subheader("Targets")
    targets_in = {}
    for key, label, unit, lo, hi, on, step, fmt in PROPERTY_ROWS:
        c = st.columns([1.2, 1, 1, 3])
        use = c[0].checkbox(f"{label} {('(' + unit + ')') if unit else ''}", value=on, key=f"use_{key}")
        lo_v = c[1].number_input("min", value=lo, step=step, format=fmt, key=f"lo_{key}", label_visibility="collapsed")
        hi_v = c[2].number_input("max", value=hi, step=step, format=fmt, key=f"hi_{key}", label_visibility="collapsed")
        targets_in[key] = (use, lo_v, hi_v)
    st.caption("Tick the properties to enforce. Enter 0 in min or max for no limit on that side.")

    st.subheader("Limits and conditions")
    c = st.columns(4)
    max_voc = c[0].number_input("Max VOC (g/L)", value=30.0, step=1.0)
    max_cost = c[1].number_input("Max cost (₹/L, 0 = no limit)", value=0.0, step=5.0)
    binder_choice = c[2].selectbox("Binder type", ["Any"] + BINDER_TYPES)
    margin = c[3].slider("Safety margin", 0.0, 1.0, 0.5, 0.1,
                         help="0 = trust the point prediction. 1 = the whole 90% range must meet the target.")

    c = st.columns(4)
    temp = c[0].slider("Application temperature (°C)", 18, 35, 27)
    rh = c[1].slider("Relative humidity (%)", 35, 80, 55)
    film = c[2].slider("Wet film thickness (µm)", 80, 150, 110)
    n_cand = c[3].selectbox("Candidates to search", [10000, 30000, 60000], index=1)

    submitted = st.form_submit_button("Find recipes", type="primary")

if submitted:
    targets = {}
    for key, (use, lo_v, hi_v) in targets_in.items():
        if use:
            targets[key] = (lo_v if lo_v > 0 else None, hi_v if hi_v > 0 else None)
    fixed_items = tuple(sorted({"temp_C": float(temp), "rh_pct": float(rh), "wet_film_um": float(film)}.items()))
    pool = get_pool(fixed_items, None if binder_choice == "Any" else binder_choice, int(n_cand), 0)
    res, rep = design.select(pool, targets, margin=margin, max_voc=max_voc,
                             max_cost=max_cost if max_cost > 0 else None, top_k=10)
    st.session_state["suggest"] = (res, rep, targets)

if "suggest" not in st.session_state:
    st.info("Set your targets above and press **Find recipes**.")
    st.stop()

res, rep, targets = st.session_state["suggest"]
st.divider()

NAMES = {"viscosity_ku": "Viscosity", "gloss_60": "Gloss", "contrast_ratio": "Contrast ratio",
         "touch_dry_min": "Touch-dry time", "voc_gpl": "VOC limit", "cost_per_L": "Cost limit"}

if rep["n_feasible"] == 0:
    st.error(f"No recipe among {rep['n_candidates']:,} candidates met every requirement.")
else:
    st.success(f"{rep['n_feasible']:,} of {rep['n_candidates']:,} candidates met every requirement. "
               f"Showing the {len(res)} cheapest.")

st.subheader("What limited the search?")
st.caption("Share of candidates that pass each requirement on its own. The lowest bars are the "
           "hardest requirements. Two requirements that pull in opposite directions (for example "
           "high gloss and high opacity) leave very few survivors together.")
pr = pd.DataFrame({"Requirement": [NAMES[k] for k in rep["pass_rate"]],
                   "Candidates passing (%)": [round(v * 100, 1) for v in rep["pass_rate"].values()]})
st.bar_chart(pr, x="Requirement", y="Candidates passing (%)", horizontal=True)

if len(res):
    st.subheader("Cheapest qualifying recipes")
    show = res[["binder", "tio2", "caco3", "talc", "clay", "silica", "hec", "hase", "binder_type",
                "viscosity_ku", "gloss_60", "contrast_ratio", "touch_dry_min", "finish",
                "pvc_cpvc", "voc_gpl", "cost_per_L"]].copy()
    show.index = show.index + 1
    show.index.name = "rank"
    st.dataframe(show)
    st.download_button("Download results (CSV)", res.to_csv(index=False), "suggested_recipes.csv", "text/csv")

    st.subheader("Full recipe sheet")
    rank = st.selectbox("Recipe", list(range(1, len(res) + 1)))
    r = res.iloc[rank - 1]
    c = st.columns(2)
    ing = pd.DataFrame({"Ingredient": [LABELS[k] for k in COMPONENTS],
                        "kg / 100 kg": [r[k] for k in COMPONENTS]})
    c[0].dataframe(ing, hide_index=True)
    proc = pd.DataFrame({
        "Setting": ["Binder type", "Binder Tg (°C)", "Extender size (µm)", "Dispersion (rpm)",
                    "Dispersion time (min)", "pH", "Temperature (°C)", "Humidity (%)", "Wet film (µm)"],
        "Value": [r["binder_type"], r["binder_tg"], r["extender_psize_um"], r["dispersion_rpm"],
                  r["disp_time_min"], r["ph"], r["temp_C"], r["rh_pct"], r["wet_film_um"]]})
    proc["Value"] = proc["Value"].astype(str)
    c[1].dataframe(proc, hide_index=True)

with st.expander("How this works and what to be careful about"):
    st.markdown(
        "- **Random search:** thousands of recipes are drawn from the same ranges the models were "
        "trained on, predicted, filtered by your targets and ranked by ingredient cost.\n"
        "- **Safety margin:** moves each prediction against you by that fraction of its 90% error "
        "range before checking the target, so recipes that only pass by luck are dropped.\n"
        "- **Beware of model exploitation:** the search picks recipes where the model happens to "
        "predict well, including where it is slightly wrong in your favour. Always confirm a suggested "
        "recipe with a real lab batch.\n"
        "- **Wide ranges at low gloss:** the gloss error range has one fixed width, so very low "
        "gloss targets with a high safety margin can be impossible to satisfy."
    )
