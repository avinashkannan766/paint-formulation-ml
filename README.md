# Paint formulation ML project (starter)

Predicts viscosity, gloss, opacity (contrast ratio), touch-dry time and finish from a
paint formulation, and suggests recipes that meet target properties. Trained on SYNTHETIC
data - not real lab measurements.

## Files
- paint_science.py : density / PVC / CPVC / volume-solids formulas and feature engineering
- generate_data.py : builds the synthetic dataset -> data/paint_formulations.csv
- train.py         : trains Ridge / RandomForest / LightGBM with grouped CV, saves models/ and reports/
- design.py        : recipe suggestion (random search over candidates, filter by targets, rank by cost)
- app.py           : Streamlit page 1 - enter a recipe, see predictions with 90% ranges
- pages/1_Suggest_recipe.py : Streamlit page 2 - enter targets, get the cheapest qualifying recipes

## Setup
    pip install -r requirements.txt

## Run order (from this folder)
    python generate_data.py     # optional, CSV is included
    python train.py             # creates models/ and reports/
    streamlit run app.py        # both pages appear in the left navigation

Command-line recipe search without the app:
    python design.py

Re-run train.py whenever you change the data or the generator.
