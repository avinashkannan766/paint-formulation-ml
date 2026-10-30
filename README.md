# Paint formulation ML project (starter)

Predicts viscosity, gloss, opacity (contrast ratio), touch-dry time and finish from a
paint formulation. Trained on SYNTHETIC data - not real lab measurements.

## Files
- paint_science.py : density / PVC / CPVC / volume-solids formulas and feature engineering
- generate_data.py : builds the synthetic dataset -> data/paint_formulations.csv
- train.py         : trains Ridge / RandomForest / LightGBM with grouped CV, saves models/ and reports/
- app.py           : Streamlit app to enter a recipe and see predictions with 90% ranges

## Setup
    pip install -r requirements.txt

## Run order (from this folder)
    python generate_data.py     # optional, CSV is included
    python train.py             # creates models/ and reports/
    streamlit run app.py        # opens the app in your browser

Re-run train.py whenever you change the data or the generator.
