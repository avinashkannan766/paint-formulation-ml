"""Train the models on first start if they are missing.

Hosted deployments (for example Streamlit Community Cloud) start from the GitHub repository,
which does not contain models/ (it is in .gitignore). This trains them once with the host's
own library versions, which also avoids saved-model version mismatches.
"""
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent


def models_ready() -> bool:
    return (BASE / "models" / "meta.json").exists()


def ensure_models() -> bool:
    """Returns True if training had to run, False if models were already present."""
    if models_ready():
        return False
    old = os.getcwd()
    os.chdir(BASE)  # train.py uses paths relative to the project folder
    try:
        data = BASE / "data" / "paint_formulations.csv"
        if not data.exists():  # regenerate the synthetic data if the CSV is missing
            import generate_data
            data.parent.mkdir(exist_ok=True)
            generate_data.build_dataset(80, 6, 42).to_csv(data, index=False)
        import train
        train.main()
    finally:
        os.chdir(old)
    return True


if __name__ == "__main__":
    print("trained" if ensure_models() else "models already present")
