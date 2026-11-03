# AGENTS.md

Standalone ML tutorial scripts (Kaggle Ames housing + concrete regression), not a package. No build system, tests, lint, CI, or codegen.

## Environment
- Use the existing venv: `source 3dai_env/bin/activate` (Python 3.14).
- Already installed: `pandas`, `numpy`, `matplotlib`, `seaborn`.
- **Not installed despite being used**: `scikit-learn` (all housing scripts) and `tensorflow`/`keras` (`DNN.py`). Install with `pip install scikit-learn tensorflow` before running those scripts.
- `README.md` requirements are incomplete: it omits TensorFlow and lists only three deps.

## Data
- Data is gitignored and absent; nothing runs end-to-end until it is added.
- Housing scripts read `data/train.csv` via a **cwd-relative path** — run from the repo root.
- Expected columns: Ames housing with target `SalePrice` and `Id` index; `DNN.py` instead reads `data/concrete.csv` (path anchored to the script) targeting `CompressiveStrength`.

## Running
- Each file is a script: `python missing_values.py`, etc. No test suite exists; verify by running the script.
- `cross_verification.py` calls `plt.show()`; use `MPLBACKEND=Agg` in headless environments.

## Known issues
- `model_build.py` is broken and executes nothing: typo `home_data.SalesPrice` (should be `SalePrice`) and guard `if __name__=="main"` (should be `"__main__"`). Use `model_validation.py` / `model_optimiation.py` as working references.
- `model_optimiation.py` (filename typo) never calls `.fit()` on `final_model` despite the printed message.
- `DNN.py` (omitted from the README script list) only constructs two Keras models and prints summaries; it performs no training.

## Conventions
- Scripts define `main()` and guard with `if __name__ == "__main__":`.
- Feature selection is a hardcoded `feature_names` list repeated across housing scripts; keep them consistent when editing.
- Fixed `random_state` values are used for reproducibility (`0` or `1`).
