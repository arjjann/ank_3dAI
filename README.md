# ank_3dAI

## Requirements

* Python 3.x
* pandas
* scikit-learn
* matplotlib

```bash
pip install pandas scikit-learn matplotlib
```

## Setup

Place `train.csv` inside a `data/` directory:
`data/train.csv`

## Scripts

1. **`model_build.py`**: Initial model training and prediction preview using Decision Tree Regressor.
2. **`model_validation.py`**: Train/validation data split and baseline MAE calculation.
3. **`model_optimiation.py`**: Hyperparameter tuning (`max_leaf_nodes`) to address overfitting and underfitting.
4. **`rf_model.py`**: Baseline ensemble modeling using Random Forest Regressor.
5. **`missing_values.py`**: Comparison of 3 strategies for missing numeric data (Dropping columns, Simple Imputation, and Imputation with indicators).