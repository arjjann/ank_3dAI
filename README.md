# ank_3dAI

## Requirements

* Python 3.x
* pandas
* scikit-learn
* matplotlib

```bash
pip install pandas scikit-learn matplotlib
```

## Directory Structure

```text
ank_3dAI/
    data/                           #   Datasets (Git-ignored)
        cads/                       #   Raw 3D STEP/STP geometry files

    graphs/                         #   Extracted PyTorch Geometric (.pt) tensors
        train.csv                   #   Tabular dataset for baseline ML scripts
    
    docs/                           #   Documentation
        AGENTS.md                   #   Agent guidelines

    logs/                           #   Logs & runtime artifacts
        curl_response

    src/                            #   Source code
        data_extraction/            #   B-Rep topology & OCCT parsing scripts
    
     preprocessing/                 #   Feature cleaning & missing-value handling
        missing_values.py
    
     models/                        #   Baseline models & evaluation pipeline
        model_build.py
        model_validation.py
        model_optimiation.py
        rf_model.py
        DNN.py
        cross_verification.py
```

## Setup

Place `train.csv` inside the `data/` directory:
`data/train.csv`

## Scripts

**Models** (`src/models/`)

1. **`model_build.py`**: Initial model training and prediction preview using Decision Tree Regressor.
2. **`model_validation.py`**: Train/validation data split and baseline MAE calculation.
3. **`model_optimiation.py`**: Hyperparameter tuning (`max_leaf_nodes`) to address overfitting and underfitting.
4. **`rf_model.py`**: Baseline ensemble modeling using Random Forest Regressor.

**Preprocessing** (`src/preprocessing/`)

5. **`missing_values.py`**: Comparison of 3 strategies for missing numeric data (Dropping columns, Simple Imputation, and Imputation with indicators).
