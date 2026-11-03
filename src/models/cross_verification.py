import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# STEP 1: DEFINE EVALUATION FUNCTION (Cross-Validation Pipeline)
def get_score(n_estimators, X, y):
    """ Builds a pipeline (Imputer + RandomForest) and computes 
    the average Mean Absolute Error (MAE) across 3 cross-validation folds.
    """
    my_pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=0))
    ])
    
    # cross_val_score splits data into 3 folds (cv=3)
    # scikit-learn scores are negative MAE, so multiply by -1
    scores = -1 * cross_val_score(
        my_pipeline, X, y,
        cv=3,
        scoring='neg_mean_absolute_error'
    )
    
    return scores.mean()


def main():
    # SETUP & DATA PREPROCESSING
    file_path = "data/train.csv"
    train_data = pd.read_csv(file_path, index_col='Id')

    # Drop rows missing the target variable
    train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = train_data.SalePrice
    train_data.drop(['SalePrice'], axis=1, inplace=True)

    # Keep numeric columns only for this exercise
    numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]
    X = train_data[numeric_cols].copy()

    print(f"Loaded dataset with {X.shape[0]} rows and {X.shape[1]} numeric features.\n")

    # STEP 2: TEST DIFFERENT HYPERPARAMETER VALUES (n_estimators)

    tree_counts = [50, 100, 150, 200, 250, 300, 350, 400]
    results = {}

    print("Evaluating models using 3-Fold Cross-Validation...")
    for n in tree_counts:
        mae_score = get_score(n, X, y)
        results[n] = mae_score
        print(f"  Trees: {n:3d}  -->  Average MAE: ${mae_score:,.2f}")

    # To FIND BEST PARAMETER VALUE & VISUALIZE
    best_n_estimators = min(results, key=results.get)
    print(f"\nBest parameter value (lowest MAE): n_estimators = {best_n_estimators}")
    print(f"Lowest Cross-Validation MAE: ${results[best_n_estimators]:,.2f}")

    # Plotting results to match Kaggle cell output
    plt.figure(figsize=(8, 4))
    plt.plot(list(results.keys()), list(results.values()), marker='o')
    plt.title("Cross-Validation MAE vs. Number of Trees")
    plt.xlabel("n_estimators")
    plt.ylabel("Average MAE ($)")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()