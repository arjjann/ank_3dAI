import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

def score_dataset(X_train, X_valid, y_train, y_valid):
    """Fits a Random Forest model and returns the validation MAE."""
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

def main():
    # Load dataset
    file_path = "data/train.csv"
    X_full = pd.read_csv(file_path, index_col='Id')

    # Remove rows with missing target value (SalePrice)
    X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X_full.SalePrice
    X_full.drop(['SalePrice'], axis=1, inplace=True)

    # To keep this example focused on missing numerical values, use numerical predictors only
    X = X_full.select_dtypes(exclude=['object'])

    # DivideS data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=0
    )

    print("Handling Missing Values Analysis")

    # APPROACH 1: Drop Columns with Missing Values
    cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

    reduced_X_train = X_train.drop(cols_with_missing, axis=1)
    reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

    mae_1 = score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid)
    print(f"MAE from Approach 1 (Drop columns with missing values): ${mae_1:,.2f}")

    # APPROACH 2: Imputation (SimpleImputer with Mean Strategy)
    my_imputer = SimpleImputer()
    
    # fit_transform removes column names, so wrap output back into a DataFrame
    imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
    imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

    # Restore column names
    imputed_X_train.columns = X_train.columns
    imputed_X_valid.columns = X_valid.columns

    mae_2 = score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid)
    print(f"MAE from Approach 2 (Imputation): ${mae_2:,.2f}")

    # APPROACH 3: Extension to Imputation (Tracking Imputed Columns)

    X_train_plus = X_train.copy()
    X_valid_plus = X_valid.copy()

    # Add boolean indicator columns for missing entries
    for col in cols_with_missing:
        X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
        X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

    # Impute missing values
    my_imputer_plus = SimpleImputer()
    imputed_X_train_plus = pd.DataFrame(my_imputer_plus.fit_transform(X_train_plus))
    imputed_X_valid_plus = pd.DataFrame(my_imputer_plus.transform(X_valid_plus))

    # Restore column names
    imputed_X_train_plus.columns = X_train_plus.columns
    imputed_X_valid_plus.columns = X_valid_plus.columns

    mae_3 = score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid)
    print(f"MAE from Approach 3 (An Extension to Imputation): ${mae_3:,.2f}")

if __name__ == "__main__":
    main()