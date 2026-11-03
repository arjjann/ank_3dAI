import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def main():

    file_path="data/train.csv"

    home_data=pd.read_csv(file_path)

    y=home_data.SalePrice
    feature_names=[
        'LotArea', 
        'YearBuilt', 
        '1stFlrSF', 
        '2ndFlrSF', 
        'FullBath', 
        'BedroomAbvGr', 
        'TotRmsAbvGrd'
    ]
    X=home_data[feature_names]

    #split data into training and calidation sets 
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    #Define and fit the model in the training data only
    iwoa_model=DecisionTreeRegressor(random_state=1)
    iwoa_model.fit(train_X, train_y)

    val_predictions =iwoa_model.predict(val_X)

    #calculation of MAE
    val_mae = mean_absolute_error(val_y, val_predictions)

    print("--- Model Validation Results ---")
    print(f"Validation MAE: ${val_mae:,.2f}")

if __name__ == "__main__":
    main()