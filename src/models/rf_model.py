import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    rf_model= RandomForestRegressor(random_state=1)
    rf_model.fit(train_X, train_y)

    rf_val_predictions=rf_model.predict(val_X)
    rf_val_mae=mean_absolute_error(val_y, rf_val_predictions)

    print("Random forest Module Results\n")
    print(f"Validation MAE:{rf_val_mae:,.3f}")

if __name__=="__main__":
    main()

    