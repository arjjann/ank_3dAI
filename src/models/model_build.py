import pandas as pd
from sklearn.tree import DecisionTreeRegressor

def main():
    #Load data set and path to train.csv
    file_path='data/train.csv'
    home_data=pd.read_csv(file_path)
    #inspect dataset
    print("Dataset Overview")
    print(home_data.head())
    print("\nSummary statistics")
    print(home_data.describe())

    #inspect the dataset
    y=home_data.SalesPrice

    #selecct feature
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

    #Build model
    model=DecisionTreeRegressor(random_state=1)
    model.fit(X, y)

    #make predicition
    print("\nModel Predection(First 5 row)")
    predictions=model.predict(X.head())
    print("Predicted prices:",predictions)
    print("Actual Prices", y.head().tolist())

if __name__=="main":
    main()
