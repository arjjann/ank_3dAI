#model optimization for overand underfitting
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    """Helper function to calculate MAE for a given tree size(max_leaf_nodes)"""
    model=DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val=model.predict(val_X)
    mae=mean_absolute_error(val_y, preds_val)
    return mae

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

    train_X, val_X, train_y, val_y=train_test_split(X, y, random_state=1)

    candidate_max_leaf_nodes=[5,50,500,5000]
    scores={}

    print("Underfitting & Overfitting analysis")
    for max_leaf_nodes in candidate_max_leaf_nodes:
        my_mae=get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
        scores[max_leaf_nodes]=my_mae
        print(f"\n Max leaf node:{max_leaf_nodes:4d}\t Mean Absolute Error is:{my_mae:,.3f}")

    best_tree_size=min(scores, key=scores.get)
    print(f"\nOptimal tree size: {best_tree_size} leaf nodes (MAE:{scores[best_tree_size]:,.3f})")

    final_model= DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)
    print("Final model trained on 100% of the data set using optimal paramaters.")

if __name__=="__main__":
    main()
