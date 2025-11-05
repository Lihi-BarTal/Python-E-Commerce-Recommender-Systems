import numpy as np
import pandas as pd
import scipy as sp

def solve():
    # Reading the tarin data and converting it to df , train_data.shape = (#num_of_rows , 3)
    train_data = pd.read_csv("train.csv")
    # Reading the test data and converting it to df , test_data.shape = (#num_of_rows , 2)
    test_data = pd.read_csv("test.csv")

    # Processing user's data
    all_users = train_data["user id"].unique()
    user_to_index = {u: idx for idx, u in enumerate(all_users)}

    # Processing item's data
    all_items = train_data["item id"].unique()
    item_to_index = {i: idx for idx, i in enumerate(all_items)}

    # Step 1 - Building A
    A = np.zeros((len(all_users), len(all_items)))
    for _,row in train_data.iterrows():
        # Filling A with train data and all other
        user_index = user_to_index[row["user id"]]
        item_index = item_to_index[row["item id"]]
        A[user_index,item_index] = row["rating"]

    # Step 2 - finding A's approximation
    k = 10
    u , singular , v_t = sp.sparse.linalg.svds(A, k = k)

    # Step 3 - Calculating prediction
    s = np.diag(singular)
    A_pred = u @ s @ v_t

    test_with_pred_rating = test_data.copy()
    test_with_pred_rating["pred rating"] = 0.0
    for index,row in test_with_pred_rating.iterrows():
        user_index = user_to_index[row["user id"]]
        item_index = item_to_index[row["item id"]]
        value = A_pred[user_index,item_index]
        if value > 5:
            test_with_pred_rating.at[index, "pred rating"] = 5
        elif value < 1:
            test_with_pred_rating.at[index, "pred rating"] = 1
        else :
            test_with_pred_rating.at[index, "pred rating"] = value

    train_with_pred_rating = train_data.copy()
    train_with_pred_rating["pred rating"] = 0.0
    for index,row in train_with_pred_rating.iterrows():
        user_index = user_to_index[row["user id"]]
        item_index = item_to_index[row["item id"]]
        train_with_pred_rating.at[index, "pred rating"] = A_pred[user_index,item_index]

    # Calculating MSE value
    mse = ((train_with_pred_rating["rating"] - train_with_pred_rating["pred rating"]) ** 2).mean()
    with open("mse.txt", "a") as f:
        f.write(f"{mse}")

    test_with_pred_rating.rename(columns={"pred rating": "rating"}, inplace=True)
    test_with_pred_rating.to_csv("pred2.csv", index=False)

if __name__ == "__main__":
    solve()



