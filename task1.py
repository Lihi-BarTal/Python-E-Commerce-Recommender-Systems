import numpy as np
import pandas as pd
import scipy as sp

def solve():
    # Reading the tarin data and converting it to df , train_data.shape = (#num_of_rows , 3)
    train_data = pd.read_csv("train.csv")
    # Reading the test data and converting it to df , test_data.shape = (#num_of_rows , 2)
    test_data = pd.read_csv("test.csv")

    # Calculating r_avg
    r_avg = train_data["rating"].mean()

    # Processing user's data
    all_users = train_data["user id"].unique()
    num_users = len(all_users)
    user_to_index = {u: idx for idx, u in enumerate(all_users)}

    # Processing item's data
    all_items = train_data["item id"].unique()
    num_items = len(all_items)
    item_to_index = {i: idx for idx, i in enumerate(all_items)}

    num_of_rows_in_train = len(train_data)
    total_users_and_items = num_users + num_items

    # Converting to LS problem
    c = train_data["rating"].to_numpy() - r_avg
    A = sp.sparse.lil_matrix((num_of_rows_in_train,total_users_and_items))
    for index,row in train_data.iterrows():
        user_index = user_to_index[row["user id"]]
        item_index = item_to_index[row["item id"]]
        A[index,user_index] = 1
        A[index,item_index + num_users] = 1
    A = A.tocsr()

    # Building the solution equation
    regularization_lambda = 1.0
    ATA = A.T @ A + regularization_lambda * np.identity(total_users_and_items)
    ATc = A.T @ c
    b = np.linalg.solve(ATA , ATc)

    # Separating b to user biases and item biases
    user_bias_from_ls = {key_user_id: b[val_user_index] for key_user_id, val_user_index in user_to_index.items()}
    item_bias_from_ls = {key_item_id: b[val_item_index + num_users] for key_item_id, val_item_index in item_to_index.items()}

    # Calculating r_hat values
    r_hat_pred_train_df = train_data.copy()
    r_hat_pred_train_df["pred_rating"] = 0.0
    for index, row in r_hat_pred_train_df.iterrows():
        user = row["user id"]
        item = row["item id"]
        user_bias = user_bias_from_ls.get(user, 0)
        item_bias = item_bias_from_ls.get(item, 0)
        value = r_avg + user_bias + item_bias
        r_hat_pred_train_df.at[index, "pred_rating"] = value

    # Calculating MSE value
    mse = ((r_hat_pred_train_df["rating"] - r_hat_pred_train_df["pred_rating"]) ** 2).mean()
    with open("mse.txt", "w") as f:
        f.write(f"{mse}\n")

    r_hat_pred_test_df = test_data.copy()
    r_hat_pred_test_df["pred_rating"] = 0.0
    for index, row in r_hat_pred_test_df.iterrows():
        user = row["user id"]
        item = row["item id"]
        user_bias = user_bias_from_ls.get(user, 0)
        item_bias = item_bias_from_ls.get(item, 0)
        value = r_avg + user_bias + item_bias
        if value > 5:
            r_hat_pred_test_df.at[index, "pred_rating"] = 5
        elif value < 1:
            r_hat_pred_test_df.at[index, "pred_rating"] = 1
        else :
            r_hat_pred_test_df.at[index, "pred_rating"] = value

    r_hat_pred_test_df.rename(columns={"pred_rating": "rating"}, inplace=True)
    r_hat_pred_test_df.to_csv("pred1.csv", index=False)


if __name__ == "__main__":
    solve()

