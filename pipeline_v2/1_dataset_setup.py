import os

import numpy as np
import pandas as pd
from configurations import *
from scipy.io import arff
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

"""
ToDo:
1. Add more datasets
2. Configurable for datasets and models
3. Configurable parameters - more
4. Update latex with chosen values per language
"""


def compute_cv_residuals(
    df, feature_cols, target_col="target", n_splits=5, seed=0
):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oos_pred = np.empty(len(df), dtype=float)
    for train_idx, test_idx in kf.split(df):
        Xtr, Xte = (
            df.iloc[train_idx][feature_cols],
            df.iloc[test_idx][feature_cols],
        )
        ytr = df.iloc[train_idx][target_col]
        m = RandomForestRegressor(n_estimators=200, random_state=seed)
        m.fit(Xtr, ytr)
        oos_pred[test_idx] = m.predict(Xte)
    return (df[target_col].to_numpy() - oos_pred) ** 2


# file_name = dataset_information["boston"]["file_name"]
# target = dataset_information["boston"]["target"]

# # Boston
# boston = fetch_openml(name="boston", version=1, as_frame=True)
# df = boston.frame

# # Col name to target cuz i no wan retype
# df.rename(columns={f"{target}": "target"}, inplace=True)

# # Other datasets


# model = RandomForestRegressor(
#     random_state=DEFAULT_RANDOM_STATE, n_estimators=NUM_ESTIMATORS
# )
# model.fit(df.drop("target", axis=1), df["target"])

# y_pred = model.predict(df.drop("target", axis=1))

# for actual, pred in list(zip(df["target"][:5], y_pred[:5])):
#     print(f"Actual target: {actual:.1f}, Predicted target: {pred:.2f}")

# df["Residual_signed"] = df["target"] - y_pred

# residuals = (df["target"] - y_pred) ** 2
# df["Residual"] = residuals

# # build CV residuals and re-mine
# feature_cols = [c for c in df.columns if c not in {"target", "Residual"}]
# df["Residual_CV"] = compute_cv_residuals(
#     df,
#     feature_cols,
#     target_col="target",
#     n_splits=NUM_SPLITS,
#     seed=DEFAULT_RANDOM_STATE,
# )


# os.makedirs("dataset_with_residuals", exist_ok=True)
# df.to_csv(
#     f"dataset_with_residuals/{file_name}_with_residuals.csv", index=False
# )


for dataset_name, data_information in dataset_information.items():
    print(dataset_name)
    file_name = data_information["file_name"]
    target = data_information["target"]
    print(file_name)
    print(target)
    print()
    if file_name == "boston_housing":
        boston = fetch_openml(name="boston", version=1, as_frame=True)
        df = boston.frame
        print(df.shape)
    elif file_name == "forestfires":
        df = pd.read_csv(
            "/Users/aniket/github/Exceptional_Model_Mining_2AMM20/datasets/forestfires.csv"
        )
        # Do strings to integer conversion.
        month_map = {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }

        day_map = {
            "sun": 1,
            "mon": 2,
            "tue": 3,
            "wed": 4,
            "thu": 5,
            "fri": 6,
            "sat": 7,
        }

        df["day"] = df["day"].map(day_map)
        df["month"] = df["month"].map(month_map)
    elif file_name == "auto-mpg":
        arff_file = arff.loadarff(
            "/Users/aniket/Downloads/regression_datasets/auto-mpg.arff"
        )
        df = pd.DataFrame(arff_file[0])
        df["cylinders"] = df["cylinders"].astype(int)
        df["model"] = df["model"].astype(int)
        df["origin"] = df["origin"].astype(int)
    elif file_name == "cmc":
        df = pd.read_csv(
            "/Users/aniket/github/Exceptional_Model_Mining_2AMM20/datasets/cmc.data",
            header=None,
            names=[
                "Wife_Age",
                "Wife_Education",
                "Husband_Education",
                "Children",
                "Wife_religion",
                "Wife_working",
                "Husband_Occupation",
                "SOLI",
                "Media_Exposure",
                "Contraceptive_Method",
            ],
        )
    else:
        print("You messed sth up dumdum. Everything is hardcoded BWAHAHAHA")
        continue
    name = file_name.split(".")[0]

    df.rename(columns={f"{target}": "target"}, inplace=True)
    model = RandomForestRegressor(
        random_state=DEFAULT_RANDOM_STATE, n_estimators=NUM_ESTIMATORS
    )
    model.fit(df.drop("target", axis=1), df["target"])

    y_pred = model.predict(df.drop("target", axis=1))

    for actual, pred in list(zip(df["target"][:5], y_pred[:5])):
        print(f"Actual target: {actual:.1f}, Predicted target: {pred:.2f}")

    df["Residual_signed"] = df["target"] - y_pred

    residuals = (df["target"] - y_pred) ** 2
    df["Residual"] = residuals

    # build CV residuals and re-mine
    feature_cols = [c for c in df.columns if c not in {"target", "Residual"}]
    df["Residual_CV"] = compute_cv_residuals(
        df,
        feature_cols,
        target_col="target",
        n_splits=NUM_SPLITS,
        seed=DEFAULT_RANDOM_STATE,
    )
    os.makedirs("dataset_with_residuals", exist_ok=True)
    df.to_csv(f"dataset_with_residuals/{name}_with_residuals.csv", index=False)
