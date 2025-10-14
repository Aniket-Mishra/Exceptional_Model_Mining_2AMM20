import numpy as np
from configurations import *
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


file_name = dataset_information["boston"]["file_name"]
target = dataset_information["boston"]["target"]

# Boston
boston = fetch_openml(name="boston", version=1, as_frame=True)
df = boston.frame

# Col name to target cuz i no wan retype
df.rename(columns={f"{target}": "target"}, inplace=True)

# Other datasets


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


df.to_csv(
    f"dataset_with_residuals/{file_name}_with_residuals.csv", index=False
)
