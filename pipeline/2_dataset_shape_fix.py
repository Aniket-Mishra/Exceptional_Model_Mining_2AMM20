import glob

import numpy as np
import pandas as pd

# df = pd.read_csv("/Users/aniket/github/Exceptional_Model_Mining_2AMM20/pipeline/dataset_with_residuals/year_prediction_msd_with_residuals.csv")

df = pd.read_csv(
    "/Users/aniket/github/Exceptional_Model_Mining_2AMM20/pipeline/dataset_with_residuals/year_prediction_msd_with_residuals.csv"
)
df.drop(columns="Unnamed: 0", inplace=True)

df["target"] = df["target"].astype(np.int16)

timbre_cols = [c for c in df.columns if c.startswith("timbre_")]
df[timbre_cols] = df[timbre_cols].astype(np.float32)

df.to_csv(
    "/Users/aniket/github/Exceptional_Model_Mining_2AMM20/pipeline/dataset_with_residuals/year_prediction_msd_with_residuals.csv",
    index=False,
)

# for all results

all_languages_results = glob.glob("outputs/emm_all_languages_results*.csv")

df = pd.DataFrame()
for file in all_languages_results:
    dfx = pd.read_csv(file)
    name = file.split("_")[-1].split(".")[0]
    dfx["dataset"] = name
    df = pd.concat([df, dfx])

df.loc[df["pareto"] == True].to_csv(
    "outputs/emm_all_pareto_front_true.csv", index=False
)

dfx = pd.read_csv("outputs/emm_all_pareto_front.csv")
