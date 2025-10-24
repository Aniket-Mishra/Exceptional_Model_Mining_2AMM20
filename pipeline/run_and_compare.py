from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from configurations import *
from I_conjunction_rules import mine_conj_rules
from II_polynomial_rules import augment_with_poly
from III_d_tree_rules import mine_ltree_rules
from IV_symbolic_rules import mine_lsymb_rules_light
from utils import (
    annotate_interpretability,
    permutation_pvalue,
)


def path_root() -> Path:
    return Path(__file__).resolve().parent


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def split_directional(df_rules: pd.DataFrame):
    """Return under/over lists sorted by |q_residual| desc."""
    if df_rules is None or len(df_rules) == 0:
        empty = pd.DataFrame(
            columns=[
                "rule",
                "length",
                "size",
                "mean_residual",
                "delta_from_global",
                "q_residual",
            ]
        )
        return empty, empty
    under = df_rules[df_rules["delta_from_global"] > 0].copy()
    over = df_rules[df_rules["delta_from_global"] < 0].copy()
    under = under.sort_values("q_residual", ascending=False).reset_index(
        drop=True
    )
    over = over.sort_values("q_residual", ascending=False).reset_index(
        drop=True
    )
    return under, over


def pareto_mark(
    df: pd.DataFrame, q_col="q_residual", i_col="I_exp"
) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    arr = df[[q_col, i_col]].to_numpy(dtype=float)
    n = len(arr)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        qi, ii = arr[i]
        mask = (
            (arr[:, 0] >= qi)
            & (arr[:, 1] >= ii)
            & ((arr[:, 0] > qi) | (arr[:, 1] > ii))
        )
        if mask.any():
            dominated[i] = True
    out = df.copy()
    out["pareto"] = ~dominated
    return out


def apply_pareto_tradeoff(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    if "I_exp" not in df.columns:
        raise ValueError(
            "Call annotate_interpretability() before Pareto ranking."
        )
    d = pareto_mark(df, q_col="q_residual", i_col="I_exp")
    d["score"] = np.sqrt(
        d["q_residual"].astype(float) * d["I_exp"].astype(float)
    )
    d = d.sort_values(
        ["pareto", "score"], ascending=[False, False]
    ).reset_index(drop=True)
    return d


def prepare(df, lang_tag):
    if df is None:
        return df
    # annotate adds complexity, I_inv, I_exp, expressiveness etc.
    df = annotate_interpretability(df, language=lang_tag)
    df = apply_pareto_tradeoff(df)
    return df.head(TOP_K_PER_LANGUAGE).reset_index(drop=True)


def add_permutation_pvalues(
    df_rules, data_frame, res_col=RES_COL, language_tag="conj"
):
    if df_rules is None or len(df_rules) == 0:
        return df_rules
    df_rules = df_rules.copy()
    pvals, qobs_list = [], []
    if language_tag == "symb" and {"expr", "operator", "threshold"}.issubset(
        df_rules.columns
    ):
        for i in range(len(df_rules)):
            expr_str = str(df_rules.loc[i, "expr"])
            op = str(df_rules.loc[i, "operator"])
            thr = float(df_rules.loc[i, "threshold"])
            tmp_df = data_frame.copy()
            tmp_col = f"__symb_expr_{i}"
            tmp_df[tmp_col] = tmp_df.eval(expr_str, engine="python")
            rule = f"{tmp_col} {op} {thr}"
            qobs, p = permutation_pvalue(
                tmp_df,
                rule,
                res_col=res_col,
                n_perm=NUM_PERMUTATIONS,
                seed=DEFAULT_RANDOM_STATE,
            )
            qobs_list.append(qobs)
            pvals.append(p)
    else:
        for i in range(len(df_rules)):
            rule = df_rules.loc[i, "rule"]
            qobs, p = permutation_pvalue(
                data_frame,
                rule,
                res_col=res_col,
                n_perm=NUM_PERMUTATIONS,
                seed=DEFAULT_RANDOM_STATE,
            )
            qobs_list.append(qobs)
            pvals.append(p)
    df_rules["q_perm"] = qobs_list
    df_rules["p_value"] = pvals
    return df_rules


def main():
    root = path_root()
    out_dir = root / "outputs"
    ensure_dir(out_dir)
    for dataset_name, data_information in dataset_information.items():
        print(dataset_name)
        file_name = data_information["file_name"]
        target = data_information["target"]
        print(file_name)
        print(target)
        print()
        df_path = Path(
            f"{root}/dataset_with_residuals/{file_name}_with_residuals.csv"
        )
        if not df_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {df_path}. Run 1_dataset_setup.py first."
            )
        df = pd.read_csv(df_path)
        df.drop(columns=[RES_COL, RES_COL_CV, RES_COL_SIGNED])

        # L_conj, can use other res
        results_conj = mine_conj_rules(
            df, res_col=RES_COL, max_rule_len=3, top_k=TOP_K_PER_LANGUAGE
        )

        # L_poly. degree 2 basis, with readability guards
        df_poly = augment_with_poly(
            df,
            res_col=RES_COL,
            max_squares=8,
            max_interactions=8,
            candidate_feature_limit=10,
        )
        results_poly = mine_conj_rules(
            df_poly,
            res_col=RES_COL,
            max_rule_len=MAX_RULE_LEN_POLY,
            top_k=TOP_K_PER_LANGUAGE,
        )

        # L_tree
        results_tree = mine_ltree_rules(
            df,
            res_col=RES_COL,
            max_depth=LTREE_MAX_DEPTH,
            min_leaf=LTREE_MIN_LEAF,
            top_k=TOP_K_PER_LANGUAGE,
            random_state=LTREE_RANDOM_STATE,
        )
        if len(results_tree) == 0:
            results_tree = mine_ltree_rules(
                df,
                res_col=RES_COL,
                max_depth=3,
                min_leaf=5,
                top_k=TOP_K_PER_LANGUAGE,
                random_state=LTREE_RANDOM_STATE,
            )

        # L_symb
        results_symb = mine_lsymb_rules_light(
            df,
            res_col=RES_COL,
            feature_exclude={
                "Residual",
                "Residual_CV",
                "Residual_signed",
            },
            top_features=SYMB_TOP_FEATURES,
            max_triplets=SYMB_MAX_TRIPLETS,
            # thresholds_per_expr=SYMB_THRESHOLDS_PER_EXPR,
            thresholds_per_expr=LOCAL_QUANTILES,
            min_support=MIN_SUPPORT,
            top_k=TOP_K_PER_LANGUAGE,
        )

        # annotate + Pareto
        results_conj = prepare(results_conj, "conj")
        results_poly = prepare(results_poly, "poly")
        results_tree = prepare(results_tree, "tree")
        results_symb = prepare(results_symb, "symb")

        # Permutation p-values
        results_conj = add_permutation_pvalues(
            results_conj, df, RES_COL, language_tag="conj"
        )
        results_poly = add_permutation_pvalues(
            results_poly, df_poly, RES_COL, language_tag="poly"
        )
        results_tree = add_permutation_pvalues(
            results_tree, df, RES_COL, language_tag="tree"
        )
        results_symb = add_permutation_pvalues(
            results_symb, df, RES_COL, language_tag="symb"
        )

        dataset_dir = os.path.join(out_dir, file_name)
        os.makedirs(dataset_dir, exist_ok=True)

        results_conj.to_csv(
            f"{dataset_dir}/1_lconj_{file_name}.csv", index=False
        )
        results_poly.to_csv(
            f"{dataset_dir}/2_lpoly_{file_name}.csv", index=False
        )
        results_tree.to_csv(
            f"{dataset_dir}/3_ltree_{file_name}.csv", index=False
        )
        results_symb.to_csv(
            f"{dataset_dir}/4_lsymb_{file_name}.csv", index=False
        )

        # Combine all languages (compare_all behavior)
        all_rules = (
            pd.concat(
                [
                    results_conj.assign(language="L_conj"),
                    results_poly.assign(language="L_poly"),
                    results_tree.assign(language="L_tree"),
                    results_symb.assign(language="L_symb"),
                ],
                ignore_index=True,
            )
            .sort_values("q_residual", ascending=False)
            .reset_index(drop=True)
        )

        # under/over lists
        all_under, all_over = split_directional(all_rules)
        all_under.to_csv(
            f"{dataset_dir}/emm_all_languages_underperform_{file_name}.csv",
            index=False,
        )
        all_over.to_csv(
            f"{dataset_dir}/emm_all_languages_overperform_{file_name}.csv",
            index=False,
        )

        all_rules.to_csv(
            f"{dataset_dir}/emm_all_languages_results_{file_name}.csv",
            index=False,
        )

        # print("\nTop 10 by q_residual across languages:\n")
        # cols = [
        #     "language",
        #     "q_residual",
        #     "size",
        #     "mean_residual",
        #     "delta_from_global",
        #     "complexity",
        #     "I_exp",
        #     "rule",
        # ]
        # print(all_rules[cols].head(10).to_string(index=False))

        # Pareto front (q vs interpretability)
        def pareto_front(df: pd.DataFrame, q_col="q_residual", i_col="I_exp"):
            pts = df[[q_col, i_col]].to_numpy()
            idxs = []
            for i, (q, i_) in enumerate(pts):
                dominated = (
                    (pts[:, 0] >= q)
                    & (pts[:, 1] >= i_)
                    & ((pts[:, 0] > q) | (pts[:, 1] > i_))
                ).any()
                if not dominated:
                    idxs.append(i)
            return df.iloc[idxs].sort_values(
                [q_col, i_col], ascending=[False, False]
            )

        pf = pareto_front(all_rules)
        pf.to_csv(
            f"{dataset_dir}/emm_pareto_front_on_all_combined_rules_{file_name}.csv",
            index=False,
        )

        # # # Plots saved to outputs # this is gpt, we gotta use plotly when we get time.
        # # plt.figure()
        # # for lang, dfg in all_rules.groupby("language"):
        # #     plt.scatter(dfg["I_exp"], dfg["q_residual"], label=lang, alpha=0.7)
        # # plt.scatter(
        # #     pf["I_exp"],
        # #     pf["q_residual"],
        # #     edgecolor="k",
        # #     facecolor="none",
        # #     s=80,
        # #     label="Pareto",
        # # )
        # # plt.xlabel("Interpretability  I_exp")
        # # plt.ylabel("Quality  q_residual")
        # # plt.title("EMM: Quality vs Interpretability across languages")
        # # plt.legend()
        # # plt.tight_layout()
        # # plt.savefig(f"{out_dir}/quality_vs_interpretability.png", dpi=160)
        # # plt.close()

        # # plt.figure()
        # # for lang, dfg in all_rules.groupby("language"):
        # #     plt.scatter(dfg["size"], dfg["mean_residual"], label=lang, alpha=0.7)
        # # plt.xlabel("Subgroup size (coverage)")
        # # plt.ylabel("Mean squared residual in subgroup")
        # # plt.title("Coverage vs Error (by language)")
        # # plt.legend()
        # # plt.tight_layout()
        # # plt.savefig(f"{out_dir}/coverage_vs_error.png", dpi=160)
        # # plt.close()

    # combined stuff:
    all_languages_results = glob.glob(
        "outputs/*/emm_pareto_front_on_all_combined_rules_*.csv",
        recursive=True,
    )

    df = pd.DataFrame()
    for file in all_languages_results:
        dfx = pd.read_csv(file)
        name = file.split("_")[-1].split(".")[0]
        dfx["dataset"] = name
        df = pd.concat([df, dfx])

    print(df["language"].value_counts().sum())
    print(df["dataset"].value_counts().sum())
    df.loc[df["pareto"] == True].to_csv(
        "outputs/emm_all_pareto_front_true.csv", index=False
    )
    df.to_csv("outputs/emm_all_pareto_front.csv", index=False)


if __name__ == "__main__":
    main()
