import numpy as np
import pandas as pd
from common_functions import annotate_interpretability, permutation_pvalue
from configurations import *
from I_conjunction_rules import *
from II_polynomial_rules import *
from III_d_tree_rules import *
from IV_symbolic_rules import *


def split_directional(df_rules: pd.DataFrame):
    if not SPLIT_DIRECTIONAL_LISTS or len(df_rules) == 0:
        return df_rules, pd.DataFrame(columns[df_rules.columns])
    under = df_rules[df_rules["delta_from_global"] > 0].copy()
    over = df_rules[df_rules["delta_from_global"] < 0].copy()
    # sort both by absolute q_residual desc
    under = under.sort_values("q_residual", ascending=False).reset_index(
        drop=True
    )
    over = over.sort_values("q_residual", ascending=False).reset_index(
        drop=True
    )
    return under, over


def add_direction_cols(df):
    """
    Adds:
      - q_signed (recomputed if missing): (mean_sub - mean_global) / sqrt(var_global / size)
      - direction: 'under' if q_signed > 0, 'over' if q_signed < 0, else 'neutral'
    Falls back to sign(delta_from_global) if attrs/size are missing.
    """
    if df is None or len(df) == 0:
        return df

    # read global stats from df attrs set by the miners
    mean_g = df.attrs.get("global_mean_residual", None)
    var_g = df.attrs.get("global_var_residual", None)

    # Ensure columns exist
    if "delta_from_global" not in df.columns:
        # If not present, derive from mean_residual - mean_g (when we have mean_g)
        if mean_g is not None and "mean_residual" in df.columns:
            df["delta_from_global"] = df["mean_residual"] - float(mean_g)
        else:
            df["delta_from_global"] = np.nan

    # q_signed: recompute if missing and we have the pieces
    if "q_signed" not in df.columns:
        can_compute = (
            ("size" in df.columns)
            and (var_g is not None)
            and np.isfinite(var_g)
        )
        if can_compute:
            denom = np.sqrt(
                np.maximum(var_g, 0)
                / np.maximum(df["size"].astype(float), 1.0)
            )
            # Avoid zero division
            denom = denom.replace(0, np.nan)
            df["q_signed"] = df["delta_from_global"].astype(float) / denom
        else:
            df["q_signed"] = (
                np.nan
            )  # will rely on delta_from_global sign for direction

    # direction
    # q_signed when available else delta sign
    use = df["q_signed"].where(
        np.isfinite(df["q_signed"]), df["delta_from_global"]
    )
    df["direction"] = np.where(
        use > 0, "under", np.where(use < 0, "over", "neutral")
    )

    return df


def _pareto_mark(
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
    d = _pareto_mark(df, q_col="q_residual", i_col="I_exp")
    d["score"] = np.sqrt(
        d["q_residual"].astype(float) * d["I_exp"].astype(float)
    )
    d = d.sort_values(
        ["pareto", "score"], ascending=[False, False]
    ).reset_index(drop=True)
    return d


def _prepare(df, lang_tag):
    df = add_direction_cols(df)
    df = annotate_interpretability(df, language=lang_tag)
    df = apply_pareto_tradeoff(df)
    return df.head(TOP_K_PER_LANGUAGE).reset_index(drop=True)


def add_permutation_pvalues(
    df_rules, data_frame, res_col=RES_COL, language_tag="conj"
):
    if df_rules is None or len(df_rules) == 0:
        return df_rules
    df_rules = df_rules.copy()
    pvals = []
    qobs_list = []
    if language_tag == "symb" and {"expr", "operator", "threshold"}.issubset(
        df_rules.columns
    ):
        # evaluate symbolic expressions safely via a temporary column and reuse permutation_pvalue
        for i in range(len(df_rules)):
            expr_str = str(df_rules.loc[i, "expr"])
            op = str(df_rules.loc[i, "operator"])
            thr = float(df_rules.loc[i, "threshold"])
            tmp_df = data_frame.copy()
            tmp_col = f"__symb_expr_{i}"
            # compute expression as a column
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


df = pd.read_csv(
    "/Users/aniket/github/Exceptional_Model_Mining_2AMM20/pipeline/dataset_with_residuals/boston_housing_with_residuals.csv"
)

results = mine_conj_rules(
    df, res_col=RES_COL, max_rule_len=3, top_k=TOP_K_PER_LANGUAGE
)
results_cv = mine_conj_rules(
    df, res_col="Residual_CV", max_rule_len=3, top_k=100
)
results_diverse = prune_by_coverage(df, results, min_new=0.4)

df_poly = augment_with_poly(
    df,
    res_col="Residual",
    max_squares=8,  # 2–12 is typical
    max_interactions=8,
    candidate_feature_limit=10,
)

results_poly = mine_conj_rules(
    df_poly,
    res_col="Residual",
    max_rule_len=2,  # recommended for poly
    top_k=50,
)

results_tree = mine_ltree_rules(
    df,
    res_col="Residual",
    max_depth=3,
    min_leaf=10,  # tune per data size; 10–25 works well on Boston
    top_k=50,
    random_state=42,
)
# If empty, try relaxing the tree params and re-mine
if len(results_tree) == 0:
    results_tree = mine_ltree_rules(
        df,
        res_col="Residual",  # or "Residual_CV"
        max_depth=3,  # try 3 or 4
        min_leaf=5,  # relax from 10 -> 5
        top_k=50,
        random_state=42,
    )
    print("re-mined rows:", len(results_tree))

results_symb = mine_lsymb_rules_light(
    df,
    res_col="Residual",
    feature_exclude={"Residual", "Residual_CV"},
    top_features=8,
    max_triplets=40,
    thresholds_per_expr=3,
    min_support=8,
    top_k=50,
)

results = _prepare(results, "conj")
results_poly = _prepare(results_poly, "poly")
results_tree = _prepare(results_tree, "tree")
results_symb = _prepare(results_symb, "symb")

# annotate interpretability + expressiveness
results = annotate_interpretability(results, language="conj")
results_poly = annotate_interpretability(results_poly, language="poly")
results_tree = annotate_interpretability(results_tree, language="tree")
results_symb = annotate_interpretability(results_symb, language="symb")

# permutation p-values
results = add_permutation_pvalues(results, df, RES_COL, language_tag="conj")
results_poly = add_permutation_pvalues(
    results_poly, df_poly, RES_COL, language_tag="poly"
)
results_tree = add_permutation_pvalues(
    results_tree, df, RES_COL, language_tag="tree"
)
results_symb = add_permutation_pvalues(
    results_symb, df, RES_COL, language_tag="symb"
)

results.to_csv("outputs/1_lconj.csv", index=False)
results_poly.to_csv("outputs/2_lpoly.csv", index=False)
results_tree.to_csv("outputs/3_ltree.csv", index=False)
results_symb.to_csv("outputs/4_lsymb.csv", index=False)

all_rules = (
    pd.concat(
        [
            results.assign(language="conj"),
            results_poly.assign(language="poly"),
            results_tree.assign(language="tree"),
            results_symb.assign(language="symb"),
        ],
        ignore_index=True,
    )
    .sort_values(["language", "q_residual"], ascending=[True, False])
    .reset_index(drop=True)
)
all_rules.to_csv(
    "outputs/_all_languages_top25_with_I_and_expr.csv", index=False
)


# under_conj, over_conj = split_directional(results)
# under_conj.to_csv("outputs/1a_lconj_under.csv", index=False)
# over_conj.to_csv("outputs/1b_lconj_over.csv", index=False)

# under_poly, over_poly = split_directional(results_poly)
# under_poly.to_csv("outputs/1a_lpoly_under.csv", index=False)
# over_poly.to_csv("outputs/1b_lpoly_over.csv", index=False)

# under_tree, over_tree = split_directional(results_tree)
# under_tree.to_csv("outputs/1a_ltree_under.csv", index=False)
# over_tree.to_csv("outputs/1b_ltree_over.csv", index=False)

# under_symb, over_symb = split_directional(results_symb)
# under_symb.to_csv("outputs/1a_lsymb_under.csv", index=False)
# over_symb.to_csv("outputs/1b_lsymb_over.csv", index=False)
