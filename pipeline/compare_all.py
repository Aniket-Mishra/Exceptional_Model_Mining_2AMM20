import re

import matplotlib.pyplot as plt
import pandas as pd
from common_functions import *
from configurations import *
from I_conjunction_rules import *
from II_polynomial_rules import *
from III_d_tree_rules import *
from IPython.display import display
from IV_symbolic_rules import *

df = pd.read_csv(
    "/Users/aniket/github/Exceptional_Model_Mining_2AMM20/pipeline/dataset_with_residuals/boston_housing_with_residuals.csv"
)


# L_conj
results_conj = mine_conj_rules(
    df, res_col="Residual", max_rule_len=3, top_k=80
)

# L_poly (degree ≤ 2 basis + readability guards already installed)
df_poly = augment_with_poly(
    df,
    res_col="Residual",
    max_squares=8,
    max_interactions=8,
    candidate_feature_limit=10,
)
results_poly = mine_conj_rules(
    df_poly, res_col="Residual", max_rule_len=2, top_k=80
)

# L_tree
results_tree = mine_ltree_rules(
    df,
    res_col="Residual",
    max_depth=3,
    min_leaf=8,
    top_k=80,
    random_state=42,
)

# L_symb
results_symb = mine_lsymb_rules_light(
    df,
    res_col="Residual",
    feature_exclude={"Residual", "Residual_CV", "MEDV"},
    top_features=8,
    max_triplets=40,
    thresholds_per_expr=3,
    min_support=8,
    top_k=80,
)


# guard empties
def _nonempty(x):
    return (
        x
        if (isinstance(x, pd.DataFrame) and len(x) > 0)
        else pd.DataFrame(
            columns=[
                "rule",
                "length",
                "size",
                "mean_residual",
                "delta_from_global",
                "q_residual",
            ]
        )
    )


results_conj = _nonempty(results_conj)
results_poly = _nonempty(results_poly)
results_tree = _nonempty(results_tree)
results_symb = _nonempty(results_symb)

# standardize, complexity and interpretability


def _tokens(expr: str) -> int:
    # rough token count for expressions/rules (operands + operators)
    # counts names, numbers, and operators + - * / <= > == ( )
    return len(re.findall(r"[A-Za-z_]\w*|[-+/*()<>]=?|==|[\d.]+", expr))


def add_meta(df_rules: pd.DataFrame, lang: str) -> pd.DataFrame:
    if len(df_rules) == 0:
        return df_rules
    out = df_rules.copy()
    out["language"] = lang

    # Complexity models per language (simple, auditable)
    if lang == "L_conj":
        # number of predicates in conjunction (we stored 'length'); fallback to count ' AND '
        if "length" in out.columns and out["length"].notna().all():
            comp = out["length"].astype(int)
        else:
            comp = out["rule"].str.count(r"\bAND\b") + 1
        out["complexity"] = comp

    elif lang == "L_poly":
        # reuse length if present, otherwise #clauses n small penalty for polynomial symbols (^2 or *)
        base = (out["rule"].str.count(r"\bAND\b") + 1).astype(int)
        poly_pen = out["rule"].str.count(r"\^2|\*").astype(int)
        out["complexity"] = base + poly_pen

    elif lang == "L_tree":
        # path length = number of threshold clauses
        path_len = (out["rule"].str.count(r"\bAND\b") + 1).astype(int)
        out["complexity"] = path_len

    elif lang == "L_symb":
        # operators + operands proxy via token count, shoudl be smol
        # also 1 for the threshold comparison
        out["complexity"] = out["rule"].apply(_tokens)

    # Interpretability I(d) = 1/(1+complexity), if high, more interpt
    out["interpretability"] = 1.0 / (1.0 + out["complexity"].astype(float))
    return out


C_conj = add_meta(results_conj, "L_conj")
C_poly = add_meta(results_poly, "L_poly")
C_tree = add_meta(results_tree, "L_tree")
C_symb = add_meta(results_symb, "L_symb")

# Merge n rank comparison
all_rules = pd.concat([C_conj, C_poly, C_tree, C_symb], ignore_index=True)

all_under = all_rules[all_rules["delta_from_global"] > 0].copy()
all_over = all_rules[all_rules["delta_from_global"] < 0].copy()
all_under = all_under.sort_values("q_residual", ascending=False).reset_index(
    drop=True
)
all_over = all_over.sort_values("q_residual", ascending=False).reset_index(
    drop=True
)

all_under.to_csv("emm_all_languages_underperform.csv", index=False)
all_over.to_csv("emm_all_languages_overperform.csv", index=False)

all_rules = all_rules.sort_values("q_residual", ascending=False).reset_index(
    drop=True
)

print("Top 10 by quality (q_residual):")
display(
    all_rules[
        [
            "language",
            "q_residual",
            "size",
            "mean_residual",
            "delta_from_global",
            "complexity",
            "interpretability",
            "rule",
        ]
    ].head(10)
)

# Save for paper
all_rules.to_csv("emm_all_languages_results.csv", index=False)


# pareto front (q vs I)


def pareto_front(
    df: pd.DataFrame, q_col="q_residual", i_col="interpretability"
):
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
    return df.iloc[idxs].sort_values([q_col, i_col], ascending=[False, False])


pf = pareto_front(all_rules)
pf.to_csv("emm_pareto_front.csv", index=False)

print("\nPareto front (q vs interpretability):")
display(
    pf[
        [
            "language",
            "q_residual",
            "interpretability",
            "complexity",
            "size",
            "rule",
        ]
    ].head(20)
)

# viz

# quality vs interpretability by language
plt.figure()
for lang, dfg in all_rules.groupby("language"):
    plt.scatter(
        dfg["interpretability"], dfg["q_residual"], label=lang, alpha=0.7
    )
plt.scatter(
    pf["interpretability"],
    pf["q_residual"],
    edgecolor="k",
    facecolor="none",
    s=80,
    label="Pareto",
)
plt.xlabel("Interpretability  I(d) = 1/(1+complexity)")
plt.ylabel("Quality  q_residual(d)")
plt.title("EMM: Quality vs Interpretability across languages")
plt.legend()
plt.show()

# coverage vs mean residual, colored by language
plt.figure()
for lang, dfg in all_rules.groupby("language"):
    plt.scatter(dfg["size"], dfg["mean_residual"], label=lang, alpha=0.7)
plt.xlabel("Subgroup size (coverage)")
plt.ylabel("Mean squared residual in subgroup")
plt.title("Coverage vs Error (by language)")
plt.legend()
plt.show()
