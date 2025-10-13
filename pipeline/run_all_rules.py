import pandas as pd

from configurations import *
from I_conjunction_rules import *
from II_polynomial_rules import *
from III_d_tree_rules import *
from IV_symbolic_rules import *

df = pd.read_csv(
    "/Users/aniket/github/Exceptional_Model_Mining_2AMM20/pipeline/dataset_with_residuals/boston_housing_with_residuals.csv"
)

print("\n\nL CONJ\n")
results = mine_conj_rules(df, res_col=RES_COL, max_rule_len=3, top_k=TOP_K)
print(f"Global mean residual: {results.attrs['global_mean_residual']:.4f}")
print(results.head(10))
results.to_csv("outputs/1_lconj_results.csv", index=False)


# for prtining in notebook
for i in range(TOP_K):
    print(f"Rule {i + 1}: {results.loc[i, 'rule']}")
#     plot_rule_vs_rest_boxplot(df, results.loc[i, 'rule'], res_col="Residual")

print()
# test top 10 rules
for i in range(TOP_K):
    rule = results.loc[i, "rule"]
    qobs, p = permutation_pvalue(
        df,
        rule,
        res_col="Residual",
        n_perm=NUM_PERMUTATIONS,
        seed=DEFAULT_RANDOM_STATE,
    )
    print(f"Rule {i + 1}: q={qobs:.2f}  p≈{p:.4f}  |  {rule}")


results_cv = mine_conj_rules(
    df, res_col="Residual_CV", max_rule_len=3, top_k=100
)
print(results_cv.head(10))

results_cv.to_csv("outputs/2_lconj_results_cv.csv", index=False)

results_diverse = prune_by_coverage(df, results, min_new=0.4)
print(results_diverse.head(10))

results_diverse.to_csv("outputs/3_lconj_results_diverse.csv", index=False)


## Poly
print("\n\nL POLY\n")

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

print(
    f"Global mean residual: {results_poly.attrs['global_mean_residual']:.4f}"
)
results_poly.head(10)


print("\n\nL DTREE\n")


# Mine L_tree subgroups (depth ≤ 3), score with q_residual, get top-50
results_tree = mine_ltree_rules(
    df,
    res_col="Residual",
    max_depth=3,
    min_leaf=10,  # tune per data size; 10–25 works well on Boston
    top_k=50,
    random_state=42,
)
print(
    f"Global mean residual: {results_tree.attrs['global_mean_residual']:.4f}"
)
results_tree.head(10)


print("rows:", len(results_tree))
print(results_tree.head(3))

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

# Viz top-N
N = min(5, len(results_tree))
for i in range(N):
    rule_i = results_tree.iloc[i]["rule"]  # iloc avoids label issues
    print(f"\nRule {i + 1}: {rule_i}")
    # plot_rule_vs_rest_boxplot(df, rule_i, res_col="Residual")


print("\n\nL SYMB\n")


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

print(
    f"Global mean residual: {results_symb.attrs['global_mean_residual']:.4f}"
)
results_symb.head(20)


for i in range(min(5, len(results_symb))):
    rule = results_symb.iloc[i]["rule"]
    print(f"\nRule {i + 1}: {rule}")
    # plot_rule_vs_rest_boxplot(df, rule, res_col="Residual")
