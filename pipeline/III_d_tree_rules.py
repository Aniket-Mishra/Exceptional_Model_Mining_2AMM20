from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from common_functions import *
from configurations import *
from sklearn.tree import DecisionTreeRegressor

# shallow decision-tree subgroup miner

def _q_residual(
    mask: np.ndarray, resid: np.ndarray, mean_g: float, var_g: float
) -> Tuple[int, float, float]:
    n = int(mask.sum())
    if n == 0:
        return 0, np.nan, -np.inf
    mean_s = resid[mask].mean()
    denom = np.sqrt(var_g / n) if var_g > 0 else 0.0
    q = (abs(mean_s - mean_g) / denom) if denom > 0 else 0.0
    return n, mean_s, q


# --- numeric features only (keeps rules readable) ---
def _numeric_feature_frame(df: pd.DataFrame, exclude: set) -> pd.DataFrame:
    from pandas.api.types import is_numeric_dtype

    cols = [
        c for c in df.columns if c not in exclude and is_numeric_dtype(df[c])
    ]
    X = df[cols].copy()
    # Coerce to float for tree safety
    for c in cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    # X = ( ## ugly deprication warning
    #     X.replace([np.inf, -np.inf], np.nan)
    #     .fillna(method="ffill")
    #     .fillna(method="bfill")
    # )
    X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return X


# --- extract human-readable path for a given leaf ---
def _extract_rule_from_leaf(
    tree: DecisionTreeRegressor, feature_names: List[str], leaf_id: int
) -> Tuple[str, int]:
    t = tree.tree_
    # Reconstruct a path by walking back from leaf: we can do a DFS to find all root->leaf paths once and cache
    paths = []

    def dfs(node_id, path):
        if t.children_left[node_id] == -1 and t.children_right[node_id] == -1:
            # leaf
            paths.append((node_id, path.copy()))
            return
        feat_idx = t.feature[node_id]
        thr = t.threshold[node_id]
        # left: feature <= thr
        left_clause = (feat_idx, "<=", thr)
        dfs(t.children_left[node_id], path + [left_clause])
        # right: feature > thr
        right_clause = (feat_idx, ">", thr)
        dfs(t.children_right[node_id], path + [right_clause])

    # Build all paths once
    if not hasattr(tree, "_all_paths_cache"):
        dfs(0, [])
        tree._all_paths_cache = {leaf: path for leaf, path in paths}

    path = tree._all_paths_cache[leaf_id]
    # Translate to readable clauses; merge same-feature bounds for niceness
    clauses = []
    for feat_idx, op, thr in path:
        feat = feature_names[feat_idx]
        # round threshold for readability
        thr_disp = float(np.round(thr, 4))
        clauses.append(f"{feat} {op} {thr_disp}")
    return " AND ".join(clauses), len(clauses)


# --- main miner ---
def mine_ltree_rules(
    df: pd.DataFrame,
    res_col: str = RES_COL,
    max_depth: int = LTREE_MAX_DEPTH,
    min_leaf: int = LTREE_MIN_LEAF,
    top_k: int = 50,
    random_state: int = LTREE_RANDOM_STATE,
    exclude_cols: Optional[set] = None,
) -> pd.DataFrame:
    if exclude_cols is None:
        exclude_cols = {res_col, "MEDV", "Residual_CV"}
    assert res_col in df.columns, f"Residual column '{res_col}' not found."

    # Build X (numeric-only) and y
    X = _numeric_feature_frame(df, exclude=exclude_cols)
    feature_names = list(X.columns)
    if len(feature_names) == 0:
        raise ValueError("No numeric features available for L_tree mining.")
    y = df[res_col].to_numpy()

    # Fit shallow tree on residuals
    tree = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
        random_state=random_state,
    )
    tree.fit(X, y)

    # leaf_ids = tree.apply(X.to_numpy())
    leaf_ids = tree.apply(X)  # which leaf each sample falls in

    resid = y
    mean_g = float(np.nanmean(resid))
    var_g = float(np.nanvar(resid))

    # Aggregate leaves -> groups
    groups = {}
    for i, lid in enumerate(leaf_ids):
        groups.setdefault(lid, []).append(i)

    # Score each leaf subgroup
    rows = []
    for lid, idxs in groups.items():
        idxs = np.array(idxs, dtype=int)
        if idxs.size < min_leaf:
            continue
        mask = np.zeros(len(df), dtype=bool)
        mask[idxs] = True
        size, mean_s, q = _q_residual(mask, resid, mean_g, var_g)
        rule_str, length = _extract_rule_from_leaf(tree, feature_names, lid)
        rows.append(
            {
                "rule": rule_str,
                "length": length,
                "size": size,
                "mean_residual": mean_s,
                "delta_from_global": mean_s - mean_g,
                "q_residual": q,
                "leaf_id": int(lid),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "rule",
                "length",
                "size",
                "mean_residual",
                "delta_from_global",
                "q_residual",
                "leaf_id",
            ]
        )

    # Rank by q, keep top_k
    out = (
        pd.DataFrame(rows)
        .sort_values("q_residual", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    out.attrs["global_mean_residual"] = mean_g
    out.attrs["global_var_residual"] = var_g
    out.attrs["feature_names"] = feature_names
    out.attrs["tree"] = tree
    return out


# --- optional: mask builder from rule string to reuse existing plot helper ---
def mask_from_rule_string(df: pd.DataFrame, rule_str: str) -> np.ndarray:
    mask = np.ones(len(df), dtype=bool)
    if not rule_str.strip():
        return mask
    for clause in rule_str.split(" AND "):
        feat, op, val = clause.split(" ", 2)
        v = float(val)
        if op == "<=":
            mask &= df[feat].values <= v
        elif op == ">":
            mask &= df[feat].values > v
        else:
            raise ValueError(f"Unsupported operator in L_tree clause: {op}")
    return mask
