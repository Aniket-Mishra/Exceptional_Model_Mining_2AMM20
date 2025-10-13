from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

from configurations import *


@dataclass(frozen=True)
class Predicate:
    feature: str
    op: str  # one of {'<=','>','=='}  (we only use these)
    value: Any  # threshold or category value
    mask: np.ndarray  # boolean array over df.index

    def describe(self) -> str:
        if self.op in ("<=", ">"):
            v = (
                f"{self.value:.4g}"
                if isinstance(self.value, (int, float, np.floating))
                else str(self.value)
            )
            return f"{self.feature} {self.op} {v}"
        return f"{self.feature} == {self.value}"


@dataclass(frozen=True)
class Rule:
    preds: Tuple[Predicate, ...]  # ordered tuple of predicates
    mask: np.ndarray  # boolean array over df.index
    size: int  # support (|S|)
    mean_resid: float  # mean residual in S
    q: float  # q_residual score

    def describe(self) -> str:
        return " AND ".join(p.describe() for p in self.preds)


def compute_q_residual(  # quality compute
    sub_mask: np.ndarray,
    resid: np.ndarray,
    mean_global: float,
    var_global: float,
) -> Tuple[int, float, float]:
    n = int(sub_mask.sum())
    if n == 0:
        return 0, np.nan, -np.inf
    mean_sub = resid[sub_mask].mean()
    denom = np.sqrt(var_global / n) if var_global > 0 and n > 0 else 0.0
    q = (abs(mean_sub - mean_global) / denom) if denom > 0 else 0.0
    return n, mean_sub, q


def infer_feature_types(
    df: pd.DataFrame,
) -> Tuple[List[str], List[str]]:  # gen predicate
    """Return (numeric_cols, categorical_cols) excluding EXCLUDE_COLS."""
    num_cols, cat_cols = [], []
    for c in df.columns:
        if c in EXCLUDE_COLS:
            continue
        s = df[c]
        if (
            is_categorical_dtype(s)
            or is_bool_dtype(s)
            or (CATEGORICAL_AS_OBJECT and is_object_dtype(s))
        ):
            cat_cols.append(c)
        elif is_numeric_dtype(s):
            num_cols.append(c)
        else:
            # ftry coercion to numeric, it makes things nan
            try:
                pd.to_numeric(s)
                num_cols.append(c)
            except Exception:
                cat_cols.append(c)
    return num_cols, cat_cols


def build_numeric_predicates(
    df: pd.DataFrame, col: str, n_quantiles: int
) -> List[Predicate]:
    """Numeric thresholds at interior quantiles; builds <= and > predicates."""
    x = pd.to_numeric(df[col], errors="coerce").to_numpy()
    finite = np.isfinite(x)
    x_f = x[finite]
    if x_f.size <= 1 or np.unique(x_f).size <= 1:
        return []
    # interior quantiles exclude 0 and 1aaa
    qs = np.linspace(0, 1, n_quantiles)[1:-1]
    thresholds = np.unique(np.quantile(x_f, qs, method="linear"))
    preds: List[Predicate] = []
    # precompute full masks once per threshold
    xv = x  # nans as False
    for t in thresholds:
        mask_le = xv <= t
        mask_gt = xv > t
        preds.append(
            Predicate(feature=col, op="<=", value=float(t), mask=mask_le)
        )
        preds.append(
            Predicate(feature=col, op=">", value=float(t), mask=mask_gt)
        )
    return preds


def build_categorical_predicates(
    df: pd.DataFrame, col: str, max_card: int = 30
) -> List[Predicate]:
    """Equality tests for categories with reasonable cardinality."""
    s = df[col]
    # For cat, use categories and value_counts forothers
    if is_categorical_dtype(s):
        cats = list(s.cat.categories)
    else:
        cats = list(s.value_counts(dropna=False).index)
    if len(cats) > max_card:
        return []
    preds: List[Predicate] = []
    for v in cats:
        mask = s.values == v
        # to exclude predicates with low support later
        preds.append(Predicate(feature=col, op="==", value=v, mask=mask))
    return preds


def build_all_predicates(df: pd.DataFrame) -> List[Predicate]:
    num_cols, cat_cols = infer_feature_types(df)
    preds: List[Predicate] = []
    for c in num_cols:
        preds.extend(build_numeric_predicates(df, c, N_QUANTILES))
    for c in cat_cols:
        preds.extend(build_categorical_predicates(df, c))
    pruned = [
        p for p in preds if int(p.mask.sum()) >= MIN_SUPPORT
    ]  # remove smol supports
    return pruned


def plot_rule_vs_rest_boxplot(df, rule_str, res_col="Residual"):
    """
    Supports:
      1) Conjunctive rules: "feat1 <= a AND feat2 > b AND feat3 == c"
      2) Symbolic rules:    "<expression> <= thr" or "<expression> > thr"
         where <expression> uses valid column names and +,-,*,/ and parentheses,
         e.g., "(LSTAT / RM) - DIS > -0.299116"
    """
    rule_str = rule_str.strip()
    mask = np.ones(len(df), dtype=bool)

    def _boxplot_from_mask(m):
        data = [df.loc[m, res_col].values, df.loc[~m, res_col].values]
        labels = ["Subgroup", "Rest"]
        plt.figure()
        plt.boxplot(data, labels=labels, showfliers=True)
        plt.title("Residuals: Subgroup vs Rest")
        plt.ylabel(res_col)
        plt.show()

    # Heuristic: if there's AND then treat as conjunctive
    if " AND " in rule_str:
        # Conjunctive case (existing behavior)
        for clause in rule_str.split(" AND "):
            feat, op, val = clause.split(" ", 3)[:3]
            if op == "==":
                v = val
                try:
                    v = type(df[feat].dropna().iloc[0])(eval(val))
                except Exception:
                    pass  # leave as string
                mask &= df[feat].values == v
            else:
                v = float(val)
                if op == "<=":
                    mask &= df[feat].values <= v
                elif op == ">":
                    mask &= df[feat].values > v
                else:
                    raise ValueError(f"Unsupported operator: {op}")
        _boxplot_from_mask(mask)
        return

    # else treat as a single symbolic expression op threshold
    # look for the last occur of "<=" or ">" to split expr and threshold.
    if " <= " in rule_str:
        expr_str, thr_str = rule_str.rsplit(" <= ", 1)
        op = "<="
    elif " > " in rule_str:
        expr_str, thr_str = rule_str.rsplit(" > ", 1)
        op = ">"
    else:
        raise ValueError(f"Unrecognized rule format: {rule_str}")

    # map valid column name to its numpy array
    local_ns = {col: df[col].to_numpy() for col in df.columns}
    try:
        vals = eval(expr_str, {"__builtins__": {}}, local_ns)
    except Exception as e:
        raise ValueError(f"Failed to eval expr '{expr_str}': {e}")

    vals = np.asarray(vals, dtype=float)
    thr = float(thr_str)
    mask = (vals <= thr) if op == "<=" else (vals > thr)
    _boxplot_from_mask(mask)


# test pvalue on rules
def q_residual_for_mask(mask, resid, mean_g, var_g):
    n = int(mask.sum())
    if n == 0 or var_g == 0:
        return 0.0
    mean_s = resid[mask].mean()
    return abs(mean_s - mean_g) / np.sqrt(var_g / n)


def permutation_pvalue(df, rule_str, res_col="Residual", n_perm=2000, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.ones(len(df), dtype=bool)
    for clause in rule_str.split(" AND "):
        feat, op, val = clause.split(" ", 2)
        if op == "==":
            try:
                v = type(df[feat].dropna().iloc[0])(eval(val))
            except Exception:
                v = val
            mask &= df[feat].values == v
        else:
            v = float(val)
            mask &= (
                (df[feat].values <= v) if op == "<=" else (df[feat].values > v)
            )

    resid = df[res_col].to_numpy()
    mean_g = float(np.nanmean(resid))
    var_g = float(np.nanvar(resid))
    q_obs = q_residual_for_mask(mask, resid, mean_g, var_g)

    greater = 0
    for _ in range(n_perm):
        resid_perm = resid.copy()
        rng.shuffle(resid_perm)
        q_perm = q_residual_for_mask(mask, resid_perm, mean_g, var_g)
        greater += q_perm >= q_obs
    pval = (greater + 1) / (n_perm + 1)
    return q_obs, pval


def prune_by_coverage(df, results_df, res_col="Residual", min_new=0.3):
    """Keep a rule if ≥ min_new fraction of its covered points are not yet covered."""
    kept = []
    covered = np.zeros(len(df), dtype=bool)
    for _, row in results_df.sort_values(
        "q_residual", ascending=False
    ).iterrows():
        # build mask
        mask = np.ones(len(df), dtype=bool)
        for clause in row["rule"].split(" AND "):
            feat, op, val = clause.split(" ", 2)
            if op == "==":
                try:
                    v = type(df[feat].dropna().iloc[0])(eval(val))
                except Exception:
                    v = val
                mask &= df[feat].values == v
            else:
                v = float(val)
                mask &= (
                    (df[feat].values <= v)
                    if op == "<="
                    else (df[feat].values > v)
                )
        n = int(mask.sum())
        if n == 0:
            continue
        new_frac = (~covered & mask).sum() / n
        if new_frac >= min_new:
            kept.append(row)
            covered |= mask
    return pd.DataFrame(kept).reset_index(drop=True)
