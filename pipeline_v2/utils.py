import re
from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from configurations import *
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_numeric_dtype,
    is_object_dtype,
)


@dataclass(frozen=True)
class Predicate:
    feature: str
    op: str  # one of {'<=','>','=='(cat)}  (we only use these)
    value: Any
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


def quantize_threshold(t: float, step: float) -> float:
    if step <= 0:
        return float(t)
    return float(np.round(t / step) * step)


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


def compute_q_residual_signed(
    sub_mask: np.ndarray,
    resid: np.ndarray,
    mean_global: float,
    var_global: float,
) -> Tuple[int, float, float, float]:
    """
    Returns: (n, mean_sub, q_signed, q_abs)
    q_signed = (mean_sub - mean_global) / sqrt(var_global / n)
    q_abs    = |q_signed|
    """
    n = int(sub_mask.sum())
    if n == 0:
        return 0, np.nan, 0.0, 0.0
    mean_sub = resid[sub_mask].mean()
    denom = np.sqrt(var_global / n) if var_global > 0 and n > 0 else 0.0
    if denom == 0.0:
        return n, mean_sub, 0.0, 0.0
    q_signed = (mean_sub - mean_global) / denom
    return n, mean_sub, q_signed, abs(q_signed)


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


def build_numeric_predicates_local(
    df: pd.DataFrame,
    col: str,
    base_mask: np.ndarray,
    n_quantiles: int,
    k_keep: int,
    resid: np.ndarray,
    mean_global: float,
    var_global: float,
    step: float = THRESH_RESOLUTION,
) -> List[Predicate]:
    """
    For the rows currently covered by `base_mask`, propose <= , > predicates
    on `col` using interior quantiles of the *local* distribution only.
    Retain the k_keep thresholds with highest absolute q_residual gain.
    """
    x_all = pd.to_numeric(df[col], errors="coerce").to_numpy()
    x = x_all[base_mask]
    idx_local = np.where(base_mask)[0]
    x = x[np.isfinite(x)]
    if x.size <= 1 or np.unique(x).size <= 1:
        return []

    qs = np.linspace(0, 1, n_quantiles)[1:-1]
    cands = np.unique(np.quantile(x, qs, method="linear"))
    # score each threshold locally using masks restricted to base_mask
    scored = []
    for t in cands:
        t_q = quantize_threshold(float(t), step)
        # build whole-dataset masks, then intersect with base_mask outside
        mask_le = x_all <= t_q
        mask_gt = x_all > t_q
        m_le = base_mask & mask_le
        m_gt = base_mask & mask_gt
        for op, m in (("<=", m_le), (">", m_gt)):
            n, mean_s, q_signed, q_abs = compute_q_residual_signed(
                m, resid, mean_global, var_global
            )
            if n >= MIN_SUPPORT:
                scored.append((q_abs, op, t_q, m))
    # keep best k thresholds (by absolute quality)
    scored.sort(key=lambda z: z[0], reverse=True)
    preds: List[Predicate] = []
    for i, op, t_q, m in scored[:k_keep]:
        preds.append(Predicate(feature=col, op=op, value=t_q, mask=m))
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
        preds.extend(build_numeric_predicates(df, c, LOCAL_QUANTILES))
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

    def boxplot_from_mask(m):
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
        boxplot_from_mask(mask)
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
    boxplot_from_mask(mask)


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
    for i in range(n_perm):
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
    for i, row in results_df.sort_values(
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


# Intepretibility

NUM_PAT = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def count_significant_digits(x_str: str) -> int:
    # counts decimals after '.' in a numeric literal; "800.25" -> 2; "800" -> 0
    if "." in x_str.lower():
        # strip exponent part if present
        base = x_str.lower().split("e")[0]
        return max(0, len(base.split(".")[-1]))
    return 0


def precision_penalty_from_rule(rule_str: str) -> float:
    # Sum 10^{-sig} over all numeric thresholds in the rule text
    penalty = 0.0
    for m in NUM_PAT.finditer(rule_str):
        s = m.group(0)
        sig = count_significant_digits(s)
        penalty += 10.0 ** (-sig) if sig > 0 else 0.0
    return float(penalty)


def count_tokens_symbolic(rule_str: str) -> int:
    # Very simple tokenization: variables/numbers/operators
    ops = [
        "+",
        "-",
        "*",
        "/",
        "<=",
        ">=",
        "<",
        ">",
        "==",
        "=",
        "∧",
        "∨",
        "(",
        ")",
    ]
    # count operators by occurrences, then count numbers and variable names as operands
    op_count = sum(rule_str.count(op) for op in ops)
    # count numeric literals
    nums = len(NUM_PAT.findall(rule_str))
    # very rough variable/identifier count: tokens of letters/underscores not in ops
    idents = len(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", rule_str))
    return op_count + nums + idents


def count_predicates_conj(rule_str: str) -> int:
    # predicates split by '∧' or 'and' or '&'
    parts = re.split(r"\s*(?:∧|and|&&?|,)\s*", rule_str)
    return len([p for p in parts if p.strip()])


def count_operators(rule_str: str) -> int:
    ops = [
        "+",
        "-",
        "*",
        "/",
        "<=",
        # ">=",
        "<",
        ">",
        "==",
        # "=",
        # "∧",
        # "∨",
        # "&&",
        "and",
    ]
    return sum(rule_str.count(op) for op in ops)


def depth_from_tree_rule(rule_str: str) -> int:
    # paths like "a>1 → b<=2 → c>3"; depth = number of edges
    if "→" in rule_str:
        return max(0, rule_str.count("→"))
    # else approximate by number of predicates - 1
    return max(0, count_predicates_conj(rule_str) - 1)


def splits_from_tree_rule(rule_str: str) -> int:
    # internal nodes count equals number of arrows for a single path
    return rule_str.count("→")


def poly_terms_and_degree(rule_str: str) -> tuple[int, int]:
    # terms separated by '+' or '-' not preceded by 'e'
    # Count occurrences of f^2 or f2 to approximate degree 2
    terms = re.split(
        r"(?<!e)([+-])", rule_str
    )  # avoid scientific notation splits
    n_terms = max(
        1,
        sum(
            1
            for t in terms
            if isinstance(t, str) and t.strip() and t.strip() not in ["+", "-"]
        ),
    )
    # degree: look for ^2, **2, or patterns like f2 - approx
    deg2 = bool(
        re.search(r"(\^2|\*\*2|[A-Za-z_][A-Za-z0-9_]*\s*2\b)", rule_str)
    )
    degree = 2 if deg2 else 1
    return n_terms, degree


def annotate_interpretability(df: pd.DataFrame, language: str) -> pd.DataFrame:
    """
    Adds columns:
      n_predicates, n_operators, depth, n_splits, precision_penalty,
      n_terms, degree, n_tokens,
      complexity, I_inv, I_exp, expressiveness
    """
    if df is None or len(df) == 0:
        return df
    rs = df["rule"].astype(str).tolist()

    n_predicates = []
    n_ops = []
    depth = []
    n_splits = []
    prec = []
    n_terms = []
    degree = []
    n_tokens = []

    for r in rs:
        n_ops.append(count_operators(r))
        prec.append(precision_penalty_from_rule(r))

        if language == "conj":
            n_predicates.append(count_predicates_conj(r))
            depth.append(0)
            n_splits.append(0)
            n_terms.append(0)
            degree.append(0)
            n_tokens.append(0)

        elif language == "tree":
            d = depth_from_tree_rule(r)
            s = splits_from_tree_rule(r)
            depth.append(d)
            n_splits.append(s)
            n_predicates.append(max(1, d + 1))
            n_terms.append(0)
            degree.append(0)
            n_tokens.append(0)

        elif language == "poly":
            t, deg = poly_terms_and_degree(r)
            n_terms.append(t)
            degree.append(deg)
            n_predicates.append(1)  # polynomial predicate <= τ
            depth.append(0)
            n_splits.append(0)
            n_tokens.append(0)

        elif language == "symb":
            tok = count_tokens_symbolic(r)
            n_tokens.append(tok)
            n_predicates.append(1)  # expression ⟂ τ
            depth.append(0)
            n_splits.append(0)
            n_terms.append(0)
            degree.append(0)

        else:
            # treat like conjunction
            n_predicates.append(count_predicates_conj(r))
            depth.append(0)
            n_splits.append(0)
            n_terms.append(0)
            degree.append(0)
            n_tokens.append(0)

    df = df.copy()
    df["n_predicates"] = n_predicates
    df["n_operators"] = n_ops
    df["depth"] = depth
    df["n_splits"] = n_splits
    df["precision_penalty"] = prec
    df["n_terms"] = n_terms
    df["degree"] = degree
    df["n_tokens"] = n_tokens

    # complexity(d) = w1*#pred + w2*#ops + w3*depth + w4*precision
    df["complexity"] = (
        W_PREDICATES * df["n_predicates"].astype(float)
        + W_OPERATORS * df["n_operators"].astype(float)
        + W_DEPTH * df["depth"].astype(float)
        + W_PRECISION * df["precision_penalty"].astype(float)
    )

    df["I_inv"] = 1.0 / (1.0 + df["complexity"])
    df["I_exp"] = np.exp(-I_DECAY_BETA * df["complexity"].astype(float))

    # expressiveness proxy per language
    if language == "conj":
        df["expressiveness"] = df["n_predicates"].astype(float)
    elif language == "tree":
        df["expressiveness"] = df["depth"].astype(float) + df[
            "n_splits"
        ].astype(float)
    elif language == "poly":
        df["expressiveness"] = df["n_terms"].astype(float) + df[
            "degree"
        ].astype(float)
    elif language == "symb":
        df["expressiveness"] = df["n_tokens"].astype(float)
    else:
        df["expressiveness"] = df["n_predicates"].astype(float)

    return df


def pareto_mark(
    df: pd.DataFrame, q_col="q_residual", i_col="I_exp"
) -> pd.DataFrame:
    """
    Adds a boolean column 'pareto' marking rules that are non-dominated
    when maximizing both q_col and i_col.
    """
    if df is None or len(df) == 0:
        return df
    arr = df[[q_col, i_col]].to_numpy(dtype=float)
    n = len(arr)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        qi, ii = arr[i]
        # dominated if exists j with q>= and I>= and at least one strict >
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
    """
    Pareto only. Returns df with:
      - pareto (bool)
      - score  (geometric tie-break: sqrt(q_residual * I_exp))
    Sorted with pareto first, then score descending.
    """
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
