# ===== Dependency-free L_symb (symbolic-expression) subgroup miner =====
# Builds simple expressions over numeric features and mines threshold rules using q_residual.

from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from common_functions import *
from pandas.api.types import is_numeric_dtype


def _numeric_frame(df: pd.DataFrame, exclude: set) -> pd.DataFrame:
    cols = [
        c for c in df.columns if c not in exclude and is_numeric_dtype(df[c])
    ]
    X = df[cols].copy()
    for c in cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return X


def _rank_features_by_assoc(
    X: pd.DataFrame, y: np.ndarray, k: int
) -> List[str]:
    # rank by |corr| with residuals n fall back to variance if constant
    scores = []
    for c in X.columns:
        x = X[c].to_numpy()
        if np.std(x) == 0 or np.std(y) == 0:
            s = 0.0
        else:
            s = abs(np.corrcoef(x, y)[0, 1])
        scores.append((c, float(s)))
    scores.sort(key=lambda t: t[1], reverse=True)
    return [c for c, _ in scores[: max(3, min(k, len(scores)))]]


def _eps(x: np.ndarray) -> float:
    # small stabilizer scaled to data
    s = float(np.nanstd(x))
    return 1e-8 if s == 0 else 1e-6 * s


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        res = a / np.where(b == 0, np.nan, b)
    # replace inf/nan from division by small stabilized denom
    bad = ~np.isfinite(res)
    if bad.any():
        res[bad] = a[bad] / (b[bad] + _eps(b))
    return res


def _gen_expressions(
    X: pd.DataFrame, bases: List[str], max_triplets: int = 30
) -> List[Tuple[str, Callable[[Dict[str, np.ndarray]], np.ndarray], int]]:
    """
    Returns list of (expr_str, evaluator, n_ops) with ≤3 operators (depth ≤ 2).
    Operators allowed: +, -, *, / ; forms:
      unary:        f
      binary:       f1 + f2, f1 - f2, f1 * f2, f1 / f2
      nested (≤3 ops): (f1+f2)/f3, (f1-f2)/f3, (f1*f2)/f3, (f1/f2)+f3, (f1/f2)-f3
    """
    exprs = []

    # map feature -> vector
    F = {c: X[c].to_numpy() for c in bases}

    # unary
    for a in bases:
        exprs.append((f"{a}", (lambda a=a: (lambda F: F[a]))(), 0))

    # binary
    for i, a in enumerate(bases):
        for j, b in enumerate(bases):
            if j <= i:
                continue
            exprs += [
                (
                    f"({a} + {b})",
                    (lambda a=a, b=b: (lambda F: F[a] + F[b]))(),
                    1,
                ),
                (
                    f"({a} - {b})",
                    (lambda a=a, b=b: (lambda F: F[a] - F[b]))(),
                    1,
                ),
                (
                    f"({b} - {a})",
                    (lambda a=a, b=b: (lambda F: F[b] - F[a]))(),
                    1,
                ),
                (
                    f"({a} * {b})",
                    (lambda a=a, b=b: (lambda F: F[a] * F[b]))(),
                    1,
                ),
                (
                    f"({a} / {b})",
                    (lambda a=a, b=b: (lambda F: _safe_div(F[a], F[b])))(),
                    1,
                ),
                (
                    f"({b} / {a})",
                    (lambda a=a, b=b: (lambda F: _safe_div(F[b], F[a])))(),
                    1,
                ),
            ]

    # nested (≤3 ops)
    nested = []
    # pick up to max_triplets distinct triplets to control explosion
    triplets = []
    for a in bases:
        for b in bases:
            if b == a:
                continue
            for c in bases:
                if c == a or c == b:
                    continue
                triplets.append((a, b, c))
    triplets = triplets[:max_triplets]

    for a, b, c in triplets:
        nested += [
            (
                f"({a} + {b}) / {c}",
                (
                    lambda a=a, b=b, c=c: (
                        lambda F: _safe_div(F[a] + F[b], F[c])
                    )
                )(),
                2,
            ),
            (
                f"({a} - {b}) / {c}",
                (
                    lambda a=a, b=b, c=c: (
                        lambda F: _safe_div(F[a] - F[b], F[c])
                    )
                )(),
                2,
            ),
            (
                f"({a} * {b}) / {c}",
                (
                    lambda a=a, b=b, c=c: (
                        lambda F: _safe_div(F[a] * F[b], F[c])
                    )
                )(),
                2,
            ),
            (
                f"({a} / {b}) + {c}",
                (
                    lambda a=a, b=b, c=c: (
                        lambda F: _safe_div(F[a], F[b]) + F[c]
                    )
                )(),
                2,
            ),
            (
                f"({a} / {b}) - {c}",
                (
                    lambda a=a, b=b, c=c: (
                        lambda F: _safe_div(F[a], F[b]) - F[c]
                    )
                )(),
                2,
            ),
        ]
    exprs.extend(nested)

    # de-duplicate by string
    seen = set()
    uniq = []
    for s, fn, nops in exprs:
        if s not in seen:
            uniq.append((s, fn, nops))
            seen.add(s)
    return uniq


def mine_lsymb_rules_light(
    df: pd.DataFrame,
    res_col: str = "Residual",
    feature_exclude: set = {"Residual", "Residual_CV", "MEDV"},
    top_features: int = 8,  # shortlist features by |corr| with residual
    max_triplets: int = 40,  # limit nested expr combos
    thresholds_per_expr: int = 3,  # number of quantile thresholds per expr
    min_support: int = 10,
    top_k: int = 50,
) -> pd.DataFrame:
    assert res_col in df.columns, f"Residual column '{res_col}' not found."
    X = _numeric_frame(df, exclude=feature_exclude | {res_col})
    y = df[res_col].to_numpy()
    mean_g = float(np.nanmean(y))
    var_g = float(np.nanvar(y))

    # shortlist features most associated with residuals
    bases = _rank_features_by_assoc(X, y, k=top_features)
    if len(bases) < 2:
        raise ValueError(
            "Not enough numeric features to build symbolic expressions."
        )

    # generate simple expr (≤3 operators)
    exprs = _gen_expressions(X, bases, max_triplets=max_triplets)

    rows = []
    for expr_str, fn, nops in exprs:
        try:
            vals = fn({c: X[c].to_numpy() for c in X.columns})
        except Exception:
            continue
        if vals is None:
            continue
        vals = np.asarray(vals, dtype=float)
        if not np.isfinite(vals).any():
            continue
        # interior quantile thresholds
        finite_vals = vals[np.isfinite(vals)]
        if finite_vals.size < min_support:
            continue
        qs = np.linspace(0, 1, thresholds_per_expr + 2)[
            1:-1
        ]  # 25/50/75 for 3
        thr_list = np.unique(np.quantile(finite_vals, qs, method="linear"))
        for thr in thr_list:
            for op in ("<=", ">"):
                mask = (vals <= thr) if op == "<=" else (vals > thr)
                n, mean_s, q = compute_q_residual(mask, y, mean_g, var_g)
                if n < min_support:
                    continue
                rows.append(
                    {
                        "rule": f"{expr_str} {op} {float(thr):.6g}",
                        "size": n,
                        "length": nops + 1,  # crude complexity proxy
                        "mean_residual": mean_s,
                        "delta_from_global": mean_s - mean_g,
                        "q_residual": q,
                        "expr": expr_str,
                        "operator": op,
                        "threshold": float(thr),
                        "n_ops": nops,
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "rule",
                "size",
                "length",
                "mean_residual",
                "delta_from_global",
                "q_residual",
                "expr",
                "operator",
                "threshold",
                "n_ops",
            ]
        )

    out = (
        pd.DataFrame(rows)
        .sort_values("q_residual", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    out.attrs["global_mean_residual"] = mean_g
    out.attrs["global_var_residual"] = var_g
    out.attrs["bases_used"] = bases
    return out
