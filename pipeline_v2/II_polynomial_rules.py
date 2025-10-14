from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from configurations import *
from pandas.api.types import (
    is_numeric_dtype,
)
from sklearn.feature_selection import mutual_info_regression
from utils import *


def numeric_cols_for_poly(
    df: pd.DataFrame, exclude: Iterable[str]
) -> List[str]:
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def safe_name_sq(c: str) -> str:
    return f"{c}^2"


def safe_name_ix(a: str, b: str) -> str:
    aa, bb = sorted([a, b])
    return f"{aa}*{bb}"


def mi_or_corr(x: np.ndarray, y: np.ndarray, rng=None) -> float:
    # Mutual information for ranking, |corr| if MI fails
    try:
        mi = mutual_info_regression(x.reshape(-1, 1), y, random_state=0)
        return float(mi[0])
    except Exception:
        # Pearson |corr|
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        return float(abs(np.corrcoef(x, y)[0, 1]))


def augment_with_poly(
    df: pd.DataFrame,
    res_col: str = "Residual",
    max_squares: int = 8,
    max_interactions: int = 8,
    candidate_feature_limit: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Builds a compact degree-2 polynomial basis:
      - rank single features by association with Residual, pick top 'candidate_feature_limit'
      - among them: create squares, rank by MI with Residual, keep top 'max_squares'
      - create pairwise interactions among shortlisted features, rank by MI, keep top 'max_interactions'
    Adds columns with readable names, updates POLY_BASES for constraints, returns df with new cols.
    """
    rng = np.random.default_rng(random_state)
    y = df[res_col].to_numpy()
    base_feats = numeric_cols_for_poly(df, exclude=EXCLUDE_COLS)
    if len(base_feats) == 0:
        return df
    # rank base features by |corr| with Residual to pick a candidate pool
    feat_scores = [(f, mi_or_corr(df[f].to_numpy(), y)) for f in base_feats]
    feat_scores.sort(key=lambda t: t[1], reverse=True)
    cand_feats = [
        f
        for f, i in feat_scores[
            : max(3, min(candidate_feature_limit, len(feat_scores)))
        ]
    ]

    # build and rank squares
    sq_candidates = []
    for f in cand_feats:
        name = safe_name_sq(f)
        x2 = df[f].to_numpy() ** 2
        score = mi_or_corr(x2, y)
        sq_candidates.append((name, f, x2, score))
    sq_candidates.sort(key=lambda t: t[3], reverse=True)
    chosen_squares = sq_candidates[:max_squares]

    # build and rank pairwise interactions
    ix_candidates = []
    for i in range(len(cand_feats)):
        for j in range(i + 1, len(cand_feats)):
            a, b = cand_feats[i], cand_feats[j]
            name = safe_name_ix(a, b)
            xij = df[a].to_numpy() * df[b].to_numpy()
            score = mi_or_corr(xij, y)
            ix_candidates.append((name, (a, b), xij, score))
    ix_candidates.sort(key=lambda t: t[3], reverse=True)
    chosen_ix = ix_candidates[:max_interactions]

    # add to df + register bases
    df_poly = df.copy()
    added = 0
    POLY_BASES.clear()
    for name, f, x2, i in chosen_squares:
        if name in df_poly.columns:
            continue
        df_poly[name] = x2
        POLY_BASES[name] = {f}
        added += 1
    for name, (a, b), xij, i in chosen_ix:
        if name in df_poly.columns:
            continue
        df_poly[name] = xij
        POLY_BASES[name] = {a, b}
        added += 1
    print(
        f"[L_poly] Added {added} polynomial features "
        f"({len(chosen_squares)} squares, {len(chosen_ix)} interactions)."
    )
    return df_poly


def is_poly_feature(feat: str) -> bool:
    return (feat in POLY_BASES) or ("^2" in feat) or ("*" in feat)


def poly_bases_of(feat: str) -> set:
    if feat in POLY_BASES:
        return set(POLY_BASES[feat])
    # fallback best-effort parse (if user provided their own poly columns)
    if "^2" in feat:
        return {feat.split("^2")[0]}
    if "*" in feat:
        a, b = feat.split("*", 1)
        return {a, b}
    return set()


def rule_ok_with(
    preds: Tuple[Predicate, ...], new_pred: Optional[Predicate] = None
) -> bool:
    """Enforce:
    - ≤ 1 polynomial term per rule
    - If a poly term is present, none of its base features may also appear in the rule
    """
    all_preds = preds + ((new_pred,) if new_pred is not None else tuple())
    poly_count = 0
    used_feats = set()
    poly_bases_used = set()

    for p in all_preds:
        used_feats.add(p.feature)
        if is_poly_feature(p.feature):
            poly_count += 1
            poly_bases_used |= poly_bases_of(p.feature)

    if poly_count > 1:
        return False

    # disallow mixing a poly feature with its base(s) in the same rule
    if poly_count == 1:
        # if any base feature is also used directly, reject
        if len(poly_bases_used & used_feats) > 0:
            return False

    return True


def is_conflict(existing: Tuple[Predicate, ...], cand: Predicate) -> bool:
    """
    Keep original L_conj guard: don't allow two predicates on the same feature.
    L_poly guard: rule_ok_with must be satisfied.
    """
    if cand.feature in {p.feature for p in existing}:
        return True
    # poly constraints
    if not rule_ok_with(existing, cand):
        return True
    return False


def level_extend(
    base_rules: List[Rule],
    uni_preds: List[Predicate],
    resid: np.ndarray,
    mean_global: float,
    var_global: float,
) -> List[Rule]:
    seen_masks = set()
    next_rules: List[Rule] = []
    for r in base_rules:
        for p in uni_preds:
            if is_conflict(r.preds, p):
                continue
            new_mask = r.mask & p.mask
            n = int(new_mask.sum())
            if n < MIN_SUPPORT:
                continue
            key = new_mask.tobytes()
            if key in seen_masks:
                continue
            seen_masks.add(key)
            size, mean_sub, q = compute_q_residual(
                new_mask, resid, mean_global, var_global
            )
            new_rule = Rule(
                preds=r.preds + (p,),
                mask=new_mask,
                size=size,
                mean_resid=mean_sub,
                q=q,
            )
            next_rules.append(new_rule)
    next_rules.sort(key=lambda r: r.q, reverse=True)
    return next_rules[:BEAM_WIDTH]
