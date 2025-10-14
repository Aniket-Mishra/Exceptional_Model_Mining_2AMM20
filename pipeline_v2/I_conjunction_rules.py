from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from configurations import *
from utils import *

"""
L_conj on residuals
takes a df with Residual (we chose squared err)
1. it makes 1d predicates per feature,
    1. for numeric, its <= val and > val over quantile thresholds.
    2. for cat, its == val
2. beam search conjugates up to k, default rn is 3.
3. Scores each rule with q_residual(S) = abs(mean_S(residual) - mean_gloabl(residual) / sqrt(variance_global(residual) / abs(S))
4. returns a df with top K (predefined in config) rules with descriptions, mean, n q scores
"""


def features_in_rule(preds: Tuple[Predicate, ...]) -> set:  # beam search
    return {p.feature for p in preds}


def is_conflict(existing: Tuple[Predicate, ...], cand: Predicate) -> bool:
    # pred on same features are skipped
    return cand.feature in features_in_rule(existing)


def level_extend(  # Build next-level rules by adding one non-conflicting predicate
    base_rules: List[Rule],
    uni_preds: List[Predicate],
    resid: np.ndarray,
    mean_global: float,
    var_global: float,
    df: pd.DataFrame,
) -> List[Rule]:
    seen_masks = set()
    next_rules: List[Rule] = []
    # precompute feature types (numeric vs categorical) once
    num_cols, cat_cols = infer_feature_types(df)

    for r in base_rules:
        used_feats = features_in_rule(r.preds)
        base_mask = r.mask

        # numeric candidates via LOCAL discretization per branch
        if USE_LOCAL_DISCRETIZATION:
            for col in num_cols:
                if col in used_feats:  # keep one predicate per feature
                    continue
                local_preds = build_numeric_predicates_local(
                    df=df,
                    col=col,
                    base_mask=base_mask,
                    n_quantiles=LOCAL_QUANTILES,
                    k_keep=LOCAL_THRESHOLDS_PER_FEAT,
                    resid=resid,
                    mean_global=mean_global,
                    var_global=var_global,
                    step=THRESH_RESOLUTION,
                )
                for p in local_preds:
                    new_mask = p.mask  # already intersected with base_mask
                    if int(new_mask.sum()) < MIN_SUPPORT:
                        continue
                    key = new_mask.tobytes()
                    if key in seen_masks:
                        continue
                    seen_masks.add(key)
                    size, mean_sub, q = compute_q_residual(
                        new_mask, resid, mean_global, var_global
                    )
                    next_rules.append(
                        Rule(
                            preds=r.preds + (p,),
                            mask=new_mask,
                            size=size,
                            mean_resid=mean_sub,
                            q=q,
                        )
                    )

        # categorical candidates stay global, so only =
        for p in uni_preds:
            if p.feature in used_feats:
                continue
            if p.feature in num_cols:  # skip numeric here if local mode used
                if USE_LOCAL_DISCRETIZATION:
                    continue
            # intersect
            new_mask = base_mask & p.mask
            if int(new_mask.sum()) < MIN_SUPPORT:
                continue
            key = new_mask.tobytes()
            if key in seen_masks:
                continue
            seen_masks.add(key)
            size, mean_sub, q = compute_q_residual(
                new_mask, resid, mean_global, var_global
            )
            next_rules.append(
                Rule(
                    preds=r.preds + (p,),
                    mask=new_mask,
                    size=size,
                    mean_resid=mean_sub,
                    q=q,
                )
            )

    next_rules.sort(key=lambda r: r.q, reverse=True)
    return next_rules[:BEAM_WIDTH]


def mine_conj_rules(
    df: pd.DataFrame,
    res_col: str = RES_COL,
    max_rule_len: int = MAX_RULE_LEN,
    top_k: int = TOP_K_PER_LANGUAGE,
) -> pd.DataFrame:
    assert res_col in df.columns, f"Residual column '{res_col}' not found."
    resid = df[res_col].values
    mean_global = float(np.nanmean(resid))
    var_global = float(np.nanvar(resid))
    # generate unary predicates
    uni_preds = build_all_predicates(df)
    # Level 1: evaluate all unary predicates
    level_rules: List[Rule] = []
    for p in uni_preds:
        size, mean_sub, q = compute_q_residual(
            p.mask, resid, mean_global, var_global
        )
        if size >= MIN_SUPPORT:
            level_rules.append(
                Rule(
                    preds=(p,),
                    mask=p.mask,
                    size=size,
                    mean_resid=mean_sub,
                    q=q,
                )
            )
    level_rules.sort(key=lambda r: r.q, reverse=True)
    level_rules = level_rules[:BEAM_WIDTH]
    all_rules = list(level_rules)

    # Levels 2: max_rule_len
    current = level_rules
    for L in range(2, max_rule_len + 1):
        if not current:
            break
        current = level_extend(
            current, uni_preds, resid, mean_global, var_global, df
        )
        all_rules.extend(current)

    # Deduplicate by mask and sort
    dedup: Dict[bytes, Rule] = {}
    for r in all_rules:
        key = r.mask.tobytes()
        # keep the best-scoring version if duplicates
        if key not in dedup or r.q > dedup[key].q:
            dedup[key] = r

    ranked = sorted(dedup.values(), key=lambda r: r.q, reverse=True)[:top_k]
    out = pd.DataFrame(
        {
            "rule": [r.describe() for r in ranked],
            "length": [len(r.preds) for r in ranked],
            "size": [r.size for r in ranked],
            "mean_residual": [r.mean_resid for r in ranked],
            "delta_from_global": [r.mean_resid - mean_global for r in ranked],
            "q_residual": [r.q for r in ranked],
        }
    )
    out.attrs["global_mean_residual"] = mean_global
    out.attrs["global_var_residual"] = var_global
    return out
