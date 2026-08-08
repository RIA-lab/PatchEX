"""Published reference numbers, used by scripts/train.py to self-check a run.

These are the values reported in the paper for the shipped configuration
(`A_esmhead`: causal mixture w_B + ESM-fed site head, lambda_anchor=1.0,
freeze='top'), as mean +/- sd over 3 seeds on the held-out test split.

A retrained model is expected to land within roughly 2 sd of these. Larger
deviations usually mean a different data split, a different ESM checkpoint, or
a changed hyper-parameter — not a broken install.
"""

REFERENCE = {
    "opt": {
        "name": "optimal temperature (Topt)",
        "unit": "C",
        "test_n": 857,
        "n_seeds": 3,
        "metrics": {
            "MAE":      (9.208, 0.044),
            "RMSE":     (12.859, 0.248),
            "R2":       (0.347, 0.025),
            "Pearson":  (0.612, 0.010),
            "Spearman": (0.407, 0.008),
        },
        "site_AUROC": (0.823, 0.013),
        "ensemble_MAE": 8.898,
        "config": "lambda_site=10, lambda_anchor=1.0, lambda_direct=0.1, freeze=top",
    },
    "ph": {
        "name": "optimal pH",
        "unit": "pH",
        "test_n": 1971,
        "n_seeds": 3,
        "metrics": {
            "MAE":      (0.615, 0.008),
            "RMSE":     (0.857, 0.008),
            "R2":       (0.449, 0.010),
            "Pearson":  (0.686, 0.004),
            "Spearman": (0.577, 0.006),
        },
        "site_AUROC": (0.855, 0.003),
        "ensemble_MAE": 0.585,
        "config": "lambda_site=1, lambda_anchor=1.0, lambda_direct=0.1, freeze=top",
    },
}


def compare(arm: str, achieved: dict, tol_sd: float = 2.0):
    """Compare achieved metrics against the published reference.

    Returns (rows, ok) where rows is a list of dicts ready to print and `ok` is
    True when every comparable metric is within `tol_sd` standard deviations.
    A single seed is naturally noisier than the 3-seed mean, so `tol_sd` is
    deliberately loose.
    """
    ref = REFERENCE[arm]
    rows, ok = [], True
    for k, (mu, sd) in ref["metrics"].items():
        got = achieved.get(k)
        if got is None:
            continue
        band = max(sd, 1e-9) * tol_sd
        within = abs(got - mu) <= band
        ok &= within
        rows.append(dict(metric=k, achieved=round(float(got), 4),
                         published=f"{mu} ± {sd}",
                         delta=round(float(got) - mu, 4),
                         status="OK" if within else f"OUTSIDE ±{tol_sd}sd"))
    return rows, ok
