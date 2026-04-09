import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from econexp.metrics import run_regression


@dataclass
class MDEResult:
    """
    Structured result returned by power analysis functions.
    mde: absolute minimum detectable effect (None when computing required_n)
    n_per_arm: per-arm sample size used or required
    power: target or achieved power (None when not applicable)
    alpha: significance level
    variance: effective variance used in calculation (post-adjustments)
    design_effect: DEFF inflation factor (1.0 unless clustered)
    notes: human-readable description of adjustments applied
    """

    mde: float | None
    n_per_arm: int | None
    power: float | None
    alpha: float
    variance: float
    design_effect: float
    notes: str


def _critical_values(
    alpha: float,
    power: float,
    two_sided: bool = True,
) -> tuple[float, float]:
    """
    Return (z_alpha, z_beta) for use in power calculations.
    z_alpha: critical value for the significance level alpha.
    z_beta: critical value for the target power (1 - beta).
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    return z_alpha, z_beta


def mde(
    variance: float,
    n: int,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> MDEResult:
    """
    Compute the minimum detectable effect given outcome variance and per-arm
    sample size. Uses the standard two-sample power formula.
    variance: outcome variance sigma^2 (scalar, must be positive).
    n: observations per arm (must be >= 2).
    alpha: significance level (default 0.05).
    power: target statistical power (default 0.80).
    two_sided: if True, use two-sided critical value (default True).
    """
    if variance <= 0:
        raise ValueError(f"variance must be positive, got {variance}")
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not (0 < power < 1):
        raise ValueError(f"power must be in (0, 1), got {power}")

    z_alpha, z_beta = _critical_values(alpha, power, two_sided)
    mde_val = (z_alpha + z_beta) * math.sqrt(2 * variance / n)

    return MDEResult(
        mde=mde_val,
        n_per_arm=n,
        power=power,
        alpha=alpha,
        variance=variance,
        design_effect=1.0,
        notes="core MDE formula: (z_alpha + z_beta) * sqrt(2 * variance / n)",
    )


def mde_binary(
    p: float,
    n: int,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> MDEResult:
    """
    Compute MDE for binary (proportion) outcomes. Variance is determined
    by the baseline proportion using the Bernoulli formula sigma^2 = p*(1-p).
    p: baseline proportion, must be in (0, 1).
    n: observations per arm.
    alpha: significance level (default 0.05).
    power: target power (default 0.80).
    two_sided: if True, use two-sided critical value (default True).
    """
    if not (0 < p < 1):
        raise ValueError(f"p must be in (0, 1), got {p}")

    variance = p * (1 - p)
    result = mde(variance, n, alpha, power, two_sided)
    result.notes = f"binary outcome; variance = p*(1-p) = {variance:.4f} (p={p})"
    return result


def required_n(
    mde_val: float,
    variance: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> MDEResult:
    """
    Compute the per-arm sample size required to detect a given effect size
    with specified power and significance. Inverse of the mde() function.
    mde_val: minimum detectable effect to target (must be positive).
    variance: outcome variance sigma^2.
    alpha: significance level (default 0.05).
    power: target power (default 0.80).
    two_sided: if True, use two-sided critical value (default True).
    """
    if mde_val <= 0:
        raise ValueError(f"mde_val must be positive, got {mde_val}")
    if variance <= 0:
        raise ValueError(f"variance must be positive, got {variance}")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not (0 < power < 1):
        raise ValueError(f"power must be in (0, 1), got {power}")

    z_alpha, z_beta = _critical_values(alpha, power, two_sided)
    n_exact = 2 * variance * ((z_alpha + z_beta) / mde_val) ** 2
    n_per_arm = math.ceil(n_exact)

    return MDEResult(
        mde=mde_val,
        n_per_arm=n_per_arm,
        power=power,
        alpha=alpha,
        variance=variance,
        design_effect=1.0,
        notes=f"required n per arm to detect mde={mde_val} (ceiling of {n_exact:.2f})",
    )


def achieved_power(
    delta: float,
    variance: float,
    n: int,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> MDEResult:
    """
    Compute the statistical power achieved for a given true effect size,
    residual variance, and per-arm sample size.
    delta: true effect size (absolute). Sign is ignored; power is symmetric.
    variance: outcome variance sigma^2.
    n: observations per arm.
    alpha: significance level (default 0.05).
    two_sided: if True, use two-sided critical value (default True).
    """
    if variance <= 0:
        raise ValueError(f"variance must be positive, got {variance}")
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    z_alpha = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    se = math.sqrt(2 * variance / n)

    if delta == 0:
        pwr = alpha / 2 if two_sided else alpha
        notes = "delta=0; power equals alpha/2 (two-sided) or alpha (one-sided)"
    else:
        pwr = float(stats.norm.cdf(abs(delta) / se - z_alpha))
        notes = f"achieved power for delta={delta}, n_per_arm={n}"

    return MDEResult(
        mde=None,
        n_per_arm=n,
        power=pwr,
        alpha=alpha,
        variance=variance,
        design_effect=1.0,
        notes=notes,
    )


def icc_from_data(
    y: np.array,
    clusters: np.array,
) -> float:
    """
    Estimate the intra-cluster correlation coefficient (ICC) from data using
    the one-way ANOVA method-of-moments estimator. Result is clipped to [0, 1]
    since negative estimates can arise in small samples.
    y: outcome variable (1D array).
    clusters: cluster membership labels, same length as y.
    """
    y = np.asarray(y, dtype=float)
    clusters = np.asarray(clusters)

    unique, counts = np.unique(clusters, return_counts=True)
    if len(unique) < 2:
        raise ValueError("need at least 2 clusters to estimate ICC")

    grand_mean = y.mean()
    n = len(y)
    g = len(unique)
    m_bar = n / g

    ss_b = sum(
        counts[i] * (y[clusters == c].mean() - grand_mean) ** 2
        for i, c in enumerate(unique)
    )
    ss_w = sum(
        np.sum((y[clusters == c] - y[clusters == c].mean()) ** 2) for c in unique
    )

    ms_b = ss_b / (g - 1)
    ms_w = ss_w / (n - g)

    if ms_b + (m_bar - 1) * ms_w == 0:
        return 0.0

    icc_val = (ms_b - ms_w) / (ms_b + (m_bar - 1) * ms_w)
    return float(np.clip(icc_val, 0.0, 1.0))


def mde_clustered(
    variance: float,
    n_clusters: int,
    cluster_size: int,
    icc: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> MDEResult:
    """
    Compute MDE for cluster-randomized trials using the design effect (DEFF).
    The DEFF inflates variance to account for within-cluster correlation.
    variance: individual-level outcome variance sigma^2.
    n_clusters: number of clusters per arm (not total).
    cluster_size: average number of observations per cluster.
    icc: intra-cluster correlation coefficient, in [0, 1].
    alpha: significance level (default 0.05).
    power: target power (default 0.80).
    two_sided: if True, use two-sided critical value (default True).
    """
    if not (0 <= icc <= 1):
        raise ValueError(f"icc must be in [0, 1], got {icc}")
    if cluster_size < 1:
        raise ValueError(f"cluster_size must be >= 1, got {cluster_size}")
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")

    deff = 1 + (cluster_size - 1) * icc
    eff_var = variance * deff
    n_obs = n_clusters * cluster_size

    result = mde(eff_var, n_obs, alpha, power, two_sided)
    result.variance = variance
    result.design_effect = deff
    result.notes = (
        f"clustered design; DEFF={deff:.4f} "
        f"(icc={icc}, cluster_size={cluster_size}); "
        f"n_clusters_per_arm={n_clusters}"
    )
    return result


def mde_did(
    variance: float,
    n: int,
    rho: float = None,
    y_pre: np.array = None,
    y_post: np.array = None,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> MDEResult:
    """
    Compute MDE for difference-in-differences or panel designs. Exploits
    within-unit pre/post correlation to reduce effective variance:
    sigma_diff^2 = 2 * sigma^2 * (1 - rho).
    variance: cross-sectional outcome variance sigma^2.
    n: observations per arm.
    rho: within-unit pre/post correlation. If None, estimated from y_pre and y_post.
    y_pre: pre-period outcome array (used to estimate rho if rho is None).
    y_post: post-period outcome array (used to estimate rho if rho is None).
    alpha: significance level (default 0.05).
    power: target power (default 0.80).
    two_sided: if True, use two-sided critical value (default True).
    """
    if rho is None and (y_pre is None or y_post is None):
        raise ValueError(
            "must provide either rho or both y_pre and y_post to estimate rho"
        )

    if rho is None:
        y_pre = np.asarray(y_pre, dtype=float)
        y_post = np.asarray(y_post, dtype=float)
        if len(y_pre) != len(y_post):
            raise ValueError(
                f"y_pre and y_post must have the same length, "
                f"got {len(y_pre)} and {len(y_post)}"
            )
        rho = float(np.corrcoef(y_pre, y_post)[0, 1])
        rho_source = f"estimated from data (rho={rho:.4f})"
    else:
        if not (-1 <= rho <= 1):
            raise ValueError(f"rho must be in [-1, 1], got {rho}")
        rho_source = f"provided (rho={rho})"

    variance_diff = 2 * variance * (1 - rho)
    result = mde(variance_diff, n, alpha, power, two_sided)
    result.variance = variance
    result.notes = f"DiD/panel design; sigma_diff^2 = 2*sigma^2*(1-rho); {rho_source}"
    return result


def mde_from_data(
    y: np.array,
    n: int,
    X: np.array = None,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
    ddof: int = 1,
) -> MDEResult:
    """
    Estimate variance from data (with optional covariate adjustment) and
    compute MDE. If X is provided, fits OLS of y on X and uses the residual
    mean squared error, equivalent to ANCOVA variance reduction sigma^2*(1-R^2).
    y: outcome variable (1D array). Used to estimate variance.
    n: hypothetical per-arm sample size for the experiment.
    X: covariate matrix (2D or 1D). If provided, OLS residual variance is used.
    alpha: significance level (default 0.05).
    power: target power (default 0.80).
    two_sided: if True, use two-sided critical value (default True).
    ddof: degrees of freedom correction when X is None (default 1).
    """
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        raise ValueError(
            f"need at least 3 observations to estimate variance, got {len(y)}"
        )

    if X is None:
        variance = float(np.var(y, ddof=ddof))
        notes = f"variance estimated from y (ddof={ddof})"
    else:
        X = np.asarray(X, dtype=float)
        results = run_regression(X, y)
        variance = float(results.mse_resid)
        r2 = float(results.rsquared)
        notes = f"variance from OLS residuals (mse_resid); R\u00b2={r2:.4f}"

    result = mde(variance, n, alpha, power, two_sided)
    result.notes = notes
    return result


def mde_multi_arm(
    variance: float,
    n_total: int,
    n_arms: int,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
    correct_alpha: bool = False,
) -> MDEResult:
    """
    Compute per-arm MDE for experiments with multiple treatment arms and one
    control arm. Assumes equal allocation across (n_arms + 1) groups.
    Optionally applies Bonferroni correction for multiple comparisons.
    variance: outcome variance sigma^2.
    n_total: total sample size across all arms.
    n_arms: number of treatment arms (not counting control).
    alpha: significance level per comparison (default 0.05).
    power: target power (default 0.80).
    two_sided: if True, use two-sided critical value (default True).
    correct_alpha: if True, apply Bonferroni correction alpha / n_arms (default False).
    """
    if n_arms < 1:
        raise ValueError(f"n_arms must be >= 1, got {n_arms}")
    if n_total < 2 * (n_arms + 1):
        raise ValueError(
            f"n_total={n_total} is insufficient for {n_arms} treatment arm(s) "
            f"and 1 control arm (need at least {2 * (n_arms + 1)})"
        )

    n_per_arm = n_total // (n_arms + 1)
    alpha_eff = alpha / n_arms if correct_alpha else alpha
    bonferroni_note = (
        f" with Bonferroni correction (alpha/{n_arms})" if correct_alpha else ""
    )

    result = mde(variance, n_per_arm, alpha_eff, power, two_sided)
    result.alpha = alpha
    result.notes = (
        f"multi-arm design; {n_arms} treatment arm(s) + 1 control; "
        f"n_per_arm={n_per_arm}{bonferroni_note}"
    )
    return result


def simulate_power_within(
    effect: float,
    n: int,
    y_pre: np.array,
    y_post: np.array = None,
    alpha: float = 0.05,
    n_sims: int = 1000,
    two_sided: bool = True,
    seed: int = None,
) -> MDEResult:
    """
    Estimate power for a within-unit (paired) experimental design via simulation.
    Useful when outcomes are non-normal or when you want to respect the actual
    empirical distribution rather than rely on the normal approximation.

    If y_post is provided alongside y_pre, bootstrap pairs (pre_i, post_i) from
    the pilot data and add the hypothetical effect to the post period. This
    captures the real within-unit correlation structure from the data.

    If only y_pre is provided, model each unit's within-unit difference as
    d_i = effect + noise_post_i - noise_pre_i, where noise is drawn independently
    from the centered empirical distribution of y_pre. This gives
    Var(d_i) = 2 * Var(y_pre), equivalent to assuming zero pre-post correlation.

    In both cases power is estimated as the fraction of paired t-tests
    (one-sample t-test on within-unit differences) that reject H0: mean(d) = 0.

    effect: hypothetical treatment effect size to test power for.
    n: number of paired observations per simulation draw.
    y_pre: pre-period outcome array. Used as the empirical distribution to
           bootstrap from.
    y_post: post-period outcome array paired with y_pre (optional). If provided,
            pairs are bootstrapped together to preserve the correlation structure.
    alpha: significance level (default 0.05).
    n_sims: number of simulation iterations (default 1000).
    two_sided: if True, use two-sided paired t-test (default True).
    seed: random seed for reproducibility (default None).
    """
    y_pre = np.asarray(y_pre, dtype=float)
    if len(y_pre) < 2:
        raise ValueError(f"y_pre must have at least 2 observations, got {len(y_pre)}")
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    if n_sims < 1:
        raise ValueError(f"n_sims must be at least 1, got {n_sims}")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    paired = y_post is not None
    if paired:
        y_post = np.asarray(y_post, dtype=float)
        if len(y_post) != len(y_pre):
            raise ValueError(
                f"y_pre and y_post must have the same length, "
                f"got {len(y_pre)} and {len(y_post)}"
            )

    rng = np.random.default_rng(seed)
    t_crit = (
        stats.t.ppf(1 - alpha / 2, df=n - 1)
        if two_sided
        else stats.t.ppf(1 - alpha, df=n - 1)
    )

    rejections = 0
    if paired:
        # Bootstrap (pre, post) pairs together to preserve correlation.
        pairs = np.column_stack([y_pre, y_post])
        for _ in range(n_sims):
            idx = rng.integers(0, len(pairs), size=n)
            pre_s = pairs[idx, 0]
            post_s = pairs[idx, 1] + effect
            diff = post_s - pre_s
            t_stat = diff.mean() / (diff.std(ddof=1) / math.sqrt(n))
            rejections += abs(t_stat) > t_crit if two_sided else t_stat > t_crit
        design_note = (
            f"bootstrap pairs from {len(y_pre)} pilot obs "
            f"(rho_pilot={float(np.corrcoef(y_pre, y_post)[0, 1]):.3f})"
        )
    else:
        # Draw independent pre and post noise from centered y_pre.
        # Var(d_i) = 2 * Var(y_pre) — equivalent to rho = 0.
        y_centered = y_pre - y_pre.mean()
        for _ in range(n_sims):
            noise_pre = rng.choice(y_centered, size=n, replace=True)
            noise_post = rng.choice(y_centered, size=n, replace=True)
            diff = effect + noise_post - noise_pre
            t_stat = diff.mean() / (diff.std(ddof=1) / math.sqrt(n))
            rejections += abs(t_stat) > t_crit if two_sided else t_stat > t_crit
        design_note = (
            f"independent pre/post noise from {len(y_pre)} obs of y_pre "
            f"(rho assumed 0)"
        )

    simulated_power = rejections / n_sims
    variance_est = float(np.var(y_pre, ddof=1))

    return MDEResult(
        mde=None,
        n_per_arm=n,
        power=simulated_power,
        alpha=alpha,
        variance=variance_est,
        design_effect=1.0,
        notes=(
            f"simulated within-design power; effect={effect}; "
            f"n={n}; n_sims={n_sims}; {design_note}"
        ),
    )
