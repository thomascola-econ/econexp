import numpy as np
import statsmodels.api as sm


def run_regression(X, y, cov_type="HC0", cov_kwds=None):
    """
    runs a regression of y on X and returns the results object.
    By default, it uses heteroskedasticity-robust standard errors (HC0),
    but you can specify other types of standard errors
    using the cov_type and cov_kwds parameters.
    X: A 2D array or DataFrame of independent variables (features).
    y: A 1D array or Series of the dependent variable (target).
    cov_type: A string specifying the type of covariance estimator to use.
                Common options include 'HC0', 'HC1', 'HC2', 'HC3
    cov_kwds: A dictionary of keyword arguments to pass to the covariance estimator.
    """

    # Add a column of ones to X for the intercept
    X = sm.add_constant(X)
    # Fit the regression model
    model = sm.OLS(y, X)
    results = model.fit(cov_type=cov_type, cov_kwds=cov_kwds)
    return results


def treatment_effect_regression(
    D: np.array,
    y: np.array,
    X: np.array = None,
    time: np.array = None,
    cov_type: str = "HC0",
    cov_kwds: dict = None,
    baseline=None,
    winsorization_quantiles=None,
):
    """
    Run a regression to estimate the treatment effect of D on y, controlling for X.
    X: control variables (2D array)
    y: dependent variable (1D array)
    D: treatment variable (1D array)
    time: optional time period variable (1D array). When provided, a separate treatment
          effect is estimated for each time period via treatment × time interactions.
          The first time period serves as the baseline; its treatment effects are the
          main D coefficients, while other periods' effects are captured by D × time
          interaction terms.
    baseline: value of D to use as reference category (defaults to the first level)
    Returns: regression results object
    """
    D = np.asarray(D)

    if winsorization_quantiles is not None:
        lower_q, upper_q = winsorization_quantiles
        y = np.clip(y, np.quantile(y, lower_q), np.quantile(y, upper_q))

    treatment_levels = np.unique(D)
    if len(treatment_levels) < 2:
        raise ValueError("D must take multiple values")

    if baseline is None:
        baseline = treatment_levels[0]
    elif baseline not in treatment_levels:
        raise ValueError(f"baseline value {baseline!r} not found in D")

    non_baseline = treatment_levels[treatment_levels != baseline]
    D_dummies = (D.reshape(-1, 1) == non_baseline).astype(float)

    if time is not None:
        time = np.asarray(time)
        time_levels = np.unique(time)
        if len(time_levels) < 2:
            raise ValueError("time must take multiple values")
        time_baseline = time_levels[0]
        non_baseline_times = time_levels[time_levels != time_baseline]
        T_dummies = (time.reshape(-1, 1) == non_baseline_times).astype(float)
        # D × time interactions: separate treatment effect per non-baseline time period
        DT_interactions = (D_dummies[:, :, None] * T_dummies[:, None, :]).reshape(
            len(D), -1
        )

        if X is None:
            X_full = np.hstack([D_dummies, DT_interactions, T_dummies])
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            X_dm = X - X.mean(axis=0)
            X_full = np.hstack(
                [
                    D_dummies,
                    DT_interactions,
                    T_dummies,
                    X_dm,
                    (D_dummies[:, :, None] * X_dm[:, None, :]).reshape(len(D), -1),
                ]
            )
    elif X is None:
        X_full = D_dummies
    else:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_dm = X - X.mean(axis=0)
        X_full = np.hstack(
            [
                D_dummies,
                X_dm,
                (D_dummies[:, :, None] * X_dm[:, None, :]).reshape(len(D), -1),
            ]
        )

    return run_regression(X_full, y, cov_type, cov_kwds)
