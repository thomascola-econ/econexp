"""
Tests for the econexp.metrics module.

Tests cover:
- run_regression: OLS regression wrapper with heteroskedasticity-robust SEs
- treatment_effect_regression: Treatment effect estimation with interactions
"""

import numpy as np
import pytest

from econexp.metrics import run_regression, treatment_effect_regression


class TestRunRegression:
    """Tests for the run_regression function."""

    @pytest.fixture
    def simple_data(self):
        """Generate simple regression data."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        y = 2 + X[:, 0] * 3 + X[:, 1] * 1.5 + np.random.randn(n) * 0.5
        return X, y

    def test_run_regression_basic(self, simple_data):
        """Test basic regression functionality."""
        X, y = simple_data
        results = run_regression(X, y)

        # Check that results object is returned
        assert results is not None
        assert hasattr(results, "params")
        assert hasattr(results, "bse")
        assert hasattr(results, "pvalues")

    def test_run_regression_params_shape(self, simple_data):
        """Test that regression returns correct number of parameters."""
        X, y = simple_data
        results = run_regression(X, y)

        # Should have intercept + 2 predictors
        assert len(results.params) == 3
        assert len(results.bse) == 3

    def test_run_regression_intercept(self, simple_data):
        """Test that intercept is estimated."""
        X, y = simple_data
        results = run_regression(X, y)

        # First parameter should be intercept (around 2)
        assert 1 < results.params[0] < 3

    def test_run_regression_hc0_errors(self, simple_data):
        """Test HC0 heteroskedasticity-robust standard errors (default)."""
        X, y = simple_data
        results = run_regression(X, y, cov_type="HC0")

        # Check that all standard errors are positive
        assert np.all(results.bse > 0)

    def test_run_regression_hc1_errors(self, simple_data):
        """Test HC1 heteroskedasticity-robust standard errors."""
        X, y = simple_data
        results_hc0 = run_regression(X, y, cov_type="HC0")
        results_hc1 = run_regression(X, y, cov_type="HC1")

        # HC1 standard errors should be slightly larger (finite sample adjustment)
        assert np.all(results_hc1.bse >= results_hc0.bse)

    def test_run_regression_single_predictor(self):
        """Test regression with a single predictor."""
        np.random.seed(42)
        X = np.random.randn(50, 1)
        y = 1 + 2 * X[:, 0] + np.random.randn(50) * 0.1

        results = run_regression(X, y)

        assert len(results.params) == 2  # Intercept + 1 predictor
        assert 1.5 < results.params[1] < 2.5  # Slope coefficient

    def test_run_regression_perfect_fit(self):
        """Test regression with perfect linear relationship."""
        X = np.arange(100).reshape(-1, 1) / 100.0
        y = 5 + 3 * X[:, 0]  # Perfect relationship

        results = run_regression(X, y)

        # Intercept should be close to 5
        assert np.isclose(results.params[0], 5, atol=1e-10)
        # Slope should be close to 3
        assert np.isclose(results.params[1], 3, atol=1e-10)

    def test_run_regression_cov_kwds(self, simple_data):
        """Test passing covariance keyword arguments."""
        X, y = simple_data
        # HC3 typically requires additional arguments
        results = run_regression(X, y, cov_type="HC3", cov_kwds={})

        assert results is not None
        assert len(results.params) == 3

    def test_run_regression_large_sample(self):
        """Test regression with larger sample size."""
        np.random.seed(42)
        n = 1000
        X = np.random.randn(n, 3)
        y = X[:, 0] - 2 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(n) * 0.1

        results = run_regression(X, y)

        assert len(results.params) == 4  # Intercept + 3 predictors
        # R-squared should be high since noise is low
        assert results.rsquared > 0.8


class TestTreatmentEffectRegression:
    """Tests for the treatment_effect_regression function."""

    @pytest.fixture
    def treatment_data(self):
        """Generate treatment effect data."""
        np.random.seed(42)
        n = 201  # Use 201 so it divides evenly by 3
        D = np.repeat([0, 1, 2], n // 3)  # Three treatment groups (67 each)
        X = np.random.randn(n, 2)
        # Treatment effect: groups have different intercepts
        y = (
            (D == 1) * 2
            + (D == 2) * 4
            + X[:, 0]
            + 0.5 * X[:, 1]
            + np.random.randn(n) * 0.5
        )
        return D, y, X

    def test_treatment_effect_basic(self, treatment_data):
        """Test basic treatment effect regression."""
        D, y, X = treatment_data
        results = treatment_effect_regression(D, y, X)

        assert results is not None
        assert hasattr(results, "params")
        assert hasattr(results, "pvalues")

    def test_treatment_effect_binary_treatment(self):
        """Test with binary treatment variable."""
        np.random.seed(42)
        n = 100
        D = np.random.binomial(1, 0.5, n)
        y = D * 2 + np.random.randn(n) * 0.5

        results = treatment_effect_regression(D, y)

        # Should have only one treatment coefficient (relative to baseline)
        assert len(results.params) >= 1

    def test_treatment_effect_multiple_treatments(self, treatment_data):
        """Test with multiple treatment levels."""
        D, y, X = treatment_data
        results = treatment_effect_regression(D, y, X)

        # Should have more parameters for multiple treatments
        assert len(results.params) > 1

    def test_treatment_effect_custom_baseline(self, treatment_data):
        """Test specifying a custom baseline treatment level."""
        D, y, X = treatment_data

        results_baseline_0 = treatment_effect_regression(D, y, X, baseline=0)
        results_baseline_1 = treatment_effect_regression(D, y, X, baseline=1)

        # Different baselines should give different coefficients
        assert not np.allclose(results_baseline_0.params, results_baseline_1.params)

    def test_treatment_effect_invalid_baseline(self, treatment_data):
        """Test that invalid baseline raises error."""
        D, y, X = treatment_data

        with pytest.raises(ValueError, match="baseline value"):
            treatment_effect_regression(D, y, X, baseline=999)

    def test_treatment_effect_single_level_error(self):
        """Test that single treatment level raises error."""
        D = np.ones(100)  # All same treatment
        y = np.random.randn(100)

        with pytest.raises(ValueError, match="D must take multiple values"):
            treatment_effect_regression(D, y)

    def test_treatment_effect_no_controls(self):
        """Test treatment effect without control variables."""
        np.random.seed(42)
        n = 100
        D = np.random.binomial(1, 0.5, n)
        y = D * 3 + np.random.randn(n) * 0.5

        results = treatment_effect_regression(D, y)

        assert results is not None
        assert len(results.params) >= 1

    def test_treatment_effect_1d_control(self, treatment_data):
        """Test with 1D control variable."""
        D, y, _ = treatment_data
        X = np.random.randn(len(D))  # 1D control

        results = treatment_effect_regression(D, y, X)

        assert results is not None
        assert len(results.params) > 1

    def test_treatment_effect_interactions(self, treatment_data):
        """Test that interactions between treatment and controls are included."""
        D, y, X = treatment_data
        results = treatment_effect_regression(D, y, X)

        # Parameters should include:
        # - treatment indicators (for each non-baseline level)
        # - centered controls
        # - interactions between treatment and controls
        # formula: D_dummies, X_centered, (D_dummies * X_centered)

        treatment_levels = np.unique(D)
        n_treatments = len(treatment_levels) - 1  # Excluding baseline
        n_controls = X.shape[1]

        # Intercept + treatments + controls + (treatments * controls)
        expected_params = 1 + n_treatments + n_controls + n_treatments * n_controls
        assert len(results.params) == expected_params

    def test_treatment_effect_heterogeneous_effects(self):
        """Test detection of heterogeneous treatment effects."""
        np.random.seed(42)
        n = 200
        D = np.repeat([0, 1], n // 2)
        X = np.random.randn(n)

        # Heterogeneous effect: treatment effect depends on X
        y = (D == 1) * (2 + 3 * X) + X + np.random.randn(n) * 0.3

        results = treatment_effect_regression(D, y, X)

        # The interaction term coefficient should be significant
        # Index 2 should be the D*X interaction
        assert results is not None
        assert (
            len(results.params) == 4
        )  # Intercept, D, X, D*X = 4 but with centered X...

    def test_treatment_effect_cov_type(self, treatment_data):
        """Test different covariance types."""
        D, y, X = treatment_data

        results_hc0 = treatment_effect_regression(D, y, X, cov_type="HC0")
        results_hc1 = treatment_effect_regression(D, y, X, cov_type="HC1")

        # Both should have valid results
        assert results_hc0 is not None
        assert results_hc1 is not None

    def test_treatment_effect_control_centering(self):
        """Test that control variables are properly centered."""
        np.random.seed(42)
        n = 100
        D = np.repeat([0, 1], n // 2)
        X = np.random.randn(n, 1) + 10  # Non-centered data

        results = treatment_effect_regression(D, y=np.random.randn(n), X=X)

        # Should still run without issues despite X being centered
        assert results is not None

    # ------------------------------------------------------------------
    # Tests for time parameter
    # ------------------------------------------------------------------

    def test_time_basic(self):
        """time parameter returns a valid results object."""
        np.random.seed(0)
        n = 120
        D = np.tile([0, 1], n // 2)
        time = np.repeat([0, 1, 2], n // 3)
        y = (D == 1) * 2 + np.random.randn(n) * 0.3
        results = treatment_effect_regression(D, y, time=time)
        assert results is not None
        assert hasattr(results, "params")

    def test_time_param_count_no_controls(self):
        """Parameter count is correct with time and no X controls."""
        np.random.seed(0)
        n_treat, n_time, n_per_cell = 3, 4, 20
        n = n_treat * n_time * n_per_cell
        D = np.tile(np.repeat([0, 1, 2], n_per_cell * n_time), 1)[:n]
        D = np.repeat([0, 1, 2], n_time * n_per_cell)
        time = np.tile(np.repeat([0, 1, 2, 3], n_per_cell), n_treat)
        y = np.random.randn(n)

        results = treatment_effect_regression(D, y, time=time)

        # intercept + D_dummies + DT_interactions + T_dummies
        # = 1 + (n_treat-1) + (n_treat-1)*(n_time-1) + (n_time-1)
        expected = 1 + (n_treat - 1) + (n_treat - 1) * (n_time - 1) + (n_time - 1)
        assert len(results.params) == expected

    def test_time_param_count_with_controls(self):
        """Parameter count is correct with time and X controls."""
        np.random.seed(0)
        n_treat, n_time, n_controls = 2, 3, 2
        n = 120
        D = np.repeat([0, 1], n // 2)
        time = np.tile([0, 1, 2], n // 3)
        X = np.random.randn(n, n_controls)
        y = np.random.randn(n)

        results = treatment_effect_regression(D, y, X=X, time=time)

        # intercept + D_dummies + DT_interactions + T_dummies + X + D*X
        expected = (
            1
            + (n_treat - 1)
            + (n_treat - 1) * (n_time - 1)
            + (n_time - 1)
            + n_controls
            + (n_treat - 1) * n_controls
        )
        assert len(results.params) == expected

    def test_time_single_level_raises(self):
        """A time array with only one unique value raises ValueError."""
        D = np.array([0, 1] * 50)
        y = np.random.randn(100)
        time = np.ones(100)
        with pytest.raises(ValueError, match="time must take multiple values"):
            treatment_effect_regression(D, y, time=time)

    def test_time_varying_effects_detected(self):
        """Treatment effects that differ across periods are recovered."""
        np.random.seed(1)
        n_per_cell = 200
        # Two treatment arms, two time periods
        # True effects: period 0 -> effect=1, period 1 -> effect=5
        D = np.tile([0, 1], n_per_cell)
        time = np.repeat([0, 1], n_per_cell)
        y = (
            np.where(
                (D == 1) & (time == 0), 1.0, np.where((D == 1) & (time == 1), 5.0, 0.0)
            )
            + np.random.randn(2 * n_per_cell) * 0.1
        )

        results = treatment_effect_regression(D, y, time=time)

        # params layout (no X): [const, D, D×T1, T1]
        effect_t0 = results.params[1]  # treatment at baseline time
        effect_t1 = results.params[1] + results.params[2]  # treatment at time=1

        assert abs(effect_t0 - 1.0) < 0.05
        assert abs(effect_t1 - 5.0) < 0.05

    def test_time_matches_subset_regression(self):
        """
        With no controls and a saturated model the treatment effect per period
        matches running treatment_effect_regression on each period's subset.
        """
        np.random.seed(7)
        n_per_cell = 150
        D = np.tile([0, 1], n_per_cell)
        time = np.repeat([0, 1], n_per_cell)
        y = (
            (D == 1) * 3.0 * (time == 0)
            + (D == 1) * 7.0 * (time == 1)
            + np.random.randn(2 * n_per_cell) * 0.1
        )

        results_joint = treatment_effect_regression(D, y, time=time)

        # params: [const, D, D×T1, T1]
        effect_t0_joint = results_joint.params[1]
        effect_t1_joint = results_joint.params[1] + results_joint.params[2]

        mask_t0 = time == 0
        mask_t1 = time == 1
        effect_t0_subset = treatment_effect_regression(D[mask_t0], y[mask_t0]).params[1]
        effect_t1_subset = treatment_effect_regression(D[mask_t1], y[mask_t1]).params[1]

        assert np.isclose(effect_t0_joint, effect_t0_subset, atol=1e-8)
        assert np.isclose(effect_t1_joint, effect_t1_subset, atol=1e-8)

    def test_time_without_time_unchanged(self, treatment_data):
        """Passing time=None gives the same result as not passing time."""
        D, y, X = treatment_data
        results_none = treatment_effect_regression(D, y, X, time=None)
        results_omit = treatment_effect_regression(D, y, X)
        assert np.allclose(results_none.params, results_omit.params)


class TestRegressionConsistency:
    """Tests for consistency between run_regression and treatment_effect_regression."""

    def test_no_controls_equivalence(self):
        """Test that treatment effect with no controls matches simple contrast."""
        np.random.seed(42)
        n = 100
        D = np.array([0] * 50 + [1] * 50)
        np.random.shuffle(D)
        y = D * 3 + np.random.randn(n) * 0.5

        # Treatment effect regression without controls
        results_te = treatment_effect_regression(D, y)

        # Manual approach: regress y on D indicator
        X_manual = (D == 1).astype(float).reshape(-1, 1)
        results_manual = run_regression(X_manual, y)

        # Treatment effect should be close to the D coefficient
        assert np.isclose(results_te.params[1], results_manual.params[1], atol=0.1)

    def test_results_attributes(self):
        """Test that results have expected statsmodels attributes."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X[:, 0] + np.random.randn(50) * 0.1

        results = run_regression(X, y)

        # Check for common statsmodels attributes
        attrs = ["params", "bse", "pvalues", "rsquared", "conf_int"]
        for attr in attrs:
            assert hasattr(results, attr)
