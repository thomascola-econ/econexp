"""
Tests for the econexp.power module.

Covers: MDEResult, mde, mde_binary, required_n, achieved_power,
icc_from_data, mde_clustered, mde_did, mde_from_data, mde_multi_arm,
and cross-function consistency.
"""

import numpy as np
import pytest

from econexp.power import (
    MDEResult,
    achieved_power,
    icc_from_data,
    mde,
    mde_binary,
    mde_clustered,
    mde_did,
    mde_from_data,
    mde_multi_arm,
    required_n,
)


class TestMDEResult:
    def test_mderesult_fields(self):
        r = MDEResult(
            mde=0.3,
            n_per_arm=100,
            power=0.80,
            alpha=0.05,
            variance=1.0,
            design_effect=1.0,
            notes="test",
        )
        assert r.mde == 0.3
        assert r.n_per_arm == 100
        assert r.power == 0.80
        assert r.alpha == 0.05
        assert r.variance == 1.0
        assert r.design_effect == 1.0
        assert r.notes == "test"

    def test_mderesult_none_fields(self):
        r = MDEResult(
            mde=None,
            n_per_arm=None,
            power=None,
            alpha=0.05,
            variance=1.0,
            design_effect=1.0,
            notes="none fields ok",
        )
        assert r.mde is None
        assert r.n_per_arm is None
        assert r.power is None


class TestMDE:
    def test_mde_basic(self):
        result = mde(1.0, 100)
        assert isinstance(result, MDEResult)
        assert result.mde > 0

    def test_mde_known_value(self):
        # (1.96 + 0.842) * sqrt(2/100) ≈ 0.3963
        result = mde(1.0, 100, alpha=0.05, power=0.80)
        assert np.isclose(result.mde, 0.3963, atol=1e-3)

    def test_mde_larger_n_smaller_mde(self):
        assert mde(1.0, 400).mde < mde(1.0, 100).mde

    def test_mde_larger_variance_larger_mde(self):
        assert mde(2.0, 100).mde > mde(1.0, 100).mde

    def test_mde_higher_power_larger_mde(self):
        assert mde(1.0, 100, power=0.90).mde > mde(1.0, 100, power=0.80).mde

    def test_mde_one_sided_smaller_than_two_sided(self):
        assert mde(1.0, 100, two_sided=False).mde < mde(1.0, 100, two_sided=True).mde

    def test_mde_design_effect_is_one(self):
        assert mde(1.0, 100).design_effect == 1.0

    def test_mde_n_per_arm_stored(self):
        assert mde(1.0, 100).n_per_arm == 100

    def test_mde_invalid_variance_zero(self):
        with pytest.raises(ValueError):
            mde(0.0, 100)

    def test_mde_invalid_variance_negative(self):
        with pytest.raises(ValueError):
            mde(-1.0, 100)

    def test_mde_invalid_n_one(self):
        with pytest.raises(ValueError):
            mde(1.0, 1)

    def test_mde_invalid_n_zero(self):
        with pytest.raises(ValueError):
            mde(1.0, 0)

    def test_mde_invalid_alpha_zero(self):
        with pytest.raises(ValueError):
            mde(1.0, 100, alpha=0)

    def test_mde_invalid_alpha_one(self):
        with pytest.raises(ValueError):
            mde(1.0, 100, alpha=1)

    def test_mde_invalid_alpha_large(self):
        with pytest.raises(ValueError):
            mde(1.0, 100, alpha=1.5)

    def test_mde_invalid_power_zero(self):
        with pytest.raises(ValueError):
            mde(1.0, 100, power=0)

    def test_mde_invalid_power_one(self):
        with pytest.raises(ValueError):
            mde(1.0, 100, power=1)


class TestMDEBinary:
    def test_mde_binary_basic(self):
        result = mde_binary(0.5, 100)
        assert isinstance(result, MDEResult)
        assert result.mde > 0

    def test_mde_binary_max_variance_at_half(self):
        # p=0.5 gives max variance 0.25; p=0.1 gives 0.09 → larger MDE
        assert mde_binary(0.5, 100).mde > mde_binary(0.1, 100).mde

    def test_mde_binary_matches_mde_bernoulli_variance(self):
        p = 0.3
        assert np.isclose(mde_binary(p, 100).mde, mde(p * (1 - p), 100).mde)

    def test_mde_binary_notes_contain_p(self):
        result = mde_binary(0.4, 100)
        assert "p=" in result.notes

    def test_mde_binary_invalid_p_zero(self):
        with pytest.raises(ValueError):
            mde_binary(0.0, 100)

    def test_mde_binary_invalid_p_one(self):
        with pytest.raises(ValueError):
            mde_binary(1.0, 100)

    def test_mde_binary_invalid_p_negative(self):
        with pytest.raises(ValueError):
            mde_binary(-0.1, 100)

    def test_mde_binary_invalid_p_above_one(self):
        with pytest.raises(ValueError):
            mde_binary(1.5, 100)


class TestRequiredN:
    def test_required_n_basic(self):
        result = required_n(0.3, 1.0)
        assert isinstance(result, MDEResult)
        assert result.n_per_arm > 0

    def test_required_n_is_int(self):
        result = required_n(0.3, 1.0)
        assert isinstance(result.n_per_arm, int)

    def test_required_n_roundtrip(self):
        # ceiling ensures that mde at the returned n is ≤ the target mde
        mde_target = 0.3
        n = required_n(mde_target, 1.0).n_per_arm
        assert mde(1.0, n).mde <= mde_target + 1e-10

    def test_required_n_larger_mde_fewer_obs(self):
        assert required_n(0.5, 1.0).n_per_arm < required_n(0.1, 1.0).n_per_arm

    def test_required_n_known_value(self):
        # mde=0.4, var=1.0, alpha=0.05, power=0.80
        # n_exact = 2 * ((1.96+0.842)/0.4)^2 ≈ 98.14 → ceil = 99
        result = required_n(0.4, 1.0, alpha=0.05, power=0.80)
        assert result.n_per_arm == 99

    def test_required_n_invalid_mde_zero(self):
        with pytest.raises(ValueError):
            required_n(0.0, 1.0)

    def test_required_n_invalid_mde_negative(self):
        with pytest.raises(ValueError):
            required_n(-0.1, 1.0)

    def test_required_n_invalid_variance(self):
        with pytest.raises(ValueError):
            required_n(0.3, 0.0)


class TestAchievedPower:
    def test_achieved_power_basic(self):
        result = achieved_power(0.5, 1.0, 100)
        assert isinstance(result, MDEResult)
        assert 0 < result.power < 1

    def test_achieved_power_larger_n_more_power(self):
        assert achieved_power(0.5, 1.0, 400).power > achieved_power(0.5, 1.0, 100).power

    def test_achieved_power_larger_effect_more_power(self):
        assert achieved_power(1.0, 1.0, 100).power > achieved_power(0.3, 1.0, 100).power

    def test_achieved_power_at_mde_equals_target(self):
        mde_val = mde(1.0, 100, power=0.80).mde
        ap = achieved_power(mde_val, 1.0, 100).power
        assert np.isclose(ap, 0.80, atol=1e-6)

    def test_achieved_power_negative_delta_same_as_positive(self):
        assert np.isclose(
            achieved_power(-0.5, 1.0, 100).power,
            achieved_power(0.5, 1.0, 100).power,
        )

    def test_achieved_power_delta_zero_returns_alpha_half(self):
        result = achieved_power(0.0, 1.0, 100, alpha=0.05, two_sided=True)
        assert np.isclose(result.power, 0.025, atol=1e-10)

    def test_achieved_power_mde_is_none(self):
        assert achieved_power(0.5, 1.0, 100).mde is None

    def test_achieved_power_invalid_variance(self):
        with pytest.raises(ValueError):
            achieved_power(0.5, 0.0, 100)

    def test_achieved_power_invalid_n(self):
        with pytest.raises(ValueError):
            achieved_power(0.5, 1.0, 1)


class TestICCFromData:
    def setup_method(self):
        np.random.seed(42)
        # strong clustering: cluster mean dominates
        self.clusters = np.repeat(np.arange(20), 10)
        self.y_high = self.clusters * 2.0 + np.random.randn(200) * 0.1
        # no clustering: random y
        self.y_low = np.random.randn(200)

    def test_icc_from_data_high_clustering(self):
        icc = icc_from_data(self.y_high, self.clusters)
        assert icc > 0.5

    def test_icc_from_data_no_clustering(self):
        icc = icc_from_data(self.y_low, self.clusters)
        assert icc < 0.1

    def test_icc_from_data_returns_float(self):
        assert isinstance(icc_from_data(self.y_low, self.clusters), float)

    def test_icc_from_data_in_range(self):
        icc = icc_from_data(self.y_low, self.clusters)
        assert 0.0 <= icc <= 1.0

    def test_icc_from_data_clipped_to_zero(self):
        # constant within clusters → negative ICC estimate possible; clipped to 0
        y = np.tile(np.arange(20), 10).astype(float)
        clusters = np.repeat(np.arange(20), 10)
        icc = icc_from_data(y + np.random.randn(200) * 100, clusters)
        assert icc >= 0.0

    def test_icc_from_data_one_cluster_raises(self):
        with pytest.raises(ValueError):
            icc_from_data(self.y_low, np.zeros(200, dtype=int))


class TestMDEClustered:
    def test_mde_clustered_basic(self):
        result = mde_clustered(1.0, n_clusters=20, cluster_size=10, icc=0.1)
        assert isinstance(result, MDEResult)
        assert result.mde > 0

    def test_mde_clustered_icc_zero_matches_unclustered(self):
        result_cl = mde_clustered(1.0, n_clusters=20, cluster_size=10, icc=0.0)
        result_uc = mde(1.0, 20 * 10)
        assert np.isclose(result_cl.mde, result_uc.mde)

    def test_mde_clustered_higher_icc_larger_mde(self):
        low = mde_clustered(1.0, 20, 10, icc=0.1)
        high = mde_clustered(1.0, 20, 10, icc=0.3)
        assert high.mde > low.mde

    def test_mde_clustered_design_effect_stored(self):
        icc = 0.2
        cluster_size = 10
        result = mde_clustered(1.0, 20, cluster_size, icc)
        expected_deff = 1 + (cluster_size - 1) * icc
        assert np.isclose(result.design_effect, expected_deff)

    def test_mde_clustered_variance_is_unadjusted(self):
        # result.variance stores the original (unadjusted) variance
        result = mde_clustered(1.0, 20, 10, icc=0.2)
        assert result.variance == 1.0

    def test_mde_clustered_invalid_icc_negative(self):
        with pytest.raises(ValueError):
            mde_clustered(1.0, 20, 10, icc=-0.1)

    def test_mde_clustered_invalid_icc_above_one(self):
        with pytest.raises(ValueError):
            mde_clustered(1.0, 20, 10, icc=1.1)

    def test_mde_clustered_invalid_cluster_size(self):
        with pytest.raises(ValueError):
            mde_clustered(1.0, 20, 0, icc=0.1)

    def test_mde_clustered_invalid_n_clusters(self):
        with pytest.raises(ValueError):
            mde_clustered(1.0, 0, 10, icc=0.1)


class TestMDEDiD:
    def setup_method(self):
        np.random.seed(42)
        self.y_pre = np.random.randn(100)
        self.y_post = 0.8 * self.y_pre + np.random.randn(100) * 0.5

    def test_mde_did_with_rho(self):
        result = mde_did(1.0, 100, rho=0.5)
        assert isinstance(result, MDEResult)
        assert result.mde > 0

    def test_mde_did_with_arrays(self):
        result = mde_did(1.0, 100, y_pre=self.y_pre, y_post=self.y_post)
        assert isinstance(result, MDEResult)
        assert result.mde > 0

    def test_mde_did_high_rho_smaller_mde(self):
        assert mde_did(1.0, 100, rho=0.9).mde < mde_did(1.0, 100, rho=0.1).mde

    def test_mde_did_rho_zero_matches_double_variance(self):
        # sigma_diff^2 = 2 * sigma^2 * (1 - 0) = 2 * sigma^2
        result_did = mde_did(1.0, 100, rho=0.0)
        result_base = mde(2.0, 100)
        assert np.isclose(result_did.mde, result_base.mde)

    def test_mde_did_arrays_match_manual_rho(self):
        rho_manual = float(np.corrcoef(self.y_pre, self.y_post)[0, 1])
        result_arrays = mde_did(1.0, 100, y_pre=self.y_pre, y_post=self.y_post)
        result_manual = mde_did(1.0, 100, rho=rho_manual)
        assert np.isclose(result_arrays.mde, result_manual.mde)

    def test_mde_did_variance_stored_unadjusted(self):
        result = mde_did(1.0, 100, rho=0.5)
        assert result.variance == 1.0

    def test_mde_did_missing_args_raises(self):
        with pytest.raises(ValueError):
            mde_did(1.0, 100)

    def test_mde_did_missing_y_post_raises(self):
        with pytest.raises(ValueError):
            mde_did(1.0, 100, y_pre=self.y_pre)

    def test_mde_did_mismatched_arrays_raises(self):
        with pytest.raises(ValueError):
            mde_did(1.0, 100, y_pre=self.y_pre, y_post=self.y_post[:50])

    def test_mde_did_invalid_rho_above_one(self):
        with pytest.raises(ValueError):
            mde_did(1.0, 100, rho=2.0)

    def test_mde_did_invalid_rho_below_neg_one(self):
        with pytest.raises(ValueError):
            mde_did(1.0, 100, rho=-2.0)


class TestMDEFromData:
    def setup_method(self):
        np.random.seed(42)
        self.y = np.random.randn(200)
        self.X = np.random.randn(200, 2)

    def test_mde_from_data_no_covariates(self):
        result = mde_from_data(self.y, 100)
        assert isinstance(result, MDEResult)
        assert np.isclose(result.variance, np.var(self.y, ddof=1), rtol=1e-6)

    def test_mde_from_data_with_covariates(self):
        result = mde_from_data(self.y, 100, X=self.X)
        assert isinstance(result, MDEResult)
        assert result.mde > 0

    def test_mde_from_data_high_r2_reduces_mde(self):
        np.random.seed(42)
        X = np.random.randn(200, 1)
        y_correlated = X[:, 0] * 5 + np.random.randn(200) * 0.1
        result_with = mde_from_data(y_correlated, 100, X=X)
        result_without = mde_from_data(y_correlated, 100)
        assert result_with.mde < result_without.mde

    def test_mde_from_data_notes_contain_r2(self):
        result = mde_from_data(self.y, 100, X=self.X)
        assert "R²" in result.notes

    def test_mde_from_data_no_covariates_notes(self):
        result = mde_from_data(self.y, 100)
        assert "variance" in result.notes.lower()

    def test_mde_from_data_too_few_obs(self):
        with pytest.raises(ValueError):
            mde_from_data(np.array([1.0, 2.0]), 10)

    def test_mde_from_data_returns_mde_result(self):
        assert isinstance(mde_from_data(self.y, 100), MDEResult)


class TestMDEMultiArm:
    def test_mde_multi_arm_basic(self):
        result = mde_multi_arm(1.0, 300, n_arms=2)
        assert isinstance(result, MDEResult)
        assert result.mde > 0

    def test_mde_multi_arm_more_arms_larger_mde(self):
        assert (
            mde_multi_arm(1.0, 300, n_arms=4).mde
            > mde_multi_arm(1.0, 300, n_arms=2).mde
        )

    def test_mde_multi_arm_one_arm_matches_two_arm(self):
        result_multi = mde_multi_arm(1.0, 200, n_arms=1)
        result_base = mde(1.0, 100)
        assert np.isclose(result_multi.mde, result_base.mde)

    def test_mde_multi_arm_bonferroni_larger_mde(self):
        no_correction = mde_multi_arm(1.0, 300, n_arms=3, correct_alpha=False)
        with_correction = mde_multi_arm(1.0, 300, n_arms=3, correct_alpha=True)
        assert with_correction.mde > no_correction.mde

    def test_mde_multi_arm_alpha_stored_uncorrected(self):
        result = mde_multi_arm(1.0, 300, n_arms=3, correct_alpha=True, alpha=0.05)
        assert result.alpha == 0.05

    def test_mde_multi_arm_invalid_n_arms_zero(self):
        with pytest.raises(ValueError):
            mde_multi_arm(1.0, 300, n_arms=0)

    def test_mde_multi_arm_insufficient_n(self):
        with pytest.raises(ValueError):
            mde_multi_arm(1.0, 2, n_arms=5)


class TestPowerConsistency:
    """Cross-function consistency checks."""

    def test_mde_required_n_roundtrip(self):
        delta = 0.3
        var = 1.0
        n = required_n(delta, var).n_per_arm
        assert mde(var, n).mde <= delta + 1e-10

    def test_achieved_power_at_mde_is_target(self):
        var = 1.0
        n = 100
        mde_val = mde(var, n, power=0.80).mde
        ap = achieved_power(mde_val, var, n).power
        assert np.isclose(ap, 0.80, atol=1e-6)

    def test_binary_matches_continuous_bernoulli_var(self):
        p = 0.4
        n = 100
        assert np.isclose(mde_binary(p, n).mde, mde(p * (1 - p), n).mde)

    def test_did_rho_zero_is_double_variance(self):
        var = 1.0
        n = 100
        assert np.isclose(
            mde_did(var, n, rho=0.0).mde,
            mde(2 * var, n).mde,
        )

    def test_clustered_icc_zero_matches_unclustered(self):
        var = 1.0
        g, m = 20, 10
        assert np.isclose(
            mde_clustered(var, g, m, icc=0.0).mde,
            mde(var, g * m).mde,
        )

    def test_required_n_ceiling_guarantees_power(self):
        # The ceiling in required_n means achieved power >= target
        target_power = 0.80
        n = required_n(0.5, 1.0, power=target_power).n_per_arm
        actual_power = achieved_power(0.5, 1.0, n).power
        assert actual_power >= target_power - 1e-6

    def test_mde_from_data_no_x_matches_mde_with_var(self):
        np.random.seed(42)
        y = np.random.randn(200)
        var_est = float(np.var(y, ddof=1))
        result_data = mde_from_data(y, 100)
        result_direct = mde(var_est, 100)
        assert np.isclose(result_data.mde, result_direct.mde)

    def test_required_n_inverse_of_mde(self):
        # mde and required_n are exact inverses (up to ceiling)
        var = 2.0
        target_mde = 0.5
        n = required_n(target_mde, var).n_per_arm
        # n-1 should NOT achieve the target mde (ceiling property)
        assert mde(var, n - 1).mde > target_mde or n == 1
        # n should achieve it
        assert mde(var, n).mde <= target_mde + 1e-10
