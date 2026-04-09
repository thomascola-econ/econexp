"""
Plots demonstrating the functionality of econexp.power.

Run with:
    python plot_power.py

Saves power_demo.png in the current directory.
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from econexp.power import (
    achieved_power,
    mde,
    mde_binary,
    mde_clustered,
    mde_did,
    mde_from_data,
    required_n,
    simulate_power_within,
)

# ── shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _label(ax, letter, title):
    ax.set_title(f"({letter})  {title}", fontsize=10, fontweight="bold", loc="left")


# ── figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 30))
gs = gridspec.GridSpec(6, 2, figure=fig, hspace=0.55, wspace=0.35)
axes = [fig.add_subplot(gs[r, c]) for r in range(6) for c in range(2)]


# ── (a) MDE vs sample size ────────────────────────────────────────────────────
ax = axes[0]
ns = np.arange(20, 1001, 10)
for i, (power_level, ls) in enumerate([(0.80, "-"), (0.90, "--"), (0.95, ":")]):
    mdes = [mde(1.0, n, power=power_level).mde for n in ns]
    ax.plot(ns, mdes, ls, color=COLORS[i], label=f"power = {power_level}")
ax.set_xlabel("Per-arm sample size (n)")
ax.set_ylabel("MDE (σ = 1)")
ax.legend(fontsize=8)
_label(ax, "a", "MDE vs. sample size")


# ── (b) Power curve ───────────────────────────────────────────────────────────
ax = axes[1]
deltas = np.linspace(0, 1.0, 200)
for i, (n, ls) in enumerate([(50, ":"), (100, "--"), (200, "-"), (500, "-.")]):
    powers = [achieved_power(d, 1.0, n).power for d in deltas]
    ax.plot(deltas, powers, ls, color=COLORS[i], label=f"n = {n}")
ax.axhline(0.80, color="grey", linewidth=0.8, linestyle="--")
ax.text(0.01, 0.81, "80% power", fontsize=7, color="grey")
ax.set_xlabel("True effect size (δ)")
ax.set_ylabel("Achieved power")
ax.set_ylim(0, 1.02)
ax.legend(fontsize=8)
_label(ax, "b", "Power curve (σ = 1)")


# ── (c) Required sample size vs target MDE ────────────────────────────────────
ax = axes[2]
mde_targets = np.linspace(0.05, 0.8, 200)
for i, (var, ls) in enumerate([(0.5, ":"), (1.0, "-"), (2.0, "--")]):
    ns_req = [required_n(m, var).n_per_arm for m in mde_targets]
    ax.plot(mde_targets, ns_req, ls, color=COLORS[i], label=f"σ² = {var}")
ax.set_xlabel("Target MDE")
ax.set_ylabel("Required n per arm")
ax.set_ylim(0, 2500)
ax.legend(fontsize=8)
_label(ax, "c", "Required sample size vs. target MDE")


# ── (d) Binary outcome: MDE vs baseline proportion ───────────────────────────
ax = axes[3]
ps = np.linspace(0.02, 0.98, 300)
for i, (n, ls) in enumerate([(100, ":"), (300, "--"), (1000, "-")]):
    mdes_bin = [mde_binary(p, n).mde for p in ps]
    ax.plot(ps, mdes_bin, ls, color=COLORS[i], label=f"n = {n}")
ax.set_xlabel("Baseline proportion (p)")
ax.set_ylabel("MDE (absolute Δp)")
ax.legend(fontsize=8)
_label(ax, "d", "Binary outcome: MDE vs. baseline proportion")


# ── (e) Clustered RCT: MDE vs ICC ─────────────────────────────────────────────
ax = axes[4]
iccs = np.linspace(0, 0.5, 200)
n_clusters = 30
for i, (m, ls) in enumerate([(5, ":"), (10, "--"), (25, "-"), (50, "-.")]):
    mdes_cl = [mde_clustered(1.0, n_clusters, m, icc).mde for icc in iccs]
    ax.plot(iccs, mdes_cl, ls, color=COLORS[i], label=f"cluster size = {m}")
unclustered = mde(1.0, n_clusters * 10).mde
ax.axhline(unclustered, color="grey", linewidth=0.8, linestyle="--")
ax.text(0.01, unclustered * 1.02, "unclustered (m=10)", fontsize=7, color="grey")
ax.set_xlabel("Intra-cluster correlation (ICC)")
ax.set_ylabel("MDE (σ = 1)")
ax.legend(fontsize=8, title=f"{n_clusters} clusters/arm")
_label(ax, "e", "Clustered RCT: MDE vs. ICC")


# ── (f) DiD / panel: MDE vs pre-post correlation ─────────────────────────────
ax = axes[5]
rhos = np.linspace(0, 0.99, 200)
for i, (n, ls) in enumerate([(50, ":"), (100, "--"), (200, "-")]):
    mdes_did = [mde_did(1.0, n, rho=r).mde for r in rhos]
    ax.plot(rhos, mdes_did, ls, color=COLORS[i], label=f"n = {n}")
# mark where DiD equals cross-section (rho = 0)
ax.set_xlabel("Pre-post correlation (ρ)")
ax.set_ylabel("MDE (σ = 1)")
ax.legend(fontsize=8)
_label(ax, "f", "DiD / panel: MDE vs. pre-post correlation")


# ── (g) ANCOVA: MDE reduction from covariate adjustment ──────────────────────
ax = axes[6]
np.random.seed(42)
n_obs = 500
r2_values = np.linspace(0, 0.95, 100)
n_per_arm = 200

# Theoretical: MDE ∝ sqrt(1 - R²)
mde_no_cov = mde(1.0, n_per_arm).mde
mde_adjusted = [mde_no_cov * np.sqrt(1 - r2) for r2 in r2_values]
ax.plot(r2_values, mde_adjusted, color=COLORS[0], label="theoretical")

# Data-driven: generate datasets with varying R² and estimate from residuals
np.random.seed(42)
r2_empirical, mde_empirical = [], []
for target_r2 in np.linspace(0.05, 0.90, 15):
    X_cov = np.random.randn(n_obs, 1)
    signal_sd = np.sqrt(target_r2 / (1 - target_r2))
    y_raw = X_cov[:, 0] * signal_sd + np.random.randn(n_obs)
    # Standardize to unit variance so mse_resid ≈ (1 - R²) and the
    # unadjusted baseline (var=1) matches the theoretical curve.
    y_obs = y_raw / y_raw.std()
    result = mde_from_data(y_obs, n_per_arm, X=X_cov)
    r2_empirical.append(1 - result.variance / np.var(y_obs, ddof=1))
    mde_empirical.append(result.mde)
ax.scatter(
    r2_empirical,
    mde_empirical,
    color=COLORS[1],
    zorder=5,
    s=30,
    label="data-driven (mde_from_data)",
)

ax.axhline(mde_no_cov, color="grey", linewidth=0.8, linestyle="--")
ax.text(0.01, mde_no_cov * 1.02, "no adjustment", fontsize=7, color="grey")
ax.set_xlabel("Covariate R²")
ax.set_ylabel("MDE (n = 200 per arm)")
ax.legend(fontsize=8)
_label(ax, "g", "ANCOVA covariate adjustment: MDE vs. R²")


# ── (h) Multi-arm: MDE vs number of arms ─────────────────────────────────────
ax = axes[7]
arm_counts = np.arange(1, 9)
n_total = 600
for i, (correct, ls, label) in enumerate(
    [
        (False, "-", "no correction"),
        (True, "--", "Bonferroni"),
    ]
):
    mdes_ma = [
        mde(1.0, n_total // (k + 1)).mde
        if not correct
        else mde(1.0, n_total // (k + 1), alpha=0.05 / k).mde
        for k in arm_counts
    ]
    ax.plot(
        arm_counts, mdes_ma, ls, color=COLORS[i], marker="o", markersize=5, label=label
    )
ax.set_xlabel("Number of treatment arms")
ax.set_ylabel("MDE per arm (σ = 1)")
ax.set_xticks(arm_counts)
ax.legend(fontsize=8)
_label(ax, "h", f"Multi-arm: MDE vs. arms (N = {n_total} total)")


# ── (i) simulate_power_within: simulation vs analytical ──────────────────────
ax = axes[8]
np.random.seed(0)
# Generate a mildly right-skewed pilot dataset (log-normal minus its mean)
pilot = np.exp(np.random.randn(300)) - np.exp(0.5)
pilot_std = pilot.std()

effects = np.linspace(0, 0.8 * pilot_std, 30)

for i, (n, ls) in enumerate([(50, ":"), (100, "--"), (200, "-")]):
    # Simulation-based power (y_pre only — assumes rho=0)
    sim_powers = [
        simulate_power_within(e, n, pilot, n_sims=2000, seed=42).power for e in effects
    ]
    ax.plot(
        effects / pilot_std,
        sim_powers,
        ls,
        color=COLORS[i],
        label=f"simulated n={n}",
        zorder=3,
    )

    # Analytical reference (achieved_power with 2*Var for rho=0)
    ana_powers = [achieved_power(e, 2 * pilot.var(ddof=1), n).power for e in effects]
    ax.plot(
        effects / pilot_std, ana_powers, ls, color=COLORS[i], alpha=0.35, linewidth=1.5
    )

ax.axhline(0.80, color="grey", linewidth=0.8, linestyle="--")
ax.text(0.01, 0.81, "80% power", fontsize=7, color="grey")
ax.set_xlabel("Effect size (δ / σ)")
ax.set_ylabel("Power")
ax.set_ylim(0, 1.02)
ax.legend(fontsize=7, title="solid=sim, faded=analytical")
_label(
    ax,
    "i",
    "Within-design simulation vs. normal approximation\n(skewed pilot; rho=0 assumed)",
)

# ── (j) simulate_power_within: effect of rho via paired pilot data ────────────
ax = axes[9]
np.random.seed(1)
n_pilot = 400
n_exp = 150
effect_fixed = 0.25 * pilot_std  # fixed effect to test

rhos_target = [0.0, 0.3, 0.6, 0.85]
for i, rho_t in enumerate(rhos_target):
    # Construct paired pilot data with a specific correlation
    z = np.random.randn(n_pilot)
    noise = np.random.randn(n_pilot)
    pre_p = z
    post_p = rho_t * z + np.sqrt(1 - rho_t**2) * noise
    # Scale to match pilot variance
    pre_p = pre_p * pilot_std
    post_p = post_p * pilot_std

    actual_rho = float(np.corrcoef(pre_p, post_p)[0, 1])

    ns_range = np.arange(20, 301, 10)
    sim_pows = [
        simulate_power_within(
            effect_fixed, n, pre_p, y_post=post_p, n_sims=1000, seed=7
        ).power
        for n in ns_range
    ]
    ax.plot(ns_range, sim_pows, color=COLORS[i], label=f"ρ ≈ {actual_rho:.2f}")

ax.axhline(0.80, color="grey", linewidth=0.8, linestyle="--")
ax.text(5, 0.81, "80%", fontsize=7, color="grey")
ax.set_xlabel("Per-experiment sample size (n)")
ax.set_ylabel("Simulated power")
ax.set_ylim(0, 1.02)
ax.legend(fontsize=8, title=f"δ = {effect_fixed:.2f}")
_label(
    ax,
    "j",
    "Within-design: power vs. n for different pre-post correlations\n"
    "(paired bootstrap from pilot data)",
)


# ── (k,l) Within vs. Between design — same units, same periods (T=2) ─────────
#
# Budget framing: n total units, each observed T=2 times (one pre, one post).
#   Between design — split n/2 to treatment, n/2 to control; estimator uses
#                    only the post-period measurement. n/2 obs per arm.
#   Within design  — all n units observed pre and post; estimator uses the
#                    within-unit difference. n paired observations.
# Both designs consume the same 2n measurements and the same n units.
#
# Between SE = sqrt(4σ²/n),  Within SE = sqrt(2σ²(1-ρ)/n)
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(2)
n_pilot_k = 600
sigma_k = 1.0

# Build pilot datasets for four ρ values
rhos_k = [0.0, 0.3, 0.6, 0.85]
pilots_k = {}
for rho_t in rhos_k:
    z = np.random.randn(n_pilot_k)
    eps = np.random.randn(n_pilot_k)
    pre_k = z * sigma_k
    post_k = (rho_t * z + np.sqrt(1 - rho_t**2) * eps) * sigma_k
    pilots_k[rho_t] = (pre_k, post_k)

# ── (k) power vs effect size: within (sim) vs between (analytical) ────────────
ax = axes[10]
n_k = 120  # total units in the budget
effects_k = np.linspace(0, 1.2 * sigma_k, 40)

# Between-design power: n_k/2 per arm, uses only post-period variance (σ²=1)
between_powers = [achieved_power(e, sigma_k**2, n_k // 2).power for e in effects_k]
ax.plot(
    effects_k / sigma_k, between_powers, "k--", linewidth=1.8, label="between (any ρ)"
)

for i, rho_t in enumerate(rhos_k):
    pre_k, post_k = pilots_k[rho_t]
    actual_rho = float(np.corrcoef(pre_k, post_k)[0, 1])
    within_pows = [
        simulate_power_within(e, n_k, pre_k, y_post=post_k, n_sims=1000, seed=42).power
        for e in effects_k
    ]
    ax.plot(
        effects_k / sigma_k,
        within_pows,
        color=COLORS[i],
        label=f"within ρ≈{actual_rho:.2f}",
    )

ax.axhline(0.80, color="grey", linewidth=0.8, linestyle="--")
ax.text(0.02, 0.815, "80%", fontsize=7, color="grey")
ax.set_xlabel("Effect size (δ / σ)")
ax.set_ylabel("Power")
ax.set_ylim(0, 1.02)
ax.legend(fontsize=7.5)
_label(
    ax,
    "k",
    f"Within vs. between: power curves (n={n_k} units, T=2 periods each)\n"
    "Between splits n/2 per arm; within uses all n as paired obs",
)

# ── (l) unit-efficiency: ratio of required n to reach 80% power ───────────────
ax = axes[11]
rhos_l = np.linspace(0.0, 0.95, 200)
effects_l = [0.2 * sigma_k, 0.4 * sigma_k, 0.6 * sigma_k]

for i, eff in enumerate(effects_l):
    # Analytical required n for between design: n_between/2 per arm
    # MDE_between = (z_a+z_b)*sqrt(4σ²/n_between) → n_between = 4σ²*(z_a+z_b)²/δ²
    n_between = [required_n(eff, sigma_k**2).n_per_arm * 2 for _ in rhos_l]

    # Analytical required n for within design: n_within paired obs
    # MDE_within = (z_a+z_b)*sqrt(2σ²(1-ρ)/n_within)
    # n_within = 2σ²(1-ρ)*(z_a+z_b)²/δ²
    n_within = [
        max(required_n(eff, 2 * sigma_k**2 * (1 - rho)).n_per_arm, 1) for rho in rhos_l
    ]

    ratio = [nb / nw for nb, nw in zip(n_between, n_within)]
    ax.plot(rhos_l, ratio, color=COLORS[i], label=f"δ = {eff:.1f}σ")

# Theoretical curve (same for all δ): ratio = 2/(1-ρ)
theoretical = 2 / (1 - rhos_l)
ax.plot(rhos_l, theoretical, "k:", linewidth=1.2, label="2/(1−ρ)  [theory]")

ax.axhline(1, color="grey", linewidth=0.8, linestyle="--")
ax.text(0.01, 1.05, "break-even", fontsize=7, color="grey")
ax.set_xlabel("Pre-post correlation (ρ)")
ax.set_ylabel("n_between / n_within  (units needed)")
ax.set_ylim(0, 22)
ax.legend(fontsize=8)
_label(
    ax,
    "l",
    "Unit-efficiency gain of within over between design\n"
    "Same T=2 periods per unit; ratio >1 means within needs fewer units",
)


# ── save ──────────────────────────────────────────────────────────────────────
fig.suptitle(
    "econexp.power — Minimum Detectable Effect Size Demo",
    fontsize=13,
    fontweight="bold",
    y=0.995,
)
plt.savefig("power_demo.png", dpi=150, bbox_inches="tight")
print("Saved power_demo.png")
