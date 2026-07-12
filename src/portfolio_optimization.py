"""
Investment Portfolio Optimization - Markowitz Mean-Variance Model
14 assets across Equity, Fixed Income, Alternatives, and Cash.
5 years of monthly return history (built to realistic risk/return/
correlation assumptions) used to estimate the risk model.

STRUCTURE OF THIS SCRIPT:
0. The decision problem
1. Data overview and quality checks
2. Exploratory analysis - risk and correlation structure
3. Unconstrained optimization - max Sharpe ratio and minimum variance
4. Constrained optimization - realistic investment policy limits
5. What the constraints actually cost (or don't)
6. Efficient frontier
7. Recommendation and impact

Run from the src/ folder:
    python portfolio_optimization.py
Charts save to ../outputs/.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

DATA_DIR = "../data"
OUTPUT_DIR = "../outputs"
sns.set_style("whitegrid")

RISK_FREE_RATE = 0.02

# ===========================================================================
# PART 0 - THE DECISION PROBLEM
# ===========================================================================
print("=== PART 0: The Decision Problem ===")
print(
    "An investor has capital to allocate across 14 assets spanning equities, "
    "fixed income, alternatives, and cash. The question: how should that "
    "capital be split to get the best possible return for a given level of "
    "risk - and what does it cost, in return given up, to also satisfy "
    "realistic investment policy limits (minimum bond holdings, maximum "
    "single-asset exposure, and so on) rather than optimizing with no "
    "constraints at all?\n"
)

# ===========================================================================
# PART 1 - DATA OVERVIEW AND QUALITY CHECKS
# ===========================================================================
returns_df = pd.read_csv(f"{DATA_DIR}/monthly_returns.csv", index_col="Date")
asset_info = pd.read_csv(f"{DATA_DIR}/asset_info.csv")

names = asset_info["Asset"].tolist()
sectors = asset_info["Sector"].tolist()
sector_array = np.array(sectors)
expected_returns = asset_info["Target_Annual_Return"].values
n_assets = len(names)

print("=== PART 1: Data Overview and Quality Checks ===")
print("Monthly returns shape:", returns_df.shape, "(60 months x 14 assets)")
print("Missing values:", returns_df.isna().sum().sum())
print("Any monthly return outside a sane range (-50% to +50%)?",
      ((returns_df < -0.5) | (returns_df > 0.5)).sum().sum())
print(
    "\nData is clean - no missing months, no implausible return values. "
    "60 months (5 years) is enough to estimate risk (volatility, "
    "correlation) reliably, but is genuinely too short to estimate average "
    "return reliably, which is why the next section uses forward-looking "
    "return assumptions instead of the raw historical average.\n"
)

# ===========================================================================
# PART 2 - EXPLORATORY ANALYSIS: risk and correlation structure
# ===========================================================================
realized_annual_return = returns_df.mean() * 12
realized_annual_vol = returns_df.std() * np.sqrt(12)

print("=== PART 2: Realized Risk vs Target Assumptions ===")
comparison = pd.DataFrame({
    "Target_Return": expected_returns,
    "Realized_Return": realized_annual_return.values,
    "Target_Vol": asset_info["Target_Annual_Vol"].values,
    "Realized_Vol": realized_annual_vol.values,
}, index=names)
print(comparison.round(3))
print(
    "\nRealized volatility tracks the target assumptions closely for every "
    "asset. Realized average returns don't - some assets that were assumed "
    "to return 8-14% a year actually averaged under 2% over this specific "
    "5-year window, and Commodities/Gold outperformed their targets. This "
    "is expected statistical noise, not a data problem: volatility can be "
    "estimated precisely from 60 months of data, average returns cannot. "
    "For this reason, the optimization below uses the forward-looking "
    "target returns (the assumptions an investor would actually hold "
    "going into a decision), combined with the historical covariance "
    "matrix (which is a reliable estimate from this same data).\n"
)

cov_matrix = returns_df.cov().values * 12

plt.figure(figsize=(10, 8))
sns.heatmap(returns_df.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            xticklabels=names, yticklabels=names)
plt.title("Asset Correlation Matrix")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/chart1_correlation_matrix.png", dpi=200)
plt.close()

# ===========================================================================
# PART 3 - UNCONSTRAINED OPTIMIZATION: max Sharpe and min variance
# ===========================================================================
def portfolio_return(w):
    return np.dot(w, expected_returns)

def portfolio_vol(w):
    return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

def neg_sharpe(w):
    return -(portfolio_return(w) - RISK_FREE_RATE) / portfolio_vol(w)

basic_constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
bounds = [(0, 0.25) for _ in range(n_assets)]  # no short-selling, max 25% per asset
init_guess = np.array([1 / n_assets] * n_assets)

result_sharpe = minimize(neg_sharpe, init_guess, method="SLSQP", bounds=bounds, constraints=basic_constraints)
weights_sharpe = result_sharpe.x

result_minvar = minimize(portfolio_vol, init_guess, method="SLSQP", bounds=bounds, constraints=basic_constraints)
weights_minvar = result_minvar.x

print("=== PART 3: Unconstrained Optimization ===")
print("Max Sharpe Ratio Portfolio:")
for name, w in zip(names, weights_sharpe):
    if w > 0.001:
        print(f"  {name}: {w:.1%}")
print(f"  Expected Return: {portfolio_return(weights_sharpe):.2%}")
print(f"  Volatility: {portfolio_vol(weights_sharpe):.2%}")
print(f"  Sharpe Ratio: {-result_sharpe.fun:.3f}")

print("\nMinimum Variance Portfolio:")
for name, w in zip(names, weights_minvar):
    if w > 0.001:
        print(f"  {name}: {w:.1%}")
print(f"  Expected Return: {portfolio_return(weights_minvar):.2%}")
print(f"  Volatility: {portfolio_vol(weights_minvar):.2%}")
print(
    "\nWith no constraints beyond 'no short-selling' and 'max 25% in any "
    "one asset', the optimizer leans heavily on bonds, cash, and gold for "
    "both portfolios - only a handful of equities make the cut, and several "
    "equity sectors get 0% allocation entirely.\n"
)

# ===========================================================================
# PART 4 - CONSTRAINED OPTIMIZATION: realistic investment policy limits
# ===========================================================================
def equity_weight(w): return np.sum(w[sector_array == "Equity"])
def fixed_income_weight(w): return np.sum(w[sector_array == "Fixed Income"])
def alternatives_weight(w): return np.sum(w[sector_array == "Alternatives"])
def cash_weight(w): return np.sum(w[sector_array == "Cash"])

# SLSQP requires inequality constraints written as "must be >= 0", so each
# policy limit (e.g. equity <= 70%) is expressed as (0.70 - equity_weight(w))
# rather than as equity_weight(w) <= 0.70 directly.
policy_constraints = [
    {"type": "eq", "fun": lambda w: np.sum(w) - 1},
    {"type": "ineq", "fun": lambda w: equity_weight(w) - 0.30},
    {"type": "ineq", "fun": lambda w: 0.70 - equity_weight(w)},
    {"type": "ineq", "fun": lambda w: fixed_income_weight(w) - 0.15},
    {"type": "ineq", "fun": lambda w: 0.20 - alternatives_weight(w)},
    {"type": "ineq", "fun": lambda w: 0.25 - cash_weight(w)},
]

result_constrained = minimize(neg_sharpe, init_guess, method="SLSQP", bounds=bounds, constraints=policy_constraints)
weights_constrained = result_constrained.x

print("=== PART 4: Constrained Optimization (Investment Policy Limits) ===")
print("Policy: Equity 30-70%, Fixed Income >= 15%, Alternatives <= 20%, Cash <= 25%\n")
print("Policy-Constrained Max Sharpe Portfolio:")
for name, sector, w in zip(names, sectors, weights_constrained):
    if w > 0.001:
        print(f"  {name} ({sector}): {w:.1%}")
print(f"  Expected Return: {portfolio_return(weights_constrained):.2%}")
print(f"  Volatility: {portfolio_vol(weights_constrained):.2%}")
print(f"  Sharpe Ratio: {-result_constrained.fun:.3f}")
print(f"\n  Sector totals: Equity {equity_weight(weights_constrained):.1%}, "
      f"Fixed Income {fixed_income_weight(weights_constrained):.1%}, "
      f"Alternatives {alternatives_weight(weights_constrained):.1%}, "
      f"Cash {cash_weight(weights_constrained):.1%}\n")

plt.figure(figsize=(10, 6))
x = np.arange(n_assets)
width = 0.35
plt.bar(x - width/2, weights_sharpe * 100, width, label="Unconstrained", color="#2E75B6")
plt.bar(x + width/2, weights_constrained * 100, width, label="Policy-Constrained", color="#C0504D")
plt.xticks(x, names, rotation=90)
plt.ylabel("Allocation (%)")
plt.title("Unconstrained vs Policy-Constrained Portfolio Allocation")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/chart2_allocation_comparison.png", dpi=200)
plt.close()

# ===========================================================================
# PART 5 - WHAT DO THE CONSTRAINTS ACTUALLY COST?
# ===========================================================================
print("=== PART 5: The Cost of Policy Constraints ===")
return_diff = portfolio_return(weights_constrained) - portfolio_return(weights_sharpe)
vol_diff = portfolio_vol(weights_constrained) - portfolio_vol(weights_sharpe)
sharpe_diff = -result_constrained.fun - (-result_sharpe.fun)

print(f"Unconstrained: {portfolio_return(weights_sharpe):.2%} return, {portfolio_vol(weights_sharpe):.2%} vol, "
      f"Sharpe {-result_sharpe.fun:.3f}")
print(f"Constrained:   {portfolio_return(weights_constrained):.2%} return, {portfolio_vol(weights_constrained):.2%} vol, "
      f"Sharpe {-result_constrained.fun:.3f}")
print(
    f"\nThe equity floor (minimum 30%) is the binding constraint - the "
    f"unconstrained optimizer actually wanted less equity than policy "
    f"allows. Being forced into more equity raises both expected return "
    f"({return_diff:+.2%}) and risk ({vol_diff:+.2%} volatility), but the "
    f"Sharpe ratio barely moves ({sharpe_diff:+.3f}). In plain terms: the "
    f"policy constraint doesn't meaningfully hurt risk-adjusted return here "
    f"- it just trades a bit more risk for a bit more return, at roughly "
    f"the same efficiency.\n"
)

# ===========================================================================
# PART 6 - EFFICIENT FRONTIER
# ===========================================================================
target_returns = np.linspace(0.033, 0.10, 20)
frontier_vols = []
for target in target_returns:
    frontier_constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w, t=target: np.dot(w, expected_returns) - t},
    ]
    result = minimize(portfolio_vol, init_guess, method="SLSQP", bounds=bounds, constraints=frontier_constraints)
    frontier_vols.append(result.fun if result.success else np.nan)

print("=== PART 6: Efficient Frontier ===")
print("Computed across 20 target return levels from 3.3% to 10%.")
print(
    "The frontier shows the minimum risk achievable for each level of "
    "target return - both portfolios above (max Sharpe and min variance) "
    "sit on this curve; anything below or to the right of it is a "
    "worse risk/return trade than what's achievable.\n"
)

plt.figure(figsize=(9, 7))
plt.plot(np.array(frontier_vols) * 100, target_returns * 100, color="#2E75B6", linewidth=2, label="Efficient Frontier")
plt.scatter([portfolio_vol(weights_sharpe) * 100], [portfolio_return(weights_sharpe) * 100],
            color="#C0504D", s=100, zorder=5, label="Max Sharpe Portfolio")
plt.scatter([portfolio_vol(weights_minvar) * 100], [portfolio_return(weights_minvar) * 100],
            color="#A9D08E", s=100, zorder=5, label="Min Variance Portfolio")
plt.scatter([portfolio_vol(weights_constrained) * 100], [portfolio_return(weights_constrained) * 100],
            color="#F4B183", s=100, zorder=5, label="Policy-Constrained Portfolio")
plt.xlabel("Volatility (%)")
plt.ylabel("Expected Return (%)")
plt.title("Efficient Frontier")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/chart3_efficient_frontier.png", dpi=200)
plt.close()

# ===========================================================================
# PART 7 - RECOMMENDATION AND IMPACT
# ===========================================================================
equal_weights = np.array([1 / n_assets] * n_assets)
equal_weight_return = portfolio_return(equal_weights)
equal_weight_vol = portfolio_vol(equal_weights)
equal_weight_sharpe = (equal_weight_return - RISK_FREE_RATE) / equal_weight_vol

print("=== PART 7: Recommendation and Impact ===")
print("Comparison against naive equal-weighting (1/14 in every asset):")
print(f"  Equal-weighted: {equal_weight_return:.2%} return, {equal_weight_vol:.2%} vol, Sharpe {equal_weight_sharpe:.3f}")
print(f"  Unconstrained optimum: Sharpe {-result_sharpe.fun:.3f}")
print(f"  Policy-constrained optimum: Sharpe {-result_constrained.fun:.3f}")
print(
    f"\nBoth optimized portfolios beat naive equal-weighting by a wide "
    f"margin on Sharpe ratio ({-result_sharpe.fun:.3f} and "
    f"{-result_constrained.fun:.3f} vs {equal_weight_sharpe:.3f}) - equal "
    f"weighting also carries much higher volatility ({equal_weight_vol:.2%}) "
    f"for not that much more return, since it holds full weight in every "
    f"high-volatility equity sector rather than balancing them against "
    f"bonds and cash.\n"
)
print(
    "1. The policy-constrained max Sharpe portfolio is the practical "
    "recommendation - it respects realistic institutional limits (minimum "
    "bond holdings, maximum single-asset and sector exposure) while giving "
    "up almost nothing in risk-adjusted return versus the theoretical "
    "unconstrained optimum.\n"
    "2. The minimum equity policy floor (30%) is doing real work here - "
    "without it, the optimizer would hold less equity than most investment "
    "policies allow, since equities look less attractive relative to bonds "
    "and gold under these risk/return assumptions.\n"
    "3. Naive equal-weighting is clearly worse on a risk-adjusted basis than "
    "either optimized portfolio - a reminder that diversification alone "
    "isn't the same thing as optimization.\n"
    "4. This only reflects the return/risk assumptions used as inputs - "
    "changing those assumptions (see Limitations in the README) would move "
    "every result here, which is the single biggest real-world caveat of "
    "mean-variance optimization."
)

print(f"\n3 charts saved to {OUTPUT_DIR}/")
