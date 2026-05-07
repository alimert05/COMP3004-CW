"""Statistical analysis script for the evacuation factorial experiment.

Reads results/experiment_results.csv and produces:
    - Descriptive statistics at four aggregation levels (per policy,
      policy by crowd, policy by map type, and policy by individual
      map for the heatmap).
    - One row per unique condition (480 conditions in the full design).
    - One-way and two-way ANOVA with effect sizes.
    - Tukey HSD pairwise comparisons with Cohen's d.
    - Cumulative evacuation curves per policy.
    - Box plots, line plots, heatmap, and bar charts.

Metrics analysed:
    survival_rate     TV / (TV + TM), the primary outcome.
    MP                Mean potency of survivors at escape.
    MT                Mean evacuation time.
    guides_survived   Number of guides alive or successfully escaped.

Usage:
    python -m experiments.analysis
    python -m experiments.analysis --input results/experiment_results.csv
    python -m experiments.analysis --output results/analysis
"""

import os
import sys
import argparse
from pathlib import Path

# -- Required dependencies --
try:
    import pandas as pd
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend; works headless.
    import matplotlib.pyplot as plt
    from scipy import stats
except ImportError as e:
    print(f"ERROR: Missing required library: {e.name}")
    print("Install with:  pip install pandas numpy matplotlib scipy")
    sys.exit(1)

# Optional: statsmodels for Tukey HSD and two-way ANOVA
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print(
        "WARNING: statsmodels not installed. Tukey HSD and two-way ANOVA will "
        "be skipped.\n         Install with:  pip install statsmodels"
    )

# seaborn for nicer plot styling
try:
    import seaborn as sns

    HAS_SEABORN = True
    sns.set_theme(style="whitegrid", context="talk")
except ImportError:
    HAS_SEABORN = False


# Constants
POLICY_ORDER = ["none", "beacon", "sweeper", "router"]
POLICY_COLORS = {
    "none": "#8E8E8E",
    "beacon": "#F2C14E",
    "sweeper": "#5B8FF9",
    "router": "#E57373",
}
BIN_EDGES = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
BIN_COLUMNS = [f"escaped_{lo}_{hi}" for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:])]

# 1. Data loading and preprocessing


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the experiment CSV and augment it with derived columns."""
    df = pd.read_csv(csv_path)

    # Derive map_type from the file name.
    df["map_type"] = df["map"].apply(
        lambda p: (
            "bottleneck"
            if "bottleneck" in p
            else ("open_plan" if "open_plan" in p else "other")
        )
    )

    # Derive a clean map name (strip path and extension).
    df["map_name"] = df["map"].apply(lambda p: os.path.splitext(os.path.basename(p))[0])

    # Order the policy column so plots and tables follow a fixed sequence.
    df["policy"] = pd.Categorical(df["policy"], categories=POLICY_ORDER, ordered=True)

    # Data-integrity check: drop any errored runs.
    n_errors = 0
    if "error" in df.columns:
        n_errors = df["error"].notna().sum()
        if n_errors:
            print(f"WARNING: {n_errors} runs had errors. Excluding them.")
            df = df[df["error"].isna()]

    print(f"Loaded {len(df)} runs (skipped {n_errors} errored).")
    print(f"  Maps:      {sorted(df['map_name'].unique())}")
    print(f"  Policies:  {list(df['policy'].cat.categories)}")
    print(f"  Crowd:     {sorted(df['n_civilians'].unique())}")
    print(f"  Fire pos:  {sorted(df['fire_position'].unique())}")
    print(
        f"  Trials:    {df.groupby(['map', 'fire_position', 'policy', 'n_civilians'], observed=True).size().describe()['mean']:.0f} (mean)"
    )

    return df


# 2. Descriptive statistics


def _ci95(series: pd.Series) -> float:
    """Half-width of the 95 percent confidence interval for a sample."""
    n = series.count()
    if n < 2:
        return float("nan")
    return 1.96 * series.std(ddof=1) / np.sqrt(n)


def descriptive_stats(df: pd.DataFrame) -> dict:
    """Compute summary tables at several aggregation levels."""
    metrics = ["survival_rate", "TV", "TM", "MP", "MT", "guides_survived"]
    tables = {}

    # Per-policy overall.
    overall = df.groupby("policy", observed=True)[metrics].agg(
        ["mean", "std", _ci95, "count"]
    )
    tables["per_policy"] = overall

    # Per policy and crowd size.
    per_policy_crowd = df.groupby(["policy", "n_civilians"], observed=True)[
        metrics
    ].agg(["mean", "std", _ci95])
    tables["per_policy_crowd"] = per_policy_crowd

    # Per policy and map type.
    per_policy_maptype = df.groupby(["policy", "map_type"], observed=True)[metrics].agg(
        ["mean", "std", _ci95]
    )
    tables["per_policy_maptype"] = per_policy_maptype

    # Per policy and individual map (used for the heatmap).
    per_policy_map = (
        df.groupby(["policy", "map_name"], observed=True)["survival_rate"]
        .mean()
        .unstack(level="map_name")
    )
    tables["per_policy_map_heatmap"] = per_policy_map

    return tables


def per_condition_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Produce one row per unique (map, fire, policy, crowd) condition.

    Each row reports n_trials and the mean, std, min and max of every
    primary metric.
    """
    group_cols = ["map_name", "fire_position", "policy", "n_civilians"]
    metrics = [
        "TV",
        "TM",
        "MP",
        "MT",
        "survival_rate",
        "evacuation_time",
        "guides_survived",
    ]

    agg_funcs = {}
    for m in metrics:
        agg_funcs[m] = ["mean", "std", "min", "max"]

    grouped = (
        df.groupby(group_cols, observed=True)
        .agg(
            n_trials=("TV", "count"),
            **{
                f"{m}_{stat}": (m, stat)
                for m in metrics
                for stat in ["mean", "std", "min", "max"]
            },
        )
        .reset_index()
    )

    return grouped


# 3. Inferential statistics


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d effect size for two independent samples (pooled SD)."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    pooled_sd = np.sqrt(
        ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
    )
    if pooled_sd == 0:
        return float("nan")
    return (x.mean() - y.mean()) / pooled_sd


def _interpret_d(d: float) -> str:
    """Return a verbal magnitude label for Cohen's d."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def _interpret_eta2(eta2: float) -> str:
    """Return a verbal magnitude label for eta-squared."""
    if eta2 < 0.01:
        return "negligible"
    if eta2 < 0.06:
        return "small"
    if eta2 < 0.14:
        return "medium"
    return "large"


def one_way_anova(df: pd.DataFrame, metric: str, factor: str = "policy") -> dict:
    """One-way ANOVA of a metric against a factor.

    Returns a dictionary with F, p, eta-squared, a verbal magnitude
    label, and per-group descriptive statistics.
    """
    groups = [g[metric].dropna().values for _, g in df.groupby(factor, observed=True)]
    if len(groups) < 2 or any(len(g) < 2 for g in groups):
        return {"error": "Not enough data for ANOVA"}

    f_stat, p_val = stats.f_oneway(*groups)

    # Eta-squared = SS_between / SS_total.
    all_values = np.concatenate(groups)
    grand_mean = all_values.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((all_values - grand_mean) ** 2).sum()
    eta2 = ss_between / ss_total if ss_total > 0 else 0.0

    return {
        "metric": metric,
        "factor": factor,
        "F": f_stat,
        "p": p_val,
        "eta2": eta2,
        "interpretation": _interpret_eta2(eta2),
        "significant": p_val < 0.05,
        "groups": {
            str(name): {"n": len(g), "mean": g.mean(), "std": g.std(ddof=1)}
            for (name, _), g in zip(df.groupby(factor, observed=True), groups)
        },
    }


def two_way_anova(df: pd.DataFrame, metric: str, factor1: str, factor2: str) -> str:
    """Two-way ANOVA with interaction term. Returns the formatted text table."""
    if not HAS_STATSMODELS:
        return "statsmodels not installed; two-way ANOVA skipped."
    try:
        formula = f"{metric} ~ C({factor1}) + C({factor2}) + C({factor1}):C({factor2})"
        model = ols(formula, data=df).fit()
        table = sm.stats.anova_lm(model, typ=2)
        return str(table)
    except Exception as e:
        return f"ANOVA failed: {e}"


def tukey_posthoc(
    df: pd.DataFrame, metric: str, factor: str = "policy"
) -> pd.DataFrame:
    """Tukey HSD pairwise comparisons with Cohen's d added per pair."""
    if not HAS_STATSMODELS:
        return pd.DataFrame({"error": ["statsmodels not installed"]})

    try:
        sub = df[[metric, factor]].dropna()
        result = pairwise_tukeyhsd(sub[metric], sub[factor].astype(str))
        rows = []
        for r in result.summary().data[1:]:
            g1, g2, meandiff, p_adj, lower, upper, reject = r
            x = sub[sub[factor].astype(str) == g1][metric].values
            y = sub[sub[factor].astype(str) == g2][metric].values
            d = _cohens_d(x, y)
            rows.append(
                {
                    "group1": g1,
                    "group2": g2,
                    "meandiff": meandiff,
                    "p_adj": p_adj,
                    "lower_ci": lower,
                    "upper_ci": upper,
                    "reject_H0": reject,
                    "cohens_d": d,
                    "d_magnitude": _interpret_d(d) if not np.isnan(d) else "n/a",
                }
            )
        return pd.DataFrame(rows)
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})


def per_map_paired_comparison(
    df: pd.DataFrame, policy_a: str, policy_b: str, metric: str = "survival_rate"
) -> pd.DataFrame:
    """Paired t-test comparing policy_a vs policy_b on each map separately.

    Uses the fully paired experimental design: within each map, the
    same (fire_position, n_civilians, trial) cell supplies one value
    for policy_a and one for policy_b. Cohen's dz is the paired effect
    size (mean_diff / SD_diff). Returns one row per map with mean
    difference, 95 percent CI, t-statistic, p-value, dz, and a
    magnitude label.
    """
    keys = ["fire_position", "n_civilians", "trial"]
    rows = []
    for map_name in sorted(df["map_name"].unique()):
        sub = df[df["map_name"] == map_name]
        a = sub[sub["policy"] == policy_a].set_index(keys)[metric].sort_index()
        b = sub[sub["policy"] == policy_b].set_index(keys)[metric].sort_index()
        # Align on the shared keys (paired design guarantees identical sets).
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        n = len(a)
        if n < 2:
            continue
        diff = a.values - b.values
        sd_diff = diff.std(ddof=1)
        if sd_diff == 0:
            continue
        t_stat, p_val = stats.ttest_rel(a.values, b.values)
        dz = diff.mean() / sd_diff
        ci_half = 1.96 * sd_diff / np.sqrt(n)
        rows.append(
            {
                "map": map_name,
                "n_pairs": n,
                "mean_diff": round(diff.mean(), 4),
                "ci_low": round(diff.mean() - ci_half, 4),
                "ci_high": round(diff.mean() + ci_half, 4),
                "t": round(t_stat, 3),
                "p": p_val,
                "cohens_dz": round(dz, 3),
                "magnitude": _interpret_d(dz),
            }
        )
    return pd.DataFrame(rows)


def two_way_anova_with_partial_eta(
    df: pd.DataFrame, metric: str, factor1: str, factor2: str
) -> pd.DataFrame:
    """Two-way ANOVA with partial eta-squared added for each effect.

    partial_eta2 = SS_effect / (SS_effect + SS_residual): the
    proportion of variance explained by an effect after accounting for
    the other terms in the model. Reported alongside F and p for every
    main effect and the interaction term.
    """
    if not HAS_STATSMODELS:
        return pd.DataFrame({"error": ["statsmodels not installed"]})
    try:
        formula = (
            f"{metric} ~ C({factor1}) + C({factor2}) " f"+ C({factor1}):C({factor2})"
        )
        model = ols(formula, data=df).fit()
        table = sm.stats.anova_lm(model, typ=2).copy()
        ss_residual = table.loc["Residual", "sum_sq"]
        table["partial_eta2"] = table["sum_sq"] / (table["sum_sq"] + ss_residual)
        table.at["Residual", "partial_eta2"] = float("nan")
        table["magnitude"] = table["partial_eta2"].apply(
            lambda x: _interpret_eta2(x) if not pd.isna(x) else "n/a"
        )
        return table
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})


def interaction_decomposition(
    df: pd.DataFrame, metric: str, maps_to_exclude: list
) -> pd.DataFrame:
    """Decompose the policy by map-type interaction by excluded maps.

    Re-runs the two-way ANOVA (policy by map_type) on the full dataset
    and on a subset with the specified maps removed. If the
    interaction's partial eta-squared drops substantially in the
    subset, those maps contribute disproportionately to the
    interaction effect.

    Used to verify the claim that a small number of outlier maps drive
    most of the interaction, rather than the interaction being a
    distributed pattern across all maps.
    """
    if not HAS_STATSMODELS:
        return pd.DataFrame({"error": ["statsmodels not installed"]})
    try:
        # Full ANOVA (all maps).
        full_table = two_way_anova_with_partial_eta(df, metric, "policy", "map_type")
        full_row = full_table.loc["C(policy):C(map_type)"]

        # Filtered ANOVA (excluding specified maps).
        df_filtered = df[~df["map_name"].isin(maps_to_exclude)].copy()
        filt_table = two_way_anova_with_partial_eta(
            df_filtered, metric, "policy", "map_type"
        )
        filt_row = filt_table.loc["C(policy):C(map_type)"]

        return pd.DataFrame(
            [
                {
                    "scenario": "full (all 8 maps)",
                    "n_runs": len(df),
                    "F": float(full_row["F"]),
                    "p": float(full_row["PR(>F)"]),
                    "partial_eta2": float(full_row["partial_eta2"]),
                    "magnitude": full_row["magnitude"],
                },
                {
                    "scenario": (
                        f"excluding {', '.join(maps_to_exclude)} "
                        f"({len(df) - len(df_filtered)} runs dropped)"
                    ),
                    "n_runs": len(df_filtered),
                    "F": float(filt_row["F"]),
                    "p": float(filt_row["PR(>F)"]),
                    "partial_eta2": float(filt_row["partial_eta2"]),
                    "magnitude": filt_row["magnitude"],
                },
            ]
        )
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})


# 4. Visualisations


def _setup_fig(figsize=(10, 6)):
    """Create a fresh figure-and-axes pair with a default size."""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def plot_boxplot_policy(
    df: pd.DataFrame, metric: str, outpath: Path, ylabel: str = None
):
    """Box plot of one metric grouped by policy."""
    fig, ax = _setup_fig()
    data = [df[df["policy"] == p][metric].dropna().values for p in POLICY_ORDER]
    bp = ax.boxplot(
        data,
        tick_labels=POLICY_ORDER,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, p in zip(bp["boxes"], POLICY_ORDER):
        patch.set_facecolor(POLICY_COLORS[p])
        patch.set_alpha(0.7)
    ax.set_xlabel("Policy")
    ax.set_ylabel(ylabel or metric)
    ax.set_title(f"{ylabel or metric} distribution by policy")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_boxplot_by_map_type(
    df: pd.DataFrame, metric: str, outpath: Path, ylabel: str = None
):
    """Grouped box plot: policy by map_type."""
    fig, ax = _setup_fig(figsize=(12, 6))
    positions = []
    data = []
    labels = []
    x = 0
    for mt in ["bottleneck", "open_plan"]:
        for p in POLICY_ORDER:
            values = (
                df[(df["policy"] == p) & (df["map_type"] == mt)][metric].dropna().values
            )
            if len(values):
                positions.append(x)
                data.append(values)
                labels.append(f"{p}\n({mt.replace('_',' ')})")
            x += 1
        x += 1  # Gap between map-type groups.

    bp = ax.boxplot(
        data,
        positions=positions,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, lab in zip(bp["boxes"], labels):
        pol = lab.split("\n")[0]
        patch.set_facecolor(POLICY_COLORS[pol])
        patch.set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0, fontsize=9)
    ax.set_ylabel(ylabel or metric)
    ax.set_title(f"{ylabel or metric} by policy and map type")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_line_crowd_vs_survival(df: pd.DataFrame, outpath: Path):
    """Line chart of mean survival_rate vs crowd size, one line per policy."""
    fig, ax = _setup_fig()
    for p in POLICY_ORDER:
        sub = (
            df[df["policy"] == p]
            .groupby("n_civilians")["survival_rate"]
            .agg(["mean", _ci95])
        )
        ax.errorbar(
            sub.index,
            sub["mean"],
            yerr=sub["_ci95"],
            marker="o",
            capsize=4,
            linewidth=2,
            label=p,
            color=POLICY_COLORS[p],
        )
    ax.set_xlabel("Crowd size (n_civilians)")
    ax.set_ylabel("Mean survival rate")
    ax.set_title("Survival rate by crowd size, with 95 percent CI")
    ax.legend(title="Policy")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_heatmap_map_policy(tables: dict, outpath: Path):
    """Heatmap of mean survival_rate across the 8 maps and 4 policies."""
    data = tables["per_policy_map_heatmap"]  # rows: policies; cols: maps.
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(data.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(list(data.index))

    # Write the value into each cell.
    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            val = data.values[i, j]
            txt_color = "white" if val < 0.35 or val > 0.75 else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=txt_color,
                fontsize=9,
            )

    ax.set_title("Mean survival rate by map and policy")
    plt.colorbar(im, ax=ax, label="Survival rate")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_bar_evac_time(df: pd.DataFrame, outpath: Path):
    """Bar chart of mean MT (evacuation time) by policy with 95 percent CI."""
    fig, ax = _setup_fig()
    means, cis, names = [], [], []
    for p in POLICY_ORDER:
        sub = df[df["policy"] == p]["MT"].dropna()
        if not len(sub):
            continue
        means.append(sub.mean())
        cis.append(_ci95(sub))
        names.append(p)
    colors = [POLICY_COLORS[n] for n in names]
    ax.bar(names, means, yerr=cis, color=colors, alpha=0.75, capsize=5)
    ax.set_xlabel("Policy")
    ax.set_ylabel("Mean evacuation time (ticks)")
    ax.set_title("Mean evacuation time of survivors by policy, with 95 percent CI")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_evacuation_curves(df: pd.DataFrame, outpath: Path, by_map_type: bool = False):
    """Cumulative percent-escaped curves over time, per policy.

    Uses the binned escape counts per run. Aggregates across trials
    for each policy and bin, producing a mean cumulative curve.
    """
    if by_map_type:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        map_types = ["bottleneck", "open_plan"]
    else:
        fig, axes = plt.subplots(figsize=(10, 6))
        axes = [axes]
        map_types = [None]

    for ax_idx, mt in enumerate(map_types):
        ax = axes[ax_idx]
        for p in POLICY_ORDER:
            sub = df[df["policy"] == p]
            if mt is not None:
                sub = sub[sub["map_type"] == mt]
            if not len(sub):
                continue

            # Mean cumulative escape count per bin, divided by the
            # mean civilian count for the policy (slice).
            mean_n = sub["n_civilians"].mean()
            cum_means = []
            tick_points = [BIN_EDGES[0]]  # Start at 0.

            running = 0
            for col in BIN_COLUMNS:
                mean_in_bin = sub[col].mean() if col in sub.columns else 0
                running += mean_in_bin
                cum_means.append(running / mean_n if mean_n else 0)

            # Step function with cumulative percent at the end of each bin.
            xs = BIN_EDGES[1:]
            ys = [c * 100 for c in cum_means]
            # Prepend (0, 0) so the curve starts at the origin.
            xs = [0] + xs
            ys = [0] + ys
            ax.plot(xs, ys, marker="o", linewidth=2.2, label=p, color=POLICY_COLORS[p])

        ax.set_xlabel("Simulation tick")
        if ax_idx == 0:
            ax.set_ylabel("Cumulative percent escaped")
        title = f"Evacuation curves: {mt}" if mt else "Evacuation curves (all maps)"
        ax.set_title(title)
        ax.set_ylim(0, 105)
        ax.set_xlim(0, 6000)
        ax.legend(title="Policy", loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# 5. Plain-text summary output


def build_summary_text(
    df: pd.DataFrame, tables: dict, anovas: dict, tukey: dict
) -> str:
    """Build a human-readable summary suitable for copy into the report.

    Uses ASCII-only characters so it prints correctly on Windows
    consoles using cp1252 encoding.
    """
    out = []
    out.append("=" * 70)
    out.append("EVACUATION EXPERIMENT ANALYSIS SUMMARY")
    out.append(f"Total runs analysed: {len(df)}")
    out.append(
        f"Maps: {df['map_name'].nunique()} | "
        f"Policies: {df['policy'].nunique()} | "
        f"Crowd sizes: {df['n_civilians'].nunique()} | "
        f"Fire positions: {df['fire_position'].nunique()}"
    )
    out.append("=" * 70 + "\n")

    # Per-policy overall.
    out.append("-- SURVIVAL RATE BY POLICY (overall) --")
    per_pol = tables["per_policy"]["survival_rate"]
    for p in POLICY_ORDER:
        if p not in per_pol.index:
            continue
        row = per_pol.loc[p]
        out.append(
            f"  {p:<8}  mean={row['mean']:.3f}  SD={row['std']:.3f}  "
            f"95% CI+/-{row['_ci95']:.3f}  n={int(row['count'])}"
        )
    out.append("")

    # ANOVA summary.
    out.append("-- ONE-WAY ANOVA RESULTS --")
    for metric, anova in anovas.items():
        if "error" in anova:
            out.append(f"  {metric}: {anova['error']}")
            continue
        sig = "SIGNIFICANT" if anova["significant"] else "not significant"
        out.append(
            f"  {metric:<18}  F={anova['F']:7.2f}  p={anova['p']:.4g}  "
            f"eta2={anova['eta2']:.3f} ({anova['interpretation']})  -> {sig}"
        )
    out.append("")

    # Tukey HSD pairwise.
    out.append("-- TUKEY HSD PAIRWISE COMPARISONS (survival_rate) --")
    tk = tukey.get("survival_rate")
    if isinstance(tk, pd.DataFrame) and "error" not in tk.columns:
        for _, r in tk.iterrows():
            flag = (
                "***"
                if r["p_adj"] < 0.001
                else "**" if r["p_adj"] < 0.01 else "*" if r["p_adj"] < 0.05 else " "
            )
            d_val = r["cohens_d"]
            out.append(
                f"  {r['group1']:<8} vs {r['group2']:<8}  "
                f"diff={r['meandiff']:+.3f}  p={r['p_adj']:.4g} {flag}  "
                f"d={d_val:+.2f} ({r['d_magnitude']})"
            )
    else:
        out.append("  (Tukey HSD unavailable; statsmodels missing?)")
    out.append("")

    # Per-map-type breakdown.
    out.append("-- SURVIVAL BY MAP TYPE x POLICY --")
    mt_stats = tables["per_policy_maptype"]["survival_rate"]
    for mt in ["bottleneck", "open_plan"]:
        out.append(f"  {mt}:")
        for p in POLICY_ORDER:
            if (p, mt) not in mt_stats.index:
                continue
            row = mt_stats.loc[(p, mt)]
            out.append(
                f"    {p:<8}  mean={row['mean']:.3f}  "
                f"SD={row['std']:.3f}  95% CI+/-{row['_ci95']:.3f}"
            )
    out.append("")

    # Crowd-size breakdown.
    out.append("-- SURVIVAL BY CROWD SIZE x POLICY --")
    cz_stats = tables["per_policy_crowd"]["survival_rate"]
    for p in POLICY_ORDER:
        out.append(f"  {p}:")
        for cz in sorted(df["n_civilians"].unique()):
            if (p, cz) not in cz_stats.index:
                continue
            row = cz_stats.loc[(p, cz)]
            out.append(
                f"    n={cz:<4}  mean={row['mean']:.3f}  "
                f"SD={row['std']:.3f}  95% CI+/-{row['_ci95']:.3f}"
            )
    out.append("")

    # Best and worst policy per map.
    out.append("-- BEST POLICY PER MAP (by mean survival) --")
    heat = tables["per_policy_map_heatmap"]
    for m in heat.columns:
        best = heat[m].idxmax()
        worst = heat[m].idxmin()
        out.append(
            f"  {m}: best={best} ({heat[m][best]:.3f}), "
            f"worst={worst} ({heat[m][worst]:.3f})"
        )
    out.append("")

    out.append("=" * 70)
    out.append("Symbols: * p<0.05   ** p<0.01   *** p<0.001")
    out.append("Cohen's d magnitudes: 0.2=small, 0.5=medium, 0.8=large")
    out.append("eta-squared magnitudes: 0.01=small, 0.06=medium, 0.14=large")
    out.append("=" * 70)

    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(
        description="Analyse evacuation experiment results"
    )
    parser.add_argument(
        "--input",
        default="results/experiment_results.csv",
        help="Path to the input CSV",
    )
    parser.add_argument("--output", default="results/analysis", help="Output directory")
    args = parser.parse_args()

    # Add project root to the import path.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    out_dir = Path(args.output)
    figs_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data.
    print(f"\nLoading: {args.input}")
    df = load_data(args.input)

    # 2. Descriptive statistics.
    print("\nComputing descriptive statistics...")
    tables = descriptive_stats(df)
    for name, t in tables.items():
        t.to_csv(out_dir / f"{name}.csv")
        print(f"  Saved {name}.csv")

    # Per-condition summary.
    print("\nBuilding per-condition summary (one row per unique condition)...")
    per_cond = per_condition_summary(df)
    per_cond.to_csv(out_dir / "per_condition_summary.csv", index=False)
    print(f"  Saved per_condition_summary.csv ({len(per_cond)} conditions)")

    # 3. Inferential statistics.
    print("\nRunning ANOVA...")
    anovas = {}
    for metric in ["survival_rate", "MP", "MT", "guides_survived"]:
        if metric in df.columns:
            anovas[metric] = one_way_anova(df, metric, factor="policy")

    # Two-way ANOVA tables (text form).
    anova_2way_text = []
    for m in ["survival_rate"]:
        txt = two_way_anova(df, m, "policy", "map_type")
        anova_2way_text.append(f"-- {m} ~ policy x map_type --\n{txt}\n")
        txt = two_way_anova(df, m, "policy", "n_civilians")
        anova_2way_text.append(f"-- {m} ~ policy x n_civilians --\n{txt}\n")

    with open(out_dir / "anova_results.txt", "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("ONE-WAY ANOVA RESULTS\n")
        f.write("=" * 70 + "\n\n")
        for metric, anova in anovas.items():
            f.write(f"Metric: {metric}\n")
            if "error" in anova:
                f.write(f"  {anova['error']}\n\n")
                continue
            f.write(f"  F = {anova['F']:.4f}\n")
            f.write(f"  p = {anova['p']:.6g}\n")
            f.write(f"  eta2 = {anova['eta2']:.4f} ({anova['interpretation']})\n")
            f.write(f"  Significant at p<0.05: {anova['significant']}\n")
            f.write("  Group stats:\n")
            for name, s in anova["groups"].items():
                f.write(
                    f"    {name:<8}  n={s['n']:<5}  "
                    f"mean={s['mean']:.4f}  SD={s['std']:.4f}\n"
                )
            f.write("\n")
        f.write("\n" + "=" * 70 + "\n")
        f.write("TWO-WAY ANOVA RESULTS\n")
        f.write("=" * 70 + "\n\n")
        for block in anova_2way_text:
            f.write(block + "\n")
    print("  Saved anova_results.txt")

    # Tukey HSD post-hoc.
    print("\nRunning Tukey HSD post-hoc...")
    tukey = {}
    for metric in ["survival_rate", "MP", "MT"]:
        if metric in df.columns:
            tukey[metric] = tukey_posthoc(df, metric, factor="policy")
            fn = out_dir / f"tukey_{metric}.csv"
            tukey[metric].to_csv(fn, index=False)
            print(f"  Saved tukey_{metric}.csv")

    # Extended statistics: per-map paired comparison plus partial eta2.
    print("\nRunning extended statistics...")

    # Per-map Router-vs-Sweeper paired comparison: directly answers
    # the topology-dependent magnitude question.
    per_map_rs = per_map_paired_comparison(
        df, "router", "sweeper", metric="survival_rate"
    )
    if not per_map_rs.empty and "error" not in per_map_rs.columns:
        per_map_rs.to_csv(out_dir / "per_map_router_vs_sweeper_paired.csv", index=False)
        print("  Saved per_map_router_vs_sweeper_paired.csv")
    else:
        print("  (per-map paired comparison produced no rows)")

    # Partial eta2 for the three two-way ANOVAs. The policy by
    # fire_position model confirms fire_position is a nuisance factor
    # with a negligible interaction, justifying the decision to
    # collapse across it in the main analysis.
    anova_parts = []
    for f1, f2 in [
        ("policy", "map_type"),
        ("policy", "n_civilians"),
        ("policy", "fire_position"),
    ]:
        table = two_way_anova_with_partial_eta(df, "survival_rate", f1, f2)
        if not isinstance(table, pd.DataFrame) or "error" in table.columns:
            continue
        table_out = table.reset_index().rename(columns={"index": "term"})
        table_out.insert(0, "model", f"survival_rate ~ {f1} x {f2}")
        anova_parts.append(table_out)

    if anova_parts:
        combined = pd.concat(anova_parts, ignore_index=True)
        combined.to_csv(out_dir / "anova_partial_eta2.csv", index=False)
        print("  Saved anova_partial_eta2.csv")
    else:
        print("  (partial eta2 computation unavailable; statsmodels missing?)")

    # Interaction-driver decomposition: verify whether bottleneck_A and
    # bottleneck_D drive most of the policy by map-type interaction.
    # If the interaction's partial eta2 drops substantially when those
    # maps are excluded, the claim holds.
    decomp = interaction_decomposition(
        df,
        "survival_rate",
        maps_to_exclude=["bottleneck_A", "bottleneck_D"],
    )
    if isinstance(decomp, pd.DataFrame) and "error" not in decomp.columns:
        decomp.to_csv(out_dir / "interaction_decomposition.csv", index=False)
        print("  Saved interaction_decomposition.csv")
    else:
        print("  (interaction decomposition unavailable)")

    # 4. Visualisations.
    print("\nGenerating figures...")
    plot_boxplot_policy(
        df, "survival_rate", figs_dir / "01_boxplot_policy.png", ylabel="Survival rate"
    )
    print("  01_boxplot_policy.png")
    plot_boxplot_by_map_type(
        df,
        "survival_rate",
        figs_dir / "02_boxplot_by_map_type.png",
        ylabel="Survival rate",
    )
    print("  02_boxplot_by_map_type.png")
    plot_line_crowd_vs_survival(df, figs_dir / "03_line_crowd_vs_survival.png")
    print("  03_line_crowd_vs_survival.png")
    plot_heatmap_map_policy(tables, figs_dir / "04_heatmap_map_policy.png")
    print("  04_heatmap_map_policy.png")
    plot_bar_evac_time(df, figs_dir / "05_bar_evac_time.png")
    print("  05_bar_evac_time.png")

    # Evacuation curves require the bin columns to be present.
    if all(c in df.columns for c in BIN_COLUMNS):
        plot_evacuation_curves(
            df, figs_dir / "06_evacuation_curves.png", by_map_type=False
        )
        print("  06_evacuation_curves.png")
        plot_evacuation_curves(
            df, figs_dir / "07_evacuation_curves_by_map_type.png", by_map_type=True
        )
        print("  07_evacuation_curves_by_map_type.png")
    else:
        print(
            "  (Evacuation curves skipped; bin columns missing from CSV. "
            "Re-run experiments with the updated runner.)"
        )

    # 5. Plain-text summary.
    print("\nBuilding summary text...")
    summary = build_summary_text(df, tables, anovas, tukey)
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    print("  Saved summary.txt")

    # Echo to console (guard against Windows cp1252 encoding errors).
    try:
        print("\n" + summary)
    except UnicodeEncodeError:
        print("\n(Summary contains non-ASCII characters; see summary.txt)")

    print(f"\nDone! All outputs in: {out_dir}/\n")


if __name__ == "__main__":
    main()
