"""
Plot group size sweep results from CSV.
Shows survival rate vs group size for each crowd size + overall.

Usage:
    python -m experiments.plot_group_sweep
"""

import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')
except ImportError:
    print("matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)


def load_results(path="results/sweep_group_size.csv"):
    with open(path) as f:
        return list(csv.DictReader(f))


def compute_stats(results, crowd_size=None):
    """Compute mean survival rate per group size."""
    from collections import defaultdict
    groups = defaultdict(list)

    for r in results:
        if crowd_size is not None and int(r["n_civilians"]) != crowd_size:
            continue
        gs = int(r["group_size"])
        groups[gs].append(float(r["survival_rate"]))

    sizes = sorted(groups.keys())
    means = [sum(groups[gs]) / len(groups[gs]) * 100 for gs in sizes]
    stds = []
    for gs in sizes:
        vals = [v * 100 for v in groups[gs]]
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        stds.append(std)

    return sizes, means, stds


def main():
    results = load_results()
    crowd_sizes = [50, 100, 150]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Sweeper Group Size Parameter Sweep", fontsize=14, fontweight='bold')

    colors = ['#2196F3', '#FF9800', '#4CAF50']

    # Individual crowd size plots
    for idx, nc in enumerate(crowd_sizes):
        ax = axes[idx // 2][idx % 2]
        sizes, means, stds = compute_stats(results, crowd_size=nc)

        ax.plot(sizes, means, 'o-', color=colors[idx], linewidth=2, markersize=4)
        ax.fill_between(sizes,
                         [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)],
                         alpha=0.2, color=colors[idx])

        # Mark the optimal
        best_idx = means.index(max(means))
        ax.plot(sizes[best_idx], means[best_idx], '*', color='red',
                markersize=15, zorder=5)
        ax.annotate(f'Best: {sizes[best_idx]} ({means[best_idx]:.1f}%)',
                     xy=(sizes[best_idx], means[best_idx]),
                     xytext=(sizes[best_idx] + 2, means[best_idx] + 3),
                     fontsize=9, color='red')

        ax.set_title(f'Crowd Size: {nc} civilians', fontsize=12)
        ax.set_xlabel('Group Size')
        ax.set_ylabel('Survival Rate (%)')
        ax.set_xlim(0, 26)
        ax.set_ylim(45, 85)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=means[best_idx], color='red', linestyle='--', alpha=0.3)

    # Overall plot (bottom-right)
    ax = axes[1][1]
    for idx, nc in enumerate(crowd_sizes):
        sizes, means, _ = compute_stats(results, crowd_size=nc)
        ax.plot(sizes, means, 'o-', color=colors[idx], linewidth=2,
                markersize=3, label=f'{nc} civilians')

    # Overall combined
    sizes, means, stds = compute_stats(results, crowd_size=None)
    ax.plot(sizes, means, 'o-', color='black', linewidth=3,
            markersize=4, label='Overall')

    best_idx = means.index(max(means))
    ax.plot(sizes[best_idx], means[best_idx], '*', color='red',
            markersize=15, zorder=5)

    ax.set_title('All Crowd Sizes Combined', fontsize=12)
    ax.set_xlabel('Group Size')
    ax.set_ylabel('Survival Rate (%)')
    ax.set_xlim(0, 26)
    ax.set_ylim(45, 85)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Add plateau annotation
    ax.axhspan(68, 72, alpha=0.1, color='green')
    ax.annotate('Plateau region (~12-25)',
                 xy=(18, 69), fontsize=9, color='green',
                 style='italic')

    plt.tight_layout()
    plt.savefig('results/sweep_group_size.png', dpi=150, bbox_inches='tight')
    print("Saved to results/sweep_group_size.png")
    plt.show()


if __name__ == "__main__":
    main()
