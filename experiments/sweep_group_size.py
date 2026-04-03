"""
Parameter sweep: Sweeper group size optimization.

Tests group sizes 1-25 across all 6 maps with 10 seeds each.
Finds the most stable and optimal group size.

Usage:
    python -m experiments.sweep_group_size
"""

import os
import sys
import time
import csv
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SimConfig
from evacsim.simulation import Simulation
from evacsim.guide import SweeperGuide


MAPS = [
    "maps/bottleneck_A.txt",
    "maps/bottleneck_B.txt",
    "maps/bottleneck_C.txt",
    "maps/open_plan_A.txt",
    "maps/open_plan_B.txt",
    "maps/open_plan_C.txt",
]

GROUP_SIZES = list(range(1, 26))  # 1 to 25
N_SEEDS = 10
CROWD_SIZES = [50, 100, 150]


def run_single_sweep(args):
    """Run one simulation with a specific group size and crowd size."""
    map_file, group_size, n_civilians, seed = args
    try:
        config = SimConfig(
            map_file=map_file,
            guide_policy="sweeper",
            n_civilians=n_civilians,
            n_guides=4,
            seed=seed,
            max_ticks=3000,
        )
        sim = Simulation(config)

        # Override group size for all sweeper guides
        for g in sim.guides:
            if isinstance(g, SweeperGuide):
                g._max_group = group_size

        result = sim.run()
        return {
            "map": map_file,
            "group_size": group_size,
            "n_civilians": n_civilians,
            "seed": seed,
            "TV": result["TV"],
            "TM": result["TM"],
            "survival_rate": result["survival_rate"],
            "MT": result["MT"],
            "MP": result["MP"],
        }
    except Exception as e:
        return {
            "map": map_file,
            "group_size": group_size,
            "n_civilians": n_civilians,
            "seed": seed,
            "TV": 0, "TM": n_civilians,
            "survival_rate": 0, "MT": 0, "MP": 0,
            "error": str(e),
        }


def main():
    # Build all job arguments
    jobs = []
    for m in MAPS:
        for gs in GROUP_SIZES:
            for nc in CROWD_SIZES:
                for seed_idx in range(N_SEEDS):
                    seed = seed_idx * 111 + gs * 7 + nc  # Unique seed per combo
                    jobs.append((m, gs, nc, seed))

    total = len(jobs)
    n_workers = max(1, cpu_count() - 1)

    print(f"Sweeper Group Size Sweep")
    print(f"========================")
    print(f"Group sizes: 1-25")
    print(f"Crowd sizes: {CROWD_SIZES}")
    print(f"Maps: {len(MAPS)}")
    print(f"Seeds per combo: {N_SEEDS}")
    print(f"Total runs: {total}")
    print(f"Workers: {n_workers}")
    print()

    start = time.time()
    results = []
    completed = 0

    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(run_single_sweep, jobs, chunksize=4):
            results.append(result)
            completed += 1
            if completed % 100 == 0 or completed == total:
                elapsed = time.time() - start
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{total}] {elapsed:.0f}s elapsed, {rate:.1f} runs/s, ETA: {eta:.0f}s")

    elapsed = time.time() - start
    print(f"\nDone! {total} runs in {elapsed:.1f}s ({total/elapsed:.1f} runs/s)")

    # Save raw results
    os.makedirs("results", exist_ok=True)
    output_file = "results/sweep_group_size.csv"
    fieldnames = ["map", "group_size", "n_civilians", "seed", "TV", "TM", "survival_rate", "MT", "MP"]
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Raw results saved to {output_file}")

    # Analyze per crowd size
    for nc in CROWD_SIZES:
        print(f"\n{'='*60}")
        print(f"  Crowd size: {nc} civilians")
        print(f"{'='*60}")
        print(f"{'GS':>3} | {'Mean TV':>8} | {'Mean Surv%':>10} | {'Std Surv%':>10} | {'Score':>8}")
        print("-" * 55)

        best_score = -1
        best_gs = -1

        for gs in GROUP_SIZES:
            gs_results = [r for r in results
                          if r["group_size"] == gs and r["n_civilians"] == nc]
            if not gs_results:
                continue

            survival_rates = [r["survival_rate"] for r in gs_results]
            mean_tv = sum(r["TV"] for r in gs_results) / len(gs_results)
            mean_sr = sum(survival_rates) / len(survival_rates)
            std_sr = (sum((s - mean_sr) ** 2 for s in survival_rates) / len(survival_rates)) ** 0.5

            score = mean_sr - std_sr * 0.5

            if score > best_score:
                best_score = score
                best_gs = gs

            print(f"{gs:>3} | {mean_tv:>8.1f} | {mean_sr*100:>9.1f}% | {std_sr*100:>9.1f}% | {score:>8.3f}")

        print(f"\n  Optimal for {nc} civilians: group_size={best_gs} (score: {best_score:.3f})")

    # Overall analysis: best across ALL crowd sizes
    print(f"\n{'='*60}")
    print(f"  OVERALL (all crowd sizes combined)")
    print(f"{'='*60}")
    print(f"{'GS':>3} | {'Mean Surv%':>10} | {'Std Surv%':>10} | {'Score':>8}")
    print("-" * 45)

    overall_best_score = -1
    overall_best_gs = -1

    for gs in GROUP_SIZES:
        gs_results = [r for r in results if r["group_size"] == gs]
        if not gs_results:
            continue
        survival_rates = [r["survival_rate"] for r in gs_results]
        mean_sr = sum(survival_rates) / len(survival_rates)
        std_sr = (sum((s - mean_sr) ** 2 for s in survival_rates) / len(survival_rates)) ** 0.5
        score = mean_sr - std_sr * 0.5

        if score > overall_best_score:
            overall_best_score = score
            overall_best_gs = gs

        print(f"{gs:>3} | {mean_sr*100:>9.1f}% | {std_sr*100:>9.1f}% | {score:>8.3f}")

    print(f"\n  OVERALL OPTIMAL: group_size={overall_best_gs} (score: {overall_best_score:.3f})")
    print(f"  Score = mean_survival - 0.5 * std_survival")


if __name__ == "__main__":
    main()
