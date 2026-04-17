"""
Experiment runner — 14,400 headless simulation runs.

Factorial design:
  8 maps × 5 fire positions × 4 policies × 3 crowd sizes × 30 trials = 14,400 runs

Maps: 4 bottleneck (A/B/C/D) + 4 open-plan (A/B/C/D), all 36×80.
Crowd sizes: 50 (well-supplied), 100 (near McCormack 1:20 optimum), 150 (stressed).
Fire positions: center + 4 corners.
Policies: none (baseline), beacon (passive), sweeper (reactive), router (hybrid).

Uses Python multiprocessing to utilise all CPU cores.
Saves results to CSV for analysis.

Usage:
    python -m experiments.runner                  # Full 14,400 runs
    python -m experiments.runner --quick          # Quick test (60 runs)
    python -m experiments.runner --medium         # Medium test (600 runs)
    python -m experiments.runner --policy beacon  # Single policy only
    python -m experiments.runner --map maps/bottleneck_A.txt  # Single map
"""

import os
import sys
import csv
import time
import argparse
from multiprocessing import Pool, cpu_count
from itertools import product

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SimConfig
from evacsim.simulation import run_single


# ── Experimental design ──────────────────────────────────────────────

MAPS = [
    # Bottleneck maps (4): rooms with narrow doorways
    "maps/bottleneck_A.txt",   # Grid of small rooms, 1-cell doorways
    "maps/bottleneck_B.txt",   # Wider 3-cell corridors, asymmetric exits
    "maps/bottleneck_C.txt",   # Asymmetric — all 4 exits on right wall
    "maps/bottleneck_D.txt",   # Hospital corridor, 2-cell central aisle
    # Open-plan maps (4): large open spaces
    "maps/open_plan_A.txt",    # Open warehouse with scattered pillars
    "maps/open_plan_B.txt",    # Cubicle partitions, lane-like
    "maps/open_plan_C.txt",    # Funnel — all exits clustered at bottom
    "maps/open_plan_D.txt",    # Two zones connected by narrow bridge
]

FIRE_POSITIONS = [
    "center",
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
]

POLICIES = ["none", "beacon", "sweeper", "router"]

# Updated from [20, 50, 100]: 20 was too sparse (policy differences compressed)
# New range: 50 (1:12.5 per guide) → 100 (near McCormack 1:20) → 150 (1:37.5 stressed)
CROWD_SIZES = [50, 100, 150]

N_TRIALS = 30


def build_configs(maps=None, fire_positions=None, policies=None,
                  crowd_sizes=None, n_trials=None):
    """Generate all SimConfig combinations."""
    maps = maps or MAPS
    fire_positions = fire_positions or FIRE_POSITIONS
    policies = policies or POLICIES
    crowd_sizes = crowd_sizes or CROWD_SIZES
    n_trials = n_trials or N_TRIALS

    configs = []
    for map_file, fire_pos, policy, n_civ in product(
            maps, fire_positions, policies, crowd_sizes):
        for trial in range(n_trials):
            seed = hash((map_file, fire_pos, policy, n_civ, trial)) % (2**31)
            config = SimConfig(
                map_file=map_file,
                fire_position=fire_pos,
                guide_policy=policy,
                n_civilians=n_civ,
                seed=seed,
                max_ticks=3000,
            )
            # Store trial number for CSV
            config._trial = trial
            configs.append(config)

    return configs


def run_and_label(config):
    """Run one simulation, return result dict with trial info."""
    try:
        result = run_single(config)
        result["trial"] = getattr(config, "_trial", 0)
        return result
    except Exception as e:
        return {
            "map": config.map_file,
            "fire_position": config.fire_position,
            "policy": config.guide_policy,
            "n_civilians": config.n_civilians,
            "seed": config.seed,
            "trial": getattr(config, "_trial", 0),
            "error": str(e),
            "TV": 0, "TM": 0, "MP": 0, "MT": 0,
            "survival_rate": 0, "evacuation_time": 0,
            "still_alive": 0, "total_civilians": config.n_civilians,
            "n_guides": 0, "guides_survived": 0,
            "escaped_0_500": 0, "escaped_500_1000": 0, "escaped_1000_1500": 0,
            "escaped_1500_2000": 0, "escaped_2000_2500": 0, "escaped_2500_3000": 0,
        }


def run_experiment(configs, output_file="results/experiment_results.csv",
                   n_workers=None):
    """Run all configs in parallel, save to CSV."""
    n_workers = n_workers or max(1, cpu_count() - 1)
    total = len(configs)

    print(f"Running {total} simulations on {n_workers} workers...")
    print(f"Output: {output_file}")
    print()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    start = time.time()
    results = []
    completed = 0

    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(run_and_label, configs, chunksize=4):
            results.append(result)
            completed += 1
            if completed % 50 == 0 or completed == total:
                elapsed = time.time() - start
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{total}] "
                      f"{elapsed:.0f}s elapsed, "
                      f"{rate:.1f} runs/s, "
                      f"ETA: {eta:.0f}s")

    elapsed = time.time() - start
    print(f"\nDone! {total} runs in {elapsed:.1f}s "
          f"({total/elapsed:.1f} runs/s)")

    # Save to CSV
    if results:
        fieldnames = [
            "map", "fire_position", "policy", "n_civilians", "trial", "seed",
            "TV", "TM", "MP", "MT", "survival_rate", "evacuation_time",
            "still_alive", "total_civilians", "n_guides", "guides_survived",
            # Evacuation-curve bins (count of civilians who escaped in each
            # 500-tick window, used to build cumulative escape curves)
            "escaped_0_500", "escaped_500_1000", "escaped_1000_1500",
            "escaped_1500_2000", "escaped_2000_2500", "escaped_2500_3000",
        ]
        # Add error field if any errors
        if any("error" in r for r in results):
            fieldnames.append("error")

        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                     extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

        print(f"Saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run evacuation experiments")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke: 1 map, 1 fire, 4 policies, 1 size, 1 trial = 4 runs")
    parser.add_argument("--quick", action="store_true",
                        help="Quick: 1 map, 1 fire, 4 policies, 3 sizes, 5 trials = 60 runs")
    parser.add_argument("--pilot", action="store_true",
                        help="Pilot: 8 maps, 1 fire, 4 policies, 3 sizes, 3 trials = 288 runs (all map types)")
    parser.add_argument("--medium", action="store_true",
                        help="Medium: 1 map, 5 fire, 4 policies, 3 sizes, 10 trials = 600 runs")
    parser.add_argument("--policy", type=str, default=None,
                        help="Run single policy only")
    parser.add_argument("--map", type=str, default=None,
                        help="Run single map only")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers")
    parser.add_argument("--output", type=str, default="results/experiment_results.csv",
                        help="Output CSV file")
    args = parser.parse_args()

    if args.smoke:
        configs = build_configs(
            maps=["maps/bottleneck_A.txt"],
            fire_positions=["center"],
            policies=POLICIES,
            crowd_sizes=[100],
            n_trials=1,
        )
        print(f"SMOKE TEST: {len(configs)} runs (sanity check — ~15 seconds)")
    elif args.quick:
        configs = build_configs(
            maps=["maps/bottleneck_A.txt"],
            fire_positions=["center"],
            policies=POLICIES,
            crowd_sizes=CROWD_SIZES,
            n_trials=5,
        )
        print(f"QUICK TEST: {len(configs)} runs")
    elif args.pilot:
        configs = build_configs(
            maps=MAPS,  # All 8 maps
            fire_positions=["center"],
            policies=POLICIES,
            crowd_sizes=CROWD_SIZES,
            n_trials=3,
        )
        print(f"PILOT TEST: {len(configs)} runs (all maps, center fire, 3 trials)")
    elif args.medium:
        configs = build_configs(
            maps=["maps/bottleneck_A.txt"],
            fire_positions=FIRE_POSITIONS,
            n_trials=10,
        )
        print(f"MEDIUM TEST: {len(configs)} runs")
    else:
        maps = [args.map] if args.map else None
        policies = [args.policy] if args.policy else None
        configs = build_configs(maps=maps, policies=policies)
        print(f"FULL EXPERIMENT: {len(configs)} runs")

    run_experiment(configs, output_file=args.output, n_workers=args.workers)


if __name__ == "__main__":
    main()
