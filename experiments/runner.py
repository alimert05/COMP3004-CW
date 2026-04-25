"""Experiment runner for the 14,400-run factorial design.

Factorial structure:
    8 maps x 5 fire positions x 4 policies x 3 crowd sizes x 30 trials
    = 14,400 simulation runs.

Maps: 4 bottleneck (A, B, C, D) and 4 open-plan (A, B, C, D), all 36x80.
Crowd sizes: 50 (well-supplied), 100 (mid), 150 (stressed).
Fire positions: centre plus four corners.
Policies: none (baseline), beacon (passive), sweeper (reactive), router
(hybrid).

Uses Python multiprocessing to run simulations in parallel and streams
each completed row to a CSV so a crash does not lose previously
completed work.

Usage:
    python -m experiments.runner                  # Full 14,400 runs.
    python -m experiments.runner --smoke          # Smoke test (4 runs, ~45 s).
    python -m experiments.runner --pilot          # Pilot (960 runs, ~32 min).
    python -m experiments.runner --policy beacon  # One policy only.
    python -m experiments.runner --map maps/bottleneck_A.txt  # One map only.

Test-mode hierarchy (increasing confidence):
    --smoke   4 runs      Quickest sanity check, one of each policy.
    --pilot   960 runs    All maps, 2 fires, 3 crowds, 5 trials.
    full      14,400 runs The main experiment.
"""

import os
import sys
import csv
import time
import zlib
import argparse
from multiprocessing import Pool
from itertools import product

# Resolve the project root (parent of this file's directory) so map
# paths and module imports work regardless of the working directory
# the runner is launched from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import SimConfig
from evacsim.simulation import run_single


# -- Experimental design --

MAPS = [
    # Bottleneck maps (4): rooms with narrow doorways.
    "maps/bottleneck_A.txt",  # Grid of small rooms, one-cell doorways.
    "maps/bottleneck_B.txt",  # Wider three-cell corridors, asymmetric exits.
    "maps/bottleneck_C.txt",  # Asymmetric: all four exits on the right wall.
    "maps/bottleneck_D.txt",  # Hospital corridor, two-cell central aisle.
    # Open-plan maps (4): large open spaces.
    "maps/open_plan_A.txt",  # Open warehouse with scattered pillars.
    "maps/open_plan_B.txt",  # Cubicle partitions, lane-like.
    "maps/open_plan_C.txt",  # Funnel: all exits clustered at the bottom.
    "maps/open_plan_D.txt",  # Two zones connected by a narrow bridge.
]

FIRE_POSITIONS = [
    "center",
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
]

POLICIES = ["none", "beacon", "sweeper", "router"]

# Crowd sizes 50, 100, 150 give per-guide ratios of about 1:12.5,
# 1:25, and 1:37.5, spanning under-loaded to over-loaded conditions.
CROWD_SIZES = [50, 100, 150]

N_TRIALS = 30


def _deterministic_seed(trial: int) -> int:
    """Return a deterministic seed derived from the trial index.

    Fully paired (randomised-block) design: only the trial index
    determines the seed, so every combination of (map, fire, crowd,
    policy) shares the same 30 random seeds. Trial N represents the
    same civilian cohort everywhere: same potency rolls, same reaction
    delays, same fire and smoke noise sequence, same RNG draw pattern
    for spawn positions. This makes every comparison in the factorial
    matched on stochastic noise, both within-cell (policy versus
    policy) and across-cell (map versus map, crowd versus crowd, fire
    versus fire). The policy, topology, crowd, and fire factors then
    isolate their own effects without interference from cohort
    variation.

    The built-in hash() is randomised per process by default, so a
    fresh process produces different seeds and breaks reproducibility
    across sessions. zlib.crc32 is deterministic across Python
    processes, sessions, and Python versions, giving stable seeds for
    any future re-run or replication.
    """
    key = f"{trial}".encode("utf-8")
    return zlib.crc32(key) % (2**31)


def build_configs(
    maps=None, fire_positions=None, policies=None, crowd_sizes=None, n_trials=None
):
    """Generate all SimConfig combinations for the requested factorial."""
    maps = maps or MAPS
    fire_positions = fire_positions or FIRE_POSITIONS
    policies = policies or POLICIES
    crowd_sizes = crowd_sizes or CROWD_SIZES
    n_trials = n_trials or N_TRIALS

    configs = []
    for map_file, fire_pos, policy, n_civ in product(
        maps, fire_positions, policies, crowd_sizes
    ):
        # Resolve relative map paths to absolute paths anchored at the
        # project root, so the simulation works from any cwd.
        resolved_map = (
            map_file
            if os.path.isabs(map_file)
            else os.path.join(PROJECT_ROOT, map_file)
        )
        for trial in range(n_trials):
            seed = _deterministic_seed(trial)
            config = SimConfig(
                map_file=resolved_map,
                fire_position=fire_pos,
                guide_policy=policy,
                n_civilians=n_civ,
                seed=seed,
                max_ticks=6000,
            )
            # Trial number for CSV output.
            config._trial = trial
            # Keep the readable relative map path for CSV display.
            config._map_display = map_file
            configs.append(config)

    return configs


def run_and_label(config):
    """Run one simulation and return its result row with trial info."""
    map_display = getattr(config, "_map_display", config.map_file)
    try:
        result = run_single(config)
        # Overwrite the absolute map path with the readable relative one.
        result["map"] = map_display
        result["trial"] = getattr(config, "_trial", 0)
        # Keep the error column present for the streaming CSV schema.
        result["error"] = ""
        return result
    except Exception as e:
        return {
            "map": map_display,
            "fire_position": config.fire_position,
            "policy": config.guide_policy,
            "n_civilians": config.n_civilians,
            "seed": config.seed,
            "trial": getattr(config, "_trial", 0),
            "error": str(e),
            "TV": 0,
            "TM": 0,
            "MP": 0,
            "MT": 0,
            "survival_rate": 0,
            "evacuation_time": 0,
            "still_alive": 0,
            "total_civilians": config.n_civilians,
            "n_guides": 0,
            "guides_survived": 0,
            "escaped_0_500": 0,
            "escaped_500_1000": 0,
            "escaped_1000_1500": 0,
            "escaped_1500_2000": 0,
            "escaped_2000_2500": 0,
            "escaped_2500_3000": 0,
            "escaped_3000_3500": 0,
            "escaped_3500_4000": 0,
            "escaped_4000_4500": 0,
            "escaped_4500_5000": 0,
            "escaped_5000_5500": 0,
            "escaped_5500_6000": 0,
        }


# CSV schema. Defined at module level so it stays stable across the
# streaming-write path; no field is added late.
CSV_FIELDNAMES = [
    "map",
    "fire_position",
    "policy",
    "n_civilians",
    "trial",
    "seed",
    "TV",
    "TM",
    "MP",
    "MT",
    "survival_rate",
    "evacuation_time",
    "still_alive",
    "total_civilians",
    "n_guides",
    "guides_survived",
    # Evacuation-curve bins (count of civilians who escaped in each
    # 500-tick window, used for cumulative escape curves). Twelve bins
    # cover 0 to 6000 ticks.
    "escaped_0_500",
    "escaped_500_1000",
    "escaped_1000_1500",
    "escaped_1500_2000",
    "escaped_2000_2500",
    "escaped_2500_3000",
    "escaped_3000_3500",
    "escaped_3500_4000",
    "escaped_4000_4500",
    "escaped_4500_5000",
    "escaped_5000_5500",
    "escaped_5500_6000",
    # Always include the error column so a late-appearing error does
    # not break the streaming writer. Blank for successful runs.
    "error",
]


def run_experiment(
    configs, output_file="results/experiment_results.csv", n_workers=None
):
    """Run every config in parallel, streaming each result to disk on completion.

    Streaming writes plus a per-row flush give durable checkpointing:
    if the run crashes (power cut, out of memory, Ctrl-C) at any
    point, everything completed so far is already on disk.
    """
    # Default eight workers, chosen for the target eight-core machine.
    # Override via --workers for machines with different core counts.
    n_workers = n_workers or 8
    total = len(configs)

    print(f"Running {total} simulations on {n_workers} workers...")
    print(f"Output: {output_file}")
    print()

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    start = time.time()
    completed = 0
    results = []  # Retained for API compatibility (returned to caller).

    # Open the CSV once, write the header, then stream each row.
    with open(output_file, "w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        csv_f.flush()

        with Pool(processes=n_workers) as pool:
            for result in pool.imap_unordered(run_and_label, configs, chunksize=4):
                # Per-row flush so a crash preserves every completed
                # run. The flush cost (about two rows per second) is
                # negligible at simulation throughput.
                writer.writerow(result)
                csv_f.flush()
                os.fsync(csv_f.fileno())

                results.append(result)
                completed += 1
                if completed % 50 == 0 or completed == total:
                    elapsed = time.time() - start
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total - completed) / rate if rate > 0 else 0
                    print(
                        f"  [{completed}/{total}] "
                        f"{elapsed:.0f}s elapsed, "
                        f"{rate:.1f} runs/s, "
                        f"ETA: {eta:.0f}s"
                    )

    elapsed = time.time() - start
    print(f"\nDone! {total} runs in {elapsed:.1f}s " f"({total/elapsed:.1f} runs/s)")
    print(f"Saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run evacuation experiments")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke: 1 map, 1 fire, 4 policies, 1 size, 1 trial = 4 runs (~45 s)",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Pilot: 8 maps, 2 fires, 4 policies, 3 sizes, 5 trials = 960 runs (~32 min)",
    )
    parser.add_argument(
        "--policy", type=str, default=None, help="Run a single policy only"
    )
    parser.add_argument("--map", type=str, default=None, help="Run a single map only")
    parser.add_argument(
        "--workers", type=int, default=None, help="Number of parallel workers"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/experiment_results.csv",
        help="Output CSV file",
    )
    args = parser.parse_args()

    if args.smoke:
        configs = build_configs(
            maps=["maps/bottleneck_A.txt"],
            fire_positions=["center"],
            policies=POLICIES,
            crowd_sizes=[100],
            n_trials=1,
        )
        print(f"SMOKE TEST: {len(configs)} runs (sanity check, about 45 seconds)")
    elif args.pilot:
        # Pilot: broad enough to serve as a reproducibility baseline.
        # 8 maps x 2 fires x 4 policies x 3 crowds x 5 trials = 960 runs.
        # Includes two fire positions (centre and top-left) to exercise
        # fire-position-dependent code paths. Five trials per cell
        # provide a small amount of variance in each. At about half a
        # run per second this takes roughly 32 minutes.
        configs = build_configs(
            maps=MAPS,
            fire_positions=["center", "top_left"],
            policies=POLICIES,
            crowd_sizes=CROWD_SIZES,
            n_trials=5,
        )
        print(
            f"PILOT TEST: {len(configs)} runs "
            f"(8 maps, 2 fires, 4 policies, 3 crowds, 5 trials)"
        )
    else:
        maps = [args.map] if args.map else None
        policies = [args.policy] if args.policy else None
        configs = build_configs(maps=maps, policies=policies)
        print(f"FULL EXPERIMENT: {len(configs)} runs")

    run_experiment(configs, output_file=args.output, n_workers=args.workers)


if __name__ == "__main__":
    main()
