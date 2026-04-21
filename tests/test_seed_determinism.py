"""Verify deterministic seed generation across processes.

Fully paired design: seeds depend ONLY on trial number. Every
combination of (map, fire, crowd, policy) shares the same 30 seeds, so
trial N represents one cohort of random draws applied everywhere.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.runner import _deterministic_seed, build_configs

# Same args, two calls -> same seed (basic determinism).
s1 = _deterministic_seed(0)
s2 = _deterministic_seed(0)
print(f"Same args, 2 calls: {s1} vs {s2} -> {'OK' if s1 == s2 else 'FAIL'}")

# Different trials should give different seeds.
s3 = _deterministic_seed(1)
print(f"Trial 0 vs Trial 1: {s1} vs {s3} -> {'OK' if s1 != s3 else 'FAIL'}")

# Full-paired check: every (map, fire, crowd, policy) combination must
# share one seed per trial. Inspect the generated configs and confirm.
configs = build_configs(
    maps=["maps/bottleneck_A.txt", "maps/open_plan_C.txt"],
    fire_positions=["center", "top_left"],
    policies=["none", "beacon", "sweeper", "router"],
    crowd_sizes=[50, 100, 150],
    n_trials=3,
)
# Group by trial, collect ALL seeds used across every (map,fire,crowd,policy).
by_trial = {}
for c in configs:
    by_trial.setdefault(c._trial, set()).add(c.seed)
all_paired = all(len(seeds) == 1 for seeds in by_trial.values())
print(f"All (map, fire, crowd, policy) combinations share one seed per trial: "
      f"{'OK' if all_paired else 'FAIL'} (per-trial seed counts: "
      f"{[len(s) for s in by_trial.values()]})")

# Show the 3 seeds for verification.
unique_seeds = sorted({c.seed for c in configs if c._trial == 0})
print(f"Trial 0 seed (used everywhere): {unique_seeds}")
unique_seeds = sorted({c.seed for c in configs if c._trial == 1})
print(f"Trial 1 seed (used everywhere): {unique_seeds}")
unique_seeds = sorted({c.seed for c in configs if c._trial == 2})
print(f"Trial 2 seed (used everywhere): {unique_seeds}")
