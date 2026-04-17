"""Verify deterministic seed generation across processes."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.runner import _deterministic_seed

s1 = _deterministic_seed("maps/bottleneck_A.txt", "center", "router", 100, 0)
s2 = _deterministic_seed("maps/bottleneck_A.txt", "center", "router", 100, 0)
print(f"Same args, 2 calls: {s1} vs {s2} -> {'OK' if s1 == s2 else 'FAIL'}")

# Different trials should give different seeds
s3 = _deterministic_seed("maps/bottleneck_A.txt", "center", "router", 100, 1)
print(f"Trial 0 vs Trial 1: {s1} vs {s3} -> {'OK' if s1 != s3 else 'FAIL'}")

# Different policies should give different seeds
s4 = _deterministic_seed("maps/bottleneck_A.txt", "center", "sweeper", 100, 0)
print(f"Router vs Sweeper: {s1} vs {s4} -> {'OK' if s1 != s4 else 'FAIL'}")
