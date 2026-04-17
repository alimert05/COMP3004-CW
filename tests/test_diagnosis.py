"""Pre-flight diagnostic tests before 7-hour factorial experiment.

Checks:
1. Python hash() determinism (seed generation reproducibility)
2. Same-seed simulation reproducibility (same config → same result)
3. State pollution across simulations in the same process
4. Memory stability across repeated runs
5. All 8 maps can spawn simulations without errors
6. Corner fire positions work on each map type
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SimConfig
from evacsim.simulation import Simulation
from evacsim.guide import SweeperGuide, RouterBlackboard


def check_hash_determinism():
    """Verify Python's hash() is deterministic across calls in same process."""
    print("\n=== CHECK 1: hash() determinism ===")
    key = ("maps/bottleneck_A.txt", "center", "router", 100, 0)
    h1 = hash(key) % (2**31)
    h2 = hash(key) % (2**31)
    print(f"  Same key, 2 calls: {h1} vs {h2} -> {'MATCH' if h1 == h2 else 'DIFFER'}")

    # Check across module reload (different Python process needed — but in-process should match)
    import subprocess
    code = (
        "h = hash(('maps/bottleneck_A.txt', 'center', 'router', 100, 0)) % (2**31); "
        "print(h)"
    )
    out1 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip()
    out2 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip()
    print(f"  Subprocess 1: {out1}")
    print(f"  Subprocess 2: {out2}")
    print(f"  Cross-process:  {'MATCH (deterministic)' if out1 == out2 else 'DIFFER (non-deterministic)'}")


def check_simulation_reproducibility():
    """Run same config twice in same process, compare results."""
    print("\n=== CHECK 2: Same-seed reproducibility ===")
    config = SimConfig(
        map_file="maps/bottleneck_A.txt", fire_position="center",
        n_civilians=50, guide_policy="router", seed=42, max_ticks=1000,
    )
    r1 = Simulation(config).run()
    r2 = Simulation(config).run()
    keys = ["TV", "TM", "MP", "MT", "survival_rate", "evacuation_time"]
    print(f"  {'Metric':<15} {'Run 1':>10} {'Run 2':>10} {'Match':>8}")
    all_match = True
    for k in keys:
        v1, v2 = r1[k], r2[k]
        match = v1 == v2
        all_match &= match
        print(f"  {k:<15} {str(v1):>10} {str(v2):>10} {'OK' if match else 'FAIL':>8}")
    print(f"  OVERALL: {'REPRODUCIBLE' if all_match else 'NON-DETERMINISTIC (BUG!)'}")


def check_state_pollution():
    """Run Sweeper sim, then Router sim, then Sweeper sim — check no class-level
    state leaks between runs in the same process."""
    print("\n=== CHECK 3: State pollution across sims ===")

    # Run 1: sweeper
    c1 = SimConfig(map_file="maps/bottleneck_A.txt", guide_policy="sweeper",
                    n_civilians=50, seed=42, max_ticks=300)
    sim1 = Simulation(c1)
    n_sweepers_1 = len(SweeperGuide._all_sweepers)
    r1 = sim1.run()

    # Run 2: router (different policy)
    c2 = SimConfig(map_file="maps/bottleneck_A.txt", guide_policy="router",
                    n_civilians=50, seed=42, max_ticks=300)
    sim2 = Simulation(c2)
    # After spawning router guides, the _all_sweepers list from run 1 should be
    # either reset or at least not affect this run.
    r2 = sim2.run()

    # Run 3: sweeper again — same seed as run 1, should produce same result as r1
    c3 = SimConfig(map_file="maps/bottleneck_A.txt", guide_policy="sweeper",
                    n_civilians=50, seed=42, max_ticks=300)
    sim3 = Simulation(c3)
    n_sweepers_3 = len(SweeperGuide._all_sweepers)
    r3 = sim3.run()

    print(f"  _all_sweepers after sim1 init: {n_sweepers_1}")
    print(f"  _all_sweepers after sim3 init: {n_sweepers_3}")
    print(f"  Should be 4 each (fresh per sim): {'OK' if n_sweepers_1 == 4 and n_sweepers_3 == 4 else 'FAIL'}")

    keys = ["TV", "TM", "MP", "MT"]
    all_match = True
    print(f"  Sim1 vs Sim3 (same config, different order):")
    for k in keys:
        v1, v3 = r1[k], r3[k]
        match = v1 == v3
        all_match &= match
        print(f"    {k}: {v1} vs {v3} -> {'OK' if match else 'DIFFER (STATE POLLUTION)'}")


def check_all_maps_work():
    """Load each map + spawn each policy, verify no crashes."""
    print("\n=== CHECK 4: All 8 maps x 4 policies ===")
    maps = [
        "bottleneck_A", "bottleneck_B", "bottleneck_C", "bottleneck_D",
        "open_plan_A", "open_plan_B", "open_plan_C", "open_plan_D",
    ]
    policies = ["none", "beacon", "sweeper", "router"]
    ok, fail = 0, 0
    for m in maps:
        for p in policies:
            try:
                cfg = SimConfig(
                    map_file=f"maps/{m}.txt", fire_position="center",
                    guide_policy=p, n_civilians=50, seed=42, max_ticks=200,
                )
                sim = Simulation(cfg)
                # Run 50 ticks to exercise the code paths
                for _ in range(50):
                    sim.step()
                ok += 1
            except Exception as e:
                fail += 1
                print(f"  FAIL: {m} x {p}: {type(e).__name__}: {e}")
    print(f"  Passed: {ok}/{len(maps)*len(policies)}")
    print(f"  Failed: {fail}/{len(maps)*len(policies)}")


def check_corner_fires():
    """Verify all 5 fire positions work on each map type."""
    print("\n=== CHECK 5: All fire positions x representative maps ===")
    maps = ["bottleneck_A", "bottleneck_D", "open_plan_A", "open_plan_D"]
    fires = ["center", "top_left", "top_right", "bottom_left", "bottom_right"]
    ok, fail = 0, 0
    for m in maps:
        for f in fires:
            try:
                cfg = SimConfig(
                    map_file=f"maps/{m}.txt", fire_position=f,
                    guide_policy="router", n_civilians=50, seed=42, max_ticks=200,
                )
                sim = Simulation(cfg)
                for _ in range(50):
                    sim.step()
                ok += 1
            except Exception as e:
                fail += 1
                print(f"  FAIL: {m} x {f}: {type(e).__name__}: {e}")
    print(f"  Passed: {ok}/{len(maps)*len(fires)}")


def check_crowd_sizes():
    """Verify all crowd sizes work."""
    print("\n=== CHECK 6: Crowd sizes 50/100/150 ===")
    sizes = [50, 100, 150]
    ok, fail = 0, 0
    for s in sizes:
        for p in ["sweeper", "router"]:
            try:
                cfg = SimConfig(
                    map_file="maps/bottleneck_A.txt", fire_position="center",
                    guide_policy=p, n_civilians=s, seed=42, max_ticks=200,
                )
                sim = Simulation(cfg)
                for _ in range(50):
                    sim.step()
                ok += 1
            except Exception as e:
                fail += 1
                print(f"  FAIL: n={s} x {p}: {type(e).__name__}: {e}")
    print(f"  Passed: {ok}/{len(sizes)*2}")


def check_memory_stability():
    """Run 20 sims back to back, check memory doesn't grow wildly."""
    print("\n=== CHECK 7: Memory stability (20 back-to-back sims) ===")
    try:
        import psutil
        import gc
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024
        for i in range(20):
            cfg = SimConfig(
                map_file="maps/bottleneck_A.txt", fire_position="center",
                guide_policy="router", n_civilians=100, seed=42 + i, max_ticks=500,
            )
            Simulation(cfg).run()
            if (i + 1) % 5 == 0:
                gc.collect()
                mem = process.memory_info().rss / 1024 / 1024
                delta = mem - mem_before
                print(f"  After {i+1} sims: {mem:.1f} MB (delta: {delta:+.1f} MB)")
        mem_after = process.memory_info().rss / 1024 / 1024
        growth = mem_after - mem_before
        print(f"  Total growth: {growth:+.1f} MB over 20 runs")
        if growth > 200:
            print(f"  WARNING: >200 MB growth suggests memory leak")
        else:
            print(f"  OK: memory growth acceptable")
    except ImportError:
        print("  (psutil not installed, skipping memory check)")
        print("  Install with: pip install psutil")


if __name__ == "__main__":
    check_hash_determinism()
    check_simulation_reproducibility()
    check_state_pollution()
    check_all_maps_work()
    check_corner_fires()
    check_crowd_sizes()
    check_memory_stability()
    print("\n=== DIAGNOSIS COMPLETE ===\n")
