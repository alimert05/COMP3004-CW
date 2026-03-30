"""Quick test of headless simulation engine."""

from config import SimConfig
from evacsim.simulation import Simulation
import time

policies = ["none", "beacon", "sweeper", "router"]

print("=" * 70)
print("HEADLESS SIMULATION TEST — Map A, 50 civilians, center fire")
print("=" * 70)

for policy in policies:
    config = SimConfig(
        map_file="maps/bottleneck_A.txt",
        fire_position="center",
        n_civilians=50,
        guide_policy=policy,
        seed=42,
        max_ticks=3000,
    )

    start = time.time()
    result = Simulation(config).run()
    elapsed = time.time() - start

    print(f"\n--- {policy.upper()} ---")
    print(f"  Escaped (TV):  {result['TV']}/{result['n_civilians']}")
    print(f"  Dead (TM):     {result['TM']}")
    print(f"  Still alive:   {result['still_alive']}")
    print(f"  Survival rate: {result['survival_rate']*100:.1f}%")
    print(f"  Mean potency:  {result['MP']:.1f}")
    print(f"  Mean escape T: {result['MT']:.1f} ticks")
    print(f"  Ticks run:     {result['evacuation_time']}")
    print(f"  Guides alive:  {result['guides_survived']}/{result['n_guides']}")
    print(f"  Time:          {elapsed:.2f}s")

print("\n" + "=" * 70)
print("DONE")
