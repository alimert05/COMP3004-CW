"""Quick test: verify new maps load and run in simulation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import SimConfig
from evacsim.simulation import Simulation

for mapname in ['bottleneck_D', 'open_plan_D']:
    f = f'maps/{mapname}.txt'
    for policy in ['none', 'beacon', 'sweeper', 'router']:
        config = SimConfig(
            map_file=f, fire_position='center',
            n_civilians=50, guide_policy=policy,
            seed=42, max_ticks=3000
        )
        sim = Simulation(config)
        exits = list(sim.env.exits)
        result = sim.run()
        print(f"{mapname} / {policy:>7}: exits={len(exits)}, "
              f"TV={result['TV']}, TM={result['TM']}, "
              f"survival={result['survival_rate']*100:.0f}%")
    print()
