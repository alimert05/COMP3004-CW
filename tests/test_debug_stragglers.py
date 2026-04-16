"""Debug test: check if civilians get lost from guide groups."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from config import SimConfig
from evacsim.simulation import Simulation
from evacsim.civilian import FOLLOWING_GUIDE, DEAD, ESCAPED

# Track stragglers across multiple maps
maps = ['bottleneck_A', 'bottleneck_C', 'open_plan_C']

for mapname in maps:
    for policy in ['sweeper', 'router']:
        config = SimConfig(
            map_file=f'maps/{mapname}.txt',
            fire_position='center',
            n_civilians=100, guide_policy=policy,
            seed=42, max_ticks=3000
        )
        sim = Simulation(config)

        # Track stats per tick
        max_stragglers_seen = 0
        total_straggler_ticks = 0  # sum across all guides/ticks
        lost_and_died = 0  # civilians who were in a group, fell behind, then died
        civs_lost_from_group = set()

        for _ in range(config.max_ticks):
            if not sim.step():
                break

            for g in sim.guides:
                if not g.alive:
                    continue
                group = getattr(g, '_group', [])
                if len(group) < 2:
                    continue

                # Check distance of each group member from guide
                for civ in group:
                    if civ.state != FOLLOWING_GUIDE:
                        continue
                    dist = np.linalg.norm(civ.pos - g.pos)
                    # Straggler threshold: >5 cells from guide (50 units)
                    if dist > 50:
                        total_straggler_ticks += 1
                        civs_lost_from_group.add(id(civ))

        # After simulation, check fate of stragglers
        for civ in sim.civilians:
            if id(civ) in civs_lost_from_group and civ.state == DEAD:
                lost_and_died += 1

        # Final results
        escaped = sum(1 for c in sim.civilians if c.state == ESCAPED)
        dead = sum(1 for c in sim.civilians if c.state == DEAD)

        print(f"{mapname:>14} / {policy:>7}: "
              f"survival={escaped}/100, "
              f"distinct_stragglers={len(civs_lost_from_group)}, "
              f"straggler_ticks={total_straggler_ticks}, "
              f"lost_and_died={lost_and_died}/{len(civs_lost_from_group) if civs_lost_from_group else 0}")
