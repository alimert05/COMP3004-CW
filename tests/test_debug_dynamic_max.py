"""Debug test: verify dynamic max_group actually changes during simulation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import SimConfig
from evacsim.simulation import Simulation

# SWEEPER - run full sim, track min effective_max seen per guide
config = SimConfig(
    map_file="maps/bottleneck_A.txt", fire_position="center",
    n_civilians=100, guide_policy="sweeper", seed=42, max_ticks=3000
)
sim = Simulation(config)

min_eff = {i: 999 for i in range(4)}
saw_change = {i: [] for i in range(4)}

for tick in range(2000):
    sim.step()
    for i, g in enumerate(sim.guides):
        if g.alive:
            eff = g._effective_max_group()
            if eff < min_eff[i]:
                min_eff[i] = eff
            if eff < g._base_max_group:
                saw_change[i].append((tick, eff, len(g._group), g._phase))

print("=== SWEEPER: Did effective_max ever drop below base? ===")
for i in range(4):
    if saw_change[i]:
        print(f"  Guide {i}: YES, dropped {len(saw_change[i])} times, min={min_eff[i]}")
        for t, e, grp, phase in saw_change[i][:5]:
            print(f"    tick={t} effective={e} group={grp} phase={phase}")
        if len(saw_change[i]) > 5:
            print(f"    ... and {len(saw_change[i])-5} more")
    else:
        alive = sim.guides[i].alive
        base = sim.guides[i]._base_max_group if alive else "dead"
        print(f"  Guide {i}: NEVER dropped below base={base}")

print()

# ROUTER - same test
config2 = SimConfig(
    map_file="maps/bottleneck_A.txt", fire_position="center",
    n_civilians=100, guide_policy="router", seed=42, max_ticks=3000
)
sim2 = Simulation(config2)

min_eff2 = {i: 999 for i in range(4)}
saw_change2 = {i: [] for i in range(4)}
exceeded_cap = []

for tick in range(2000):
    sim2.step()
    for i, g in enumerate(sim2.guides):
        if g.alive:
            eff = g._effective_max_group()
            if eff < min_eff2[i]:
                min_eff2[i] = eff
            if eff < g._base_max_group:
                saw_change2[i].append((tick, eff, len(g._group), g._phase))
            if len(g._group) > eff:
                exceeded_cap.append((tick, i, len(g._group), eff, g._phase))

print("=== ROUTER: Did effective_max ever drop below base? ===")
for i in range(4):
    if saw_change2[i]:
        print(f"  Guide {i}: YES, dropped {len(saw_change2[i])} times, min={min_eff2[i]}")
        for t, e, grp, phase in saw_change2[i][:5]:
            print(f"    tick={t} effective={e} group={grp} phase={phase}")
        if len(saw_change2[i]) > 5:
            print(f"    ... and {len(saw_change2[i])-5} more")
    else:
        print(f"  Guide {i}: NEVER dropped below base")

print()
print(f"=== ROUTER: Group exceeded effective_max? {len(exceeded_cap)} instances ===")
for t, i, grp, eff, phase in exceeded_cap[:10]:
    print(f"  tick={t} guide={i} group={grp} effective_max={eff} phase={phase}")
if len(exceeded_cap) > 10:
    print(f"  ... and {len(exceeded_cap)-10} more")
