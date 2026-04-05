"""
Debug test: Verify Router intelligence features actually fire.
Tracks: hotspot detection, exit selection, Dijkstra vs BFS path differences,
congestion counts, collecting/escorting triggers.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from config import SimConfig
from evacsim.simulation import Simulation
from evacsim.guide import RouterGuide, bfs_path, dijkstra_congestion_path

config = SimConfig(
    map_file="maps/bottleneck_A.txt",
    guide_policy="router",
    n_civilians=100,
    n_guides=4,
    seed=42,
    max_ticks=3000,
)

sim = Simulation(config)

# Monkey-patch Router methods to add debug logging
debug_log = []

original_find_hotspot = RouterGuide._find_hotspot
original_start_escort = RouterGuide._start_escort
original_do_collecting = RouterGuide._do_collecting
original_do_scanning = RouterGuide._do_scanning
original_update_exit_counts = RouterGuide._update_exit_counts

def debug_find_hotspot(self, env, smoke, civilians):
    result = original_find_hotspot(self, env, smoke, civilians)
    if result is not None:
        # Count civilians near hotspot
        from evacsim.civilian import ESCAPED, DEAD, IDLE, REACTING
        n_near = sum(1 for c in civilians
                     if c.state not in (ESCAPED, DEAD, IDLE, REACTING)
                     and c.assigned_guide is None
                     and np.linalg.norm(c.pos - result) < env.cell_size * 5)
        debug_log.append({
            "tick": sim.tick,
            "event": "HOTSPOT_FOUND",
            "guide_id": id(self),
            "hotspot_pos": f"({result[0]:.0f},{result[1]:.0f})",
            "civilians_near": n_near,
            "territory": self._territory,
        })
    return result

def debug_start_escort(self, env, smoke, civilians):
    # Before escort — record which exit was chosen and why
    old_exit = self._escort_exit
    original_start_escort(self, env, smoke, civilians)

    # Log exit selection details
    exit_scores = {}
    fg = getattr(self, "_fire_grid", None)
    for ex, ey in self.known_exits:
        dist = np.hypot(self.pos[0] - ex, self.pos[1] - ey)
        congestion = self._exit_counts.get((ex, ey), 0)
        smoke_val = smoke.get_density_at(ex, ey)
        score = dist + congestion * 30.0 + smoke_val * 200.0
        exit_scores[(ex, ey)] = {
            "dist": dist,
            "congestion": congestion,
            "smoke": smoke_val,
            "score": score,
        }

    # Check if Dijkstra path differs from BFS
    chosen_exit = self._escort_exit
    dijkstra_len = len(self._waypoints) if self._waypoints else 0
    bfs_wp = bfs_path(env, self.pos, np.array(chosen_exit), fire_grid=fg) if chosen_exit else []
    bfs_len = len(bfs_wp) if bfs_wp else 0

    # Is the chosen exit the nearest?
    nearest_exit = min(self.known_exits,
                       key=lambda e: np.hypot(self.pos[0]-e[0], self.pos[1]-e[1]))
    chose_nearest = (chosen_exit == nearest_exit) if chosen_exit else None

    debug_log.append({
        "tick": sim.tick,
        "event": "ESCORT_START",
        "guide_id": id(self),
        "group_size": len(self._group),
        "chosen_exit": f"({chosen_exit[0]:.0f},{chosen_exit[1]:.0f})" if chosen_exit else "None",
        "nearest_exit": f"({nearest_exit[0]:.0f},{nearest_exit[1]:.0f})",
        "chose_nearest": chose_nearest,
        "dijkstra_path_len": dijkstra_len,
        "bfs_path_len": bfs_len,
        "path_differs": dijkstra_len != bfs_len,
        "exit_scores": {f"({k[0]:.0f},{k[1]:.0f})": v for k, v in exit_scores.items()},
    })

def debug_do_collecting(self, env, smoke, civilians):
    before_group = len(self._group)
    before_phase = self._phase
    original_do_collecting(self, env, smoke, civilians)
    after_group = len(self._group)

    if before_phase != self._phase:
        trigger = "max_group" if after_group >= self._max_group else "timeout"
        debug_log.append({
            "tick": sim.tick,
            "event": "COLLECT_END",
            "guide_id": id(self),
            "trigger": trigger,
            "group_size": after_group,
            "wait_ticks": self._collect_wait,
            "new_phase": self._phase,
        })

def debug_update_exit_counts(self, civilians):
    original_update_exit_counts(self, civilians)
    if any(v > 0 for v in self._exit_counts.values()):
        debug_log.append({
            "tick": sim.tick,
            "event": "EXIT_CONGESTION",
            "guide_id": id(self),
            "counts": {f"({k[0]:.0f},{k[1]:.0f})": v for k, v in self._exit_counts.items()},
        })

# Apply patches
RouterGuide._find_hotspot = debug_find_hotspot
RouterGuide._start_escort = debug_start_escort
RouterGuide._do_collecting = debug_do_collecting
RouterGuide._update_exit_counts = debug_update_exit_counts

# Run simulation
print("Running Router debug test (100 civilians, bottleneck_A, seed=42)...")
print(f"Guides: {len(sim.guides)}")
print()

result = sim.run()

print(f"\n{'='*70}")
print(f"RESULT: TV={result['TV']}, TM={result['TM']}, "
      f"survival={result['survival_rate']*100:.1f}%, ticks={sim.tick}")
print(f"{'='*70}\n")

# Analyze debug log
hotspot_events = [e for e in debug_log if e["event"] == "HOTSPOT_FOUND"]
escort_events = [e for e in debug_log if e["event"] == "ESCORT_START"]
collect_events = [e for e in debug_log if e["event"] == "COLLECT_END"]
congestion_events = [e for e in debug_log if e["event"] == "EXIT_CONGESTION"]

print(f"HOTSPOT detections: {len(hotspot_events)}")
for e in hotspot_events[:15]:
    print(f"  tick {e['tick']:>4}: pos={e['hotspot_pos']}, "
          f"civilians_near={e['civilians_near']}")
if len(hotspot_events) > 15:
    print(f"  ... and {len(hotspot_events)-15} more")

print(f"\nESCORT starts: {len(escort_events)}")
chose_non_nearest = 0
path_differed = 0
for e in escort_events:
    print(f"  tick {e['tick']:>4}: group={e['group_size']}, "
          f"exit={e['chosen_exit']}, nearest={e['nearest_exit']}, "
          f"chose_nearest={e['chose_nearest']}, "
          f"dijkstra={e['dijkstra_path_len']} vs bfs={e['bfs_path_len']}, "
          f"path_differs={e['path_differs']}")
    if not e['chose_nearest']:
        chose_non_nearest += 1
    if e['path_differs']:
        path_differed += 1

print(f"\n  KEY STATS:")
print(f"    Chose NON-nearest exit: {chose_non_nearest}/{len(escort_events)} times")
print(f"    Dijkstra path differed from BFS: {path_differed}/{len(escort_events)} times")

print(f"\nCOLLECT phase endings: {len(collect_events)}")
timeout_triggers = sum(1 for e in collect_events if e["trigger"] == "timeout")
maxgroup_triggers = sum(1 for e in collect_events if e["trigger"] == "max_group")
for e in collect_events:
    print(f"  tick {e['tick']:>4}: trigger={e['trigger']}, "
          f"group={e['group_size']}, waited={e['wait_ticks']} ticks")
print(f"  timeout triggers: {timeout_triggers}, max_group triggers: {maxgroup_triggers}")

print(f"\nEXIT CONGESTION snapshots: {len(congestion_events)}")
for e in congestion_events[:10]:
    print(f"  tick {e['tick']:>4}: {e['counts']}")
if len(congestion_events) > 10:
    print(f"  ... and {len(congestion_events)-10} more")

# Summary
print(f"\n{'='*70}")
print(f"INTELLIGENCE SUMMARY")
print(f"{'='*70}")
print(f"  Hotspots found:           {len(hotspot_events)}")
print(f"  Escorts started:          {len(escort_events)}")
print(f"  Chose non-nearest exit:   {chose_non_nearest}/{len(escort_events)} "
      f"({'YES — congestion intelligence works!' if chose_non_nearest > 0 else 'NEVER — always picks nearest'})")
print(f"  Dijkstra != BFS path:     {path_differed}/{len(escort_events)} "
      f"({'YES — density avoidance works!' if path_differed > 0 else 'NEVER — same as BFS'})")
print(f"  Congestion detected:      {len(congestion_events)} snapshots")
print(f"  max_group(6) triggers:    {maxgroup_triggers}")
print(f"  timeout(30) triggers:     {timeout_triggers}")
