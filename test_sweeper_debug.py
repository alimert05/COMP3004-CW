"""
Debug test: Verify Sweeper intelligence features actually fire.
Tracks: min-5 threshold, max_group triggers, danger escorts,
patrol coverage, cleared rooms, releasing phase, recruit counts.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from config import SimConfig
from evacsim.simulation import Simulation
from evacsim.guide import SweeperGuide

config = SimConfig(
    map_file="maps/bottleneck_A.txt",
    guide_policy="sweeper",
    n_civilians=100,
    n_guides=4,
    seed=42,
    max_ticks=3000,
)

sim = Simulation(config)

# Debug log
debug_log = []

# Monkey-patch key methods
original_do_collecting = SweeperGuide._do_collecting
original_do_escorting = SweeperGuide._do_escorting
original_do_releasing = SweeperGuide._do_releasing
original_start_escort = SweeperGuide._start_escort
original_recruit_nearby = SweeperGuide._recruit_nearby
original_recruit_nearby_escort = SweeperGuide._recruit_nearby_escort
original_advance_patrol = SweeperGuide._advance_patrol

# Track per-guide state
guide_stats = {}

def get_gid(guide):
    """Short guide ID."""
    for i, g in enumerate(sim.guides):
        if g is guide:
            return i
    return -1

def debug_do_collecting(self, env, smoke, civilians):
    gid = get_gid(self)
    before_phase = self._phase
    before_group = len(self._group)

    original_do_collecting(self, env, smoke, civilians)

    after_phase = self._phase
    after_group = len(self._group)

    if before_phase != after_phase and after_phase == "escorting":
        # Determine WHY escort started
        sg = getattr(self, "_smoke_grid", None)
        fg = getattr(self, "_fire_grid", None)

        # Check danger conditions (same logic as in _do_collecting)
        danger = False
        if before_group > 0:
            rows, cols = env.grid.shape
            gr, gc = env.world_to_grid(self.pos[0], self.pos[1])
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = gr + dr, gc + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if fg is not None and fg[nr, nc] > 0.1:
                            danger = True
                        if sg is not None and sg[nr, nc] > 0.4:
                            danger = True

            if not danger:
                for civ in self._group:
                    if smoke.get_density_at(civ.pos[0], civ.pos[1]) > 0.3:
                        danger = True
                        break

        if after_group >= self._max_group:
            trigger = "max_group"
        elif danger:
            trigger = "DANGER"
        elif after_group >= 5:
            trigger = "min-5"
        else:
            trigger = "unknown"

        debug_log.append({
            "tick": sim.tick,
            "event": "ESCORT_START",
            "guide": gid,
            "trigger": trigger,
            "group_size": after_group,
            "exit": f"({self._escort_exit[0]:.0f},{self._escort_exit[1]:.0f})" if self._escort_exit else "None",
        })


def debug_do_escorting(self, env, smoke, civilians):
    before_group = len(self._group)
    original_do_escorting(self, env, smoke, civilians)
    after_group = len(self._group)

    # Track escort pickups
    if after_group > before_group:
        gid = get_gid(self)
        debug_log.append({
            "tick": sim.tick,
            "event": "ESCORT_PICKUP",
            "guide": gid,
            "picked_up": after_group - before_group,
            "total_group": after_group,
        })


def debug_do_releasing(self, env):
    gid = get_gid(self)
    before_group = len(self._group)
    before_phase = self._phase
    original_do_releasing(self, env)
    after_group = len(self._group)
    after_phase = self._phase

    if before_phase != after_phase:
        released = before_group - after_group
        debug_log.append({
            "tick": sim.tick,
            "event": "RELEASE_DONE",
            "guide": gid,
            "released": released,
            "waited": self._release_wait,
            "timeout": self._release_wait >= self._release_max_wait,
        })


def debug_advance_patrol(self, env):
    gid = get_gid(self)
    before_idx = self._patrol_idx
    before_cleared = len(self._cleared_points)
    original_advance_patrol(self, env)
    after_idx = self._patrol_idx

    skipped = (after_idx - before_idx - 1) % len(self._patrol_route) if self._patrol_route else 0
    if skipped > 0:
        debug_log.append({
            "tick": sim.tick,
            "event": "PATROL_SKIP",
            "guide": gid,
            "skipped": skipped,
            "cleared_count": len(self._cleared_points),
            "new_idx": after_idx,
        })


# Apply patches
SweeperGuide._do_collecting = debug_do_collecting
SweeperGuide._do_escorting = debug_do_escorting
SweeperGuide._do_releasing = debug_do_releasing
SweeperGuide._advance_patrol = debug_advance_patrol

# Run simulation
print("Running Sweeper debug test (100 civilians, bottleneck_A, seed=42)...")
print(f"Guides: {len(sim.guides)}")
for i, g in enumerate(sim.guides):
    print(f"  Guide {i}: patrol_points={len(g._patrol_route)}, "
          f"max_group={g._max_group}, pos=({g.pos[0]:.0f},{g.pos[1]:.0f})")
print()

result = sim.run()

print(f"\n{'='*70}")
print(f"RESULT: TV={result['TV']}, TM={result['TM']}, "
      f"survival={result['survival_rate']*100:.1f}%, ticks={sim.tick}")
print(f"{'='*70}\n")

# ── Analyze ──

escort_events = [e for e in debug_log if e["event"] == "ESCORT_START"]
pickup_events = [e for e in debug_log if e["event"] == "ESCORT_PICKUP"]
release_events = [e for e in debug_log if e["event"] == "RELEASE_DONE"]
skip_events = [e for e in debug_log if e["event"] == "PATROL_SKIP"]

# 1. Escort triggers
print("ESCORT TRIGGERS:")
min5_count = sum(1 for e in escort_events if e["trigger"] == "min-5")
danger_count = sum(1 for e in escort_events if e["trigger"] == "DANGER")
maxgroup_count = sum(1 for e in escort_events if e["trigger"] == "max_group")
unknown_count = sum(1 for e in escort_events if e["trigger"] == "unknown")

print(f"  Total escorts: {len(escort_events)}")
print(f"  min-5 triggers:     {min5_count}")
print(f"  DANGER triggers:    {danger_count}")
print(f"  max_group triggers: {maxgroup_count}")
print(f"  unknown triggers:   {unknown_count}")
print()

for e in escort_events:
    print(f"  tick {e['tick']:>4} | Guide {e['guide']} | trigger={e['trigger']:>10} | "
          f"group={e['group_size']:>2} | exit={e['exit']}")

# 2. Escort pickups (unlimited recruit during escort)
print(f"\nESCORT PICKUPS (unlimited recruit while escorting):")
print(f"  Total pickup events: {len(pickup_events)}")
total_picked = sum(e["picked_up"] for e in pickup_events)
print(f"  Total civilians picked up during escort: {total_picked}")
for e in pickup_events[:15]:
    print(f"  tick {e['tick']:>4} | Guide {e['guide']} | +{e['picked_up']} → total {e['total_group']}")
if len(pickup_events) > 15:
    print(f"  ... and {len(pickup_events)-15} more")

# 3. Release phase
print(f"\nRELEASE PHASE:")
print(f"  Total releases: {len(release_events)}")
timeout_releases = sum(1 for e in release_events if e["timeout"])
for e in release_events:
    print(f"  tick {e['tick']:>4} | Guide {e['guide']} | released={e['released']} | "
          f"waited={e['waited']} ticks | timeout={e['timeout']}")
print(f"  Timeout releases: {timeout_releases}/{len(release_events)}")

# 4. Patrol skips (cleared rooms)
print(f"\nPATROL SKIPS (cleared rooms):")
print(f"  Total skip events: {len(skip_events)}")
for e in skip_events[:10]:
    print(f"  tick {e['tick']:>4} | Guide {e['guide']} | skipped={e['skipped']} | "
          f"cleared_total={e['cleared_count']}")
if len(skip_events) > 10:
    print(f"  ... and {len(skip_events)-10} more")

# 5. Per-guide summary
print(f"\nPER-GUIDE SUMMARY:")
for gid in range(len(sim.guides)):
    g = sim.guides[gid]
    g_escorts = [e for e in escort_events if e["guide"] == gid]
    g_pickups = sum(e["picked_up"] for e in pickup_events if e["guide"] == gid)
    g_releases = [e for e in release_events if e["guide"] == gid]
    total_escorted = sum(e["group_size"] for e in g_escorts) + g_pickups
    print(f"  Guide {gid}: {len(g_escorts)} escorts, {g_pickups} escort-pickups, "
          f"~{total_escorted} civilians handled, alive={g.alive}, health={g.health:.0f}")

# 6. Summary
print(f"\n{'='*70}")
print(f"INTELLIGENCE SUMMARY")
print(f"{'='*70}")
print(f"  min-5 threshold fires:  {min5_count} times {'✓' if min5_count > 0 else '✗ NEVER FIRES'}")
print(f"  DANGER escort fires:    {danger_count} times {'✓' if danger_count > 0 else '✗ NEVER FIRES'}")
print(f"  max_group(19) fires:    {maxgroup_count} times {'(expected rare)' if maxgroup_count == 0 else '✓'}")
print(f"  Escort pickups work:    {len(pickup_events)} events {'✓' if pickup_events else '✗ NEVER'}")
print(f"  Patrol skipping works:  {len(skip_events)} events {'✓' if skip_events else '✗ NEVER'}")
print(f"  Release phase works:    {len(release_events)} events {'✓' if release_events else '✗ NEVER'}")
