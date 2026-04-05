"""
FULL Router Diagnosis — run across multiple maps, crowd sizes, track EVERYTHING.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from config import SimConfig
from evacsim.simulation import Simulation
from evacsim.guide import RouterGuide, bfs_path

def run_router_debug(map_file, n_civilians, seed=42):
    config = SimConfig(
        map_file=map_file,
        guide_policy="router",
        n_civilians=n_civilians,
        n_guides=4,
        seed=seed,
        max_ticks=3000,
    )
    sim = Simulation(config)
    debug_log = []

    # ── Patches ──
    original_find_hotspot = RouterGuide._find_hotspot
    original_start_escort = RouterGuide._start_escort
    original_do_collecting = RouterGuide._do_collecting
    original_do_scanning = RouterGuide._do_scanning
    original_do_returning = RouterGuide._do_returning
    original_do_releasing = RouterGuide._do_releasing
    original_recruit_nearby = RouterGuide._recruit_nearby
    original_recruit_unlimited = RouterGuide._recruit_nearby_unlimited

    def get_gid(guide):
        for i, g in enumerate(sim.guides):
            if g is guide: return i
        return -1

    def debug_find_hotspot(self, env, smoke, civilians):
        result = original_find_hotspot(self, env, smoke, civilians)
        if result is not None:
            from evacsim.civilian import ESCAPED, DEAD, IDLE, REACTING
            n_near = sum(1 for c in civilians
                         if c.state not in (ESCAPED, DEAD, IDLE, REACTING)
                         and c.assigned_guide is None
                         and np.linalg.norm(c.pos - result) < env.cell_size * 5)
            debug_log.append({
                "tick": sim.tick, "event": "HOTSPOT", "guide": get_gid(self),
                "civilians_near": n_near,
                "pos": f"({result[0]:.0f},{result[1]:.0f})",
            })
        return result

    def debug_start_escort(self, env, smoke, civilians):
        original_start_escort(self, env, smoke, civilians)
        chosen = self._escort_exit
        if not chosen: return

        # Is it nearest?
        nearest = min(self.known_exits, key=lambda e: np.hypot(self.pos[0]-e[0], self.pos[1]-e[1]))
        chose_nearest = (chosen == nearest)

        # Dijkstra vs BFS comparison
        fg = getattr(self, "_fire_grid", None)
        bfs_wp = bfs_path(env, self.pos, np.array(chosen), fire_grid=fg)
        dij_len = len(self._waypoints) if self._waypoints else 0
        bfs_len = len(bfs_wp) if bfs_wp else 0

        # What triggered escort? Check caller
        # Log congestion data
        cong = {f"({k[0]:.0f},{k[1]:.0f})": v for k, v in self._exit_counts.items()}

        debug_log.append({
            "tick": sim.tick, "event": "ESCORT_START", "guide": get_gid(self),
            "group_size": len(self._group),
            "exit": f"({chosen[0]:.0f},{chosen[1]:.0f})",
            "chose_nearest": chose_nearest,
            "dijkstra_len": dij_len, "bfs_len": bfs_len,
            "path_differs": dij_len != bfs_len,
            "congestion": cong,
            "phase_before": self._phase,
        })

    def debug_do_collecting(self, env, smoke, civilians):
        before_phase = self._phase
        before_group = len(self._group)
        original_do_collecting(self, env, smoke, civilians)
        if before_phase != self._phase:
            trigger = "max_group" if before_group >= self._max_group else (
                "timeout" if self._collect_wait >= self._max_collect_wait else "other")
            debug_log.append({
                "tick": sim.tick, "event": "COLLECT_END", "guide": get_gid(self),
                "trigger": trigger, "group_size": len(self._group),
                "waited": self._collect_wait,
            })

    def debug_do_scanning(self, env, smoke, civilians):
        before_phase = self._phase
        before_group = len(self._group)
        original_do_scanning(self, env, smoke, civilians)
        if before_phase != self._phase:
            debug_log.append({
                "tick": sim.tick, "event": f"SCAN->{self._phase.upper()}",
                "guide": get_gid(self), "group_size": len(self._group),
            })

    def debug_do_returning(self, env, smoke, civilians):
        before_phase = self._phase
        before_group = len(self._group)
        original_do_returning(self, env, smoke, civilians)
        if before_phase != self._phase and self._phase == "escorting":
            debug_log.append({
                "tick": sim.tick, "event": "RETURN->ESCORT",
                "guide": get_gid(self), "group_size": len(self._group),
                "reason": "picked_up_on_way_home",
            })

    def debug_do_releasing(self, env):
        before_group = len(self._group)
        before_phase = self._phase
        original_do_releasing(self, env)
        if before_phase != self._phase:
            debug_log.append({
                "tick": sim.tick, "event": "RELEASE_DONE", "guide": get_gid(self),
                "released": before_group,
                "waited": self._release_wait,
                "timeout": self._release_wait >= self._release_max_wait,
            })

    # Track every recruit event
    recruit_events = []
    def debug_recruit(self, env, smoke, civilians):
        before = len(self._group)
        original_recruit_nearby(self, env, smoke, civilians)
        after = len(self._group)
        if after > before:
            recruit_events.append({
                "tick": sim.tick, "guide": get_gid(self),
                "phase": self._phase, "recruited": after - before,
                "total": after, "limited": True,
            })

    def debug_recruit_unlimited(self, env, smoke, civilians):
        before = len(self._group)
        original_recruit_unlimited(self, env, smoke, civilians)
        after = len(self._group)
        if after > before:
            recruit_events.append({
                "tick": sim.tick, "guide": get_gid(self),
                "phase": self._phase, "recruited": after - before,
                "total": after, "limited": False,
            })

    # Apply patches
    RouterGuide._find_hotspot = debug_find_hotspot
    RouterGuide._start_escort = debug_start_escort
    RouterGuide._do_collecting = debug_do_collecting
    RouterGuide._do_scanning = debug_do_scanning
    RouterGuide._do_returning = debug_do_returning
    RouterGuide._do_releasing = debug_do_releasing
    RouterGuide._recruit_nearby = debug_recruit
    RouterGuide._recruit_nearby_unlimited = debug_recruit_unlimited

    # Run
    result = sim.run()

    # Restore originals
    RouterGuide._find_hotspot = original_find_hotspot
    RouterGuide._start_escort = original_start_escort
    RouterGuide._do_collecting = original_do_collecting
    RouterGuide._do_scanning = original_do_scanning
    RouterGuide._do_returning = original_do_returning
    RouterGuide._do_releasing = original_do_releasing
    RouterGuide._recruit_nearby = original_recruit_nearby
    RouterGuide._recruit_nearby_unlimited = original_recruit_unlimited

    return result, debug_log, recruit_events, sim


# ── Run on 3 different maps + 2 crowd sizes ──
tests = [
    ("maps/bottleneck_A.txt", 50),
    ("maps/bottleneck_A.txt", 100),
    ("maps/open_plan_A.txt", 50),
    ("maps/open_plan_A.txt", 100),
    ("maps/bottleneck_C.txt", 100),
    ("maps/open_plan_B.txt", 100),
]

for map_file, n_civ in tests:
    map_name = map_file.split("/")[1].replace(".txt","")
    print(f"\n{'='*70}")
    print(f"MAP: {map_name} | CIVILIANS: {n_civ}")
    print(f"{'='*70}")

    result, debug_log, recruit_events, sim = run_router_debug(map_file, n_civ)

    print(f"RESULT: TV={result['TV']}, TM={result['TM']}, "
          f"survival={result['survival_rate']*100:.1f}%, ticks={sim.tick}")

    # Escort analysis
    escorts = [e for e in debug_log if e["event"] == "ESCORT_START"]
    hotspots = [e for e in debug_log if e["event"] == "HOTSPOT"]
    collects = [e for e in debug_log if e["event"] == "COLLECT_END"]
    releases = [e for e in debug_log if e["event"] == "RELEASE_DONE"]
    return_escorts = [e for e in debug_log if e["event"] == "RETURN->ESCORT"]

    chose_non_nearest = sum(1 for e in escorts if not e["chose_nearest"])
    path_differed = sum(1 for e in escorts if e["path_differs"])

    # Group size distribution
    group_sizes = [e["group_size"] for e in escorts]
    single_escorts = sum(1 for g in group_sizes if g == 1)
    small_escorts = sum(1 for g in group_sizes if 1 < g < 5)
    decent_escorts = sum(1 for g in group_sizes if g >= 5)

    # Recruit analysis
    limited_recruits = [r for r in recruit_events if r["limited"]]
    unlimited_recruits = [r for r in recruit_events if not r["limited"]]
    recruit_by_phase = {}
    for r in recruit_events:
        p = r["phase"]
        recruit_by_phase[p] = recruit_by_phase.get(p, 0) + r["recruited"]

    # Collect trigger analysis
    timeout_collects = sum(1 for c in collects if c["trigger"] == "timeout")
    maxgroup_collects = sum(1 for c in collects if c["trigger"] == "max_group")

    # Guide health
    guide_alive = sum(1 for g in sim.guides if g.alive)
    guide_healths = [f"{g.health:.0%}" for g in sim.guides]

    # Time analysis — how much time Router spends in each activity
    # (We can't track this precisely without more instrumentation, but we can infer)

    print(f"\n  HOTSPOTS found:           {len(hotspots)}")
    for h in hotspots:
        print(f"    tick {h['tick']:>4} G{h['guide']}: pos={h['pos']}, civs_near={h['civilians_near']}")

    print(f"\n  ESCORTS started:          {len(escorts)}")
    print(f"    Chose non-nearest exit: {chose_non_nearest}/{len(escorts)}")
    print(f"    Dijkstra != BFS path:   {path_differed}/{len(escorts)}")
    print(f"    Group=1 (wasted trip):  {single_escorts}/{len(escorts)}")
    print(f"    Group 2-4 (small):      {small_escorts}/{len(escorts)}")
    print(f"    Group 5+ (decent):      {decent_escorts}/{len(escorts)}")
    if group_sizes:
        print(f"    Mean group size:        {np.mean(group_sizes):.1f}")

    for e in escorts:
        tag = "" if e["chose_nearest"] else " <-- NON-NEAREST"
        dtag = "" if not e["path_differs"] else " <-- DIJKSTRA DIFFERS"
        print(f"    tick {e['tick']:>4} G{e['guide']}: grp={e['group_size']:>2}, "
              f"exit={e['exit']}, nearest={e['chose_nearest']}{tag}{dtag}")

    print(f"\n  COLLECTING phase ends:    {len(collects)}")
    print(f"    timeout(30) triggers:   {timeout_collects}")
    print(f"    max_group(6) triggers:  {maxgroup_collects}")

    print(f"\n  RETURN->ESCORT (picked up on way home): {len(return_escorts)}")
    for re in return_escorts:
        print(f"    tick {re['tick']:>4} G{re['guide']}: grp={re['group_size']}")

    print(f"\n  RECRUIT events: {len(recruit_events)} total")
    print(f"    Limited (capped at 6):     {len(limited_recruits)}")
    print(f"    Unlimited (during escort): {len(unlimited_recruits)}")
    print(f"    By phase: {recruit_by_phase}")
    total_recruited = sum(r["recruited"] for r in recruit_events)
    print(f"    Total civilians recruited: {total_recruited}")

    print(f"\n  RELEASES: {len(releases)}")
    timeout_rels = sum(1 for r in releases if r["timeout"])
    print(f"    Timeout releases: {timeout_rels}/{len(releases)}")

    print(f"\n  GUIDE STATUS: {guide_alive}/4 alive, healths={guide_healths}")

    # Efficiency metric
    if total_recruited > 0:
        trips = len(escorts)
        print(f"\n  EFFICIENCY: {total_recruited} recruited across {trips} trips = "
              f"{total_recruited/max(trips,1):.1f} civs/trip")
