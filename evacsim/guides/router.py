"""RouterGuide + RouterBlackboard — hybrid reactive-deliberative policy (L3)."""
import math
import numpy as np
from collections import deque
from evacsim.environment import Environment, CellType
from evacsim.smoke import SmokeSystem
from evacsim.fire import FireSystem
from evacsim.guides.base import Guide
from evacsim.guides.common import (
    bfs_path,
    dijkstra_congestion_path,
    predict_spread,
    dbscan_cluster,
    build_density_grid,
)


class RouterBlackboard:
    """Shared coordination blackboard for Router guides.

    Multi-agent coordination mechanism (blackboard architecture):

    1. CLAIMS — Routers write claimed hotspot positions when committing
       to a rescue. Other Routers skip claimed hotspots to avoid overlap.
       Claims auto-expire after CLAIM_EXPIRY ticks to handle guide death
       or rerouting (claims represent future commitment — they need to
       persist long enough to cover travel + collection time).

    2. POSITIONS — each Router reports its current position every tick.
       Other Routers read these positions when scoring hotspots and apply
       a distance-aware repulsion field (cumulative inverse-distance):
       hotspots near clusters of other guides receive STACKED penalties,
       encouraging decentralised spread without hard territory assignment.
       Positions expire after POSITION_EXPIRY ticks (much shorter than
       claims because they track "where I AM", not "where I'm GOING").

    This implements decentralised coordination via shared spatial
    awareness — an inter-agent potential-field approach inspired by
    Reynolds' boids (1987) and potential-field path planning. See the
    intelligentAgents_coordination lecture on cell division vs.
    decentralised spreading through potential functions.
    """

    CLAIM_EXPIRY = 60         # ticks before a hotspot claim expires
    POSITION_EXPIRY = 10      # ticks before a position becomes stale
    EXIT_CLAIM_EXPIRY = 10    # ticks before an exit claim expires
                              # (refreshed every tick during ESCORTING,
                              # so TTL matches the positions pattern)

    def __init__(self):
        self._claims = {}       # guide_id → (position, tick_claimed)
        self._positions = {}    # guide_id → (position, tick_updated)
        self._exit_claims = {}  # guide_id → (exit_pos, group_size, tick)

    # ── Claims (existing) ──

    def claim(self, guide_id, position, tick):
        """Claim a hotspot position."""
        self._claims[guide_id] = (np.array(position), tick)

    def release(self, guide_id):
        """Release a claim (guide finished escorting or died).
        Also clears any exit claim (defensive — any cleanup path that
        releases a hotspot claim should also release the exit claim)."""
        self._claims.pop(guide_id, None)
        self._exit_claims.pop(guide_id, None)

    def is_claimed(self, position, current_tick, exclude_guide=None,
                   radius=30.0):
        """Check if a position is already claimed by another guide.
        Uses radius matching — positions within `radius` units count as same hotspot."""
        for gid, (pos, tick) in list(self._claims.items()):
            if gid == exclude_guide:
                continue
            if current_tick - tick > self.CLAIM_EXPIRY:
                del self._claims[gid]  # expired
                continue
            if np.linalg.norm(pos - np.array(position)) < radius:
                return True
        return False

    # ── Positions (NEW — inter-guide spread) ──

    def update_position(self, guide_id, position, tick):
        """Record the current position of a Router on the blackboard.
        Called every tick by each Router."""
        self._positions[guide_id] = (np.array(position, dtype=float), tick)

    def get_other_positions(self, exclude_guide, current_tick):
        """Return list of currently-fresh positions of OTHER Routers.
        Stale positions (older than POSITION_EXPIRY) are filtered out
        and cleaned up lazily, so dead or departed guides don't keep
        "holding" territory.
        """
        result = []
        for gid, (pos, tick) in list(self._positions.items()):
            if gid == exclude_guide:
                continue
            if current_tick - tick > self.POSITION_EXPIRY:
                del self._positions[gid]  # lazy cleanup of stale entries
                continue
            result.append(pos)
        return result

    # ── Exit claims (A1 — destination coordination) ──

    def update_exit_claim(self, guide_id, exit_pos, group_size, tick):
        """Refresh my exit claim. Called every tick while ESCORTING so the
        entry stays fresh. Same pattern as `update_position` — if the guide
        dies silently, the entry auto-expires in EXIT_CLAIM_EXPIRY ticks."""
        self._exit_claims[guide_id] = (
            np.array(exit_pos, dtype=float), int(group_size), tick,
        )

    def release_exit_claim(self, guide_id):
        """Release exit claim (arrived at exit → group is now in the queue,
        no longer 'incoming'). Avoids double-counting for the brief window
        between arrival and natural expiry."""
        self._exit_claims.pop(guide_id, None)

    def get_incoming_count(self, exit_pos, current_tick,
                           exclude_guide=None, radius=15.0):
        """Sum of `group_size` across OTHER Routers currently escorting
        toward this exit (within `radius` world units — handles minor
        position variation).

        Used in exit scoring: current_queue + incoming = effective queue
        a newly arriving group will face. Prevents multiple Routers from
        independently picking the same 'cleanest' exit and piling up.
        """
        total = 0
        target = np.array(exit_pos, dtype=float)
        for gid, (pos, size, tick) in list(self._exit_claims.items()):
            if gid == exclude_guide:
                continue
            if current_tick - tick > self.EXIT_CLAIM_EXPIRY:
                del self._exit_claims[gid]  # lazy cleanup
                continue
            if np.linalg.norm(pos - target) < radius:
                total += size
        return total

    def reset(self):
        """Clear all blackboard state (new simulation)."""
        self._claims.clear()
        self._positions.clear()
        self._exit_claims.clear()


class RouterGuide(Guide):
    """
    Congestion-Aware Router (inspired by Zhang et al. 2020, IEEE ICRA).

    Uses congestion-aware Dijkstra pathfinding where path cost includes
    civilian density, smoke, and fire. Routes civilians on the LEAST
    CONGESTED PATH to the LEAST CONGESTED EXIT.

    Fundamentally different from other policies:
      - Beacon: STATIC at exits, broadcasts one exit
      - Sweeper: PATROLS rooms, collects groups, escorts to NEAREST exit
      - Router: REACTS to congestion hotspots, collects stuck civilians,
                escorts to OPTIMAL exit via LEAST CONGESTED path

    Behaviour:
      1. Go to territory center, scan for congestion hotspots
      2. Hotspot found → go there, collect civilians (min 5 for efficiency)
      3. Escort group to least congested exit using Dijkstra density path
      4. Danger detected → emergency escort regardless of group size
      5. Release at exit → return to territory center → scan again

    Intelligence (Zhang et al. 2020):
      - Population density map for congestion detection
      - Dijkstra pathfinding with density-weighted costs
      - Urgency-weighted hotspot scoring (smoke exposure increases priority)
      - Danger-aware: emergency escort if fire/smoke approaches group
      - Smoke-aware solo movement (guides avoid unnecessary smoke exposure)
    """

    SCANNING = "scanning"
    RESPONDING = "responding"
    COLLECTING = "collecting"
    ESCORTING = "escorting"
    RELEASING = "releasing"
    RETURNING = "returning"

    def __init__(self, x, y, speed, env, rng, blackboard=None,
                 n_civilians: int = 100, n_guides: int = 4):
        super().__init__(x, y, speed, env, rng)

        # Global scanning — no fixed territory. Blackboard coordinates Routers.
        self._blackboard = blackboard or RouterBlackboard()
        self._guide_id = id(self)  # unique ID for blackboard claims

        self._spawn_pos = np.array([x, y], dtype=float)  # remember spawn exit
        self._waypoints = []
        self._phase = self.SCANNING

        # Dynamic commitment ceiling (same derivation as Sweeper):
        # n* ∝ √(territory_population) from collection-escort cost optimization.
        # Same base for both policies — asymmetry carried by danger sensing:
        # Router uses PREDICTED danger (deliberative), Sweeper uses CURRENT (reactive).
        self._group = []
        self._base_max_group = round(3 * math.sqrt(n_civilians / max(1, n_guides)))
        self._floor = 5
        self._min_group = 5   # Don't waste trips on 1-2 people
        self._cached_effective_max = self._base_max_group
        self._cached_max_tick = -1

        # Congestion data
        self._exit_counts = {}
        self._density_grid = None
        self._eval_interval = 20
        self._eval_counter = 0

        # Escort target
        self._escort_exit = None

        # Releasing state
        self._release_wait = 0
        self._release_max_wait = 50

        # Collecting wait
        self._collect_wait = 0
        self._max_collect_wait = 30

        # Internal tick counter + danger reroute cooldown
        self._tick = 0
        self._danger_reroute_tick = -999

        # Adaptive search: DBSCAN → individual fallback (two-phase strategy)
        # After N consecutive scans with no clusters, switch to individual search
        self._no_cluster_count = 0
        self._no_cluster_threshold = 3  # 3 failed scans × 20 ticks = ~60 ticks

        # Start scanning immediately from spawn position (no territory walk)

    def _effective_max_group(self) -> int:
        """Dynamic commitment ceiling via T-S fuzzy inference (deliberative).

        Same first-principles derivation as Sweeper (n* ∝ √territory), but
        uses PREDICTED danger 30 ticks ahead via predict_spread() — this is
        the hybrid reactive-deliberative architecture (MPC-inspired receding
        horizon risk perception). Router foresees danger and reduces commitment
        before it arrives; Sweeper only reacts to current conditions.

        Takagi-Sugeno (1985) zero-order fuzzy with triangular membership:
            effective = base − d × (base − floor)
        """
        if self._cached_max_tick == self._tick:
            return self._cached_effective_max

        fire_sys = getattr(self, "_fire_system", None)
        smoke_sys = getattr(self, "_smoke_system", None)

        if fire_sys is None and smoke_sys is None:
            self._cached_effective_max = self._base_max_group
            self._cached_max_tick = self._tick
            return self._base_max_group

        # Predict danger 30 ticks ahead (deliberative planning)
        HORIZON = 30
        predicted_fg = None
        predicted_sg = None
        if fire_sys is not None:
            predicted_fg = predict_spread(
                fire_sys.density, fire_sys._burnable,
                fire_sys._fire_cells, fire_sys.spread_rate, HORIZON
            )
        if smoke_sys is not None:
            predicted_sg = predict_spread(
                smoke_sys.density, smoke_sys._smokeable,
                smoke_sys._fire_cells, smoke_sys.spread_rate, HORIZON
            )

        gr, gc = getattr(self, "_last_grid_pos", (0, 0))
        grid = predicted_fg if predicted_fg is not None else predicted_sg
        rows, cols = grid.shape

        peak_fire = 0.0
        peak_smoke = 0.0
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                nr, nc = gr + dr, gc + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if predicted_fg is not None and predicted_fg[nr, nc] > peak_fire:
                        peak_fire = predicted_fg[nr, nc]
                    if predicted_sg is not None and predicted_sg[nr, nc] > peak_smoke:
                        peak_smoke = predicted_sg[nr, nc]

        # Normalize to [0, 1] using existing emergency thresholds
        fire_factor = min(1.0, peak_fire / 0.1)
        smoke_factor = min(1.0, peak_smoke / 0.4)
        d = max(fire_factor, smoke_factor)

        base = self._base_max_group
        effective = max(self._floor, round(base - d * (base - self._floor)))

        self._cached_effective_max = effective
        self._cached_max_tick = self._tick
        return effective

    def update(self, env, smoke, fire, civilians):
        super().update(env, smoke, fire, civilians)
        if not self.alive:
            self._release_group_wandering()
            return
        if self._do_self_evacuation(env):
            self._release_group_wandering()
            return

        # Store references for deliberative planning (fire spread prediction)
        self._fire_system = fire
        self._smoke_system = smoke

        self._tick += 1

        # Publish current position to blackboard so other Routers can apply
        # distance-aware hotspot scoring (inter-guide spread via cumulative
        # repulsion field — see RouterBlackboard.get_other_positions).
        self._blackboard.update_position(self._guide_id, self.pos, self._tick)

        # A1 — exit coordination. While escorting, publish my destination
        # and group size so other Routers' exit scoring sees "N civilians
        # incoming from guide G" and can prefer less-targeted exits,
        # preventing independent-decision pile-ups at the same exit.
        if self._phase == self.ESCORTING and self._escort_exit is not None:
            self._blackboard.update_exit_claim(
                self._guide_id, self._escort_exit,
                len(self._group), self._tick,
            )

        # Update density + congestion data
        self._eval_counter += 1
        if self._eval_counter >= self._eval_interval:
            self._eval_counter = 0
            self._density_grid = build_density_grid(env, civilians)
            self._update_exit_counts(civilians)

        # Router does NOT broadcast — it physically escorts.
        # This is the key difference from Beacon (which broadcasts).

        # Clean group
        self._clean_group()

        # Phase FSM
        if self._phase == self.SCANNING:
            self._do_scanning(env, smoke, civilians)
        elif self._phase == self.RESPONDING:
            self._do_responding(env, smoke, civilians)
        elif self._phase == self.COLLECTING:
            self._do_collecting(env, smoke, civilians)
        elif self._phase == self.ESCORTING:
            self._do_escorting(env, smoke, civilians)
        elif self._phase == self.RELEASING:
            self._do_releasing(env, smoke)
        elif self._phase == self.RETURNING:
            self._do_returning(env, smoke, civilians)

    # ── Phase behaviors ──

    def _do_scanning(self, env, smoke, civilians):
        # DANGER CHECK: if guide has ANY civilians and smoke/fire nearby,
        # escort immediately — don't let them die waiting for min_group.
        # Identical pattern to _do_collecting and _do_returning.
        if self._group:
            danger = False
            sg = getattr(self, "_smoke_grid", None)
            fg = getattr(self, "_fire_grid", None)
            rows, cols = env.grid.shape
            gr, gc = env.world_to_grid(self.pos[0], self.pos[1])
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = gr + dr, gc + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if fg is not None and fg[nr, nc] > 0.1:
                            danger = True
                            break
                        if sg is not None and sg[nr, nc] > 0.4:
                            danger = True
                            break
                if danger:
                    break
            if not danger:
                for civ in self._group:
                    if smoke.get_density_at(civ.pos[0], civ.pos[1]) > 0.3:
                        danger = True
                        break
            if danger:
                self._start_escort(env, smoke, civilians)
                return

        if self._waypoints:
            self._follow_path(env)
            self._recruit_nearby(env, smoke, civilians)
            # max_group reached while walking → escort immediately
            if len(self._group) >= self._effective_max_group():
                self._start_escort(env, smoke, civilians)
            return
        self.vel = np.zeros(2)
        # Scan for congestion hotspot in territory
        hotspot = self._find_hotspot(env, smoke, civilians)
        if hotspot is not None:
            fg = getattr(self, "_fire_grid", None)
            sg = getattr(self, "_smoke_grid", None)
            danger = self._build_solo_danger_grid(fg, sg)
            self._waypoints = bfs_path(env, self.pos, hotspot, fire_grid=danger)
            self._phase = self.RESPONDING
        elif len(self._group) >= self._min_group:
            self._start_escort(env, smoke, civilians)
        elif not self._waypoints:
            # No hotspot found. Two cases to handle:
            # 1) Safety net: if we somehow still have a group here, escort
            #    them (shouldn't happen after the _do_collecting fix, but
            #    guard anyway — NEVER leave civilians attached to an idle guide).
            # 2) Mission complete: if no civilian anywhere needs help,
            #    self-evacuate (walk to nearest exit and leave the building).
            from evacsim.civilian import (
                IDLE, REACTING, WANDERING, FOLLOWING, FLEEING_ROOM,
                WALL_FOLLOWING,
            )
            NEEDS_HELP = (IDLE, REACTING, WANDERING, FOLLOWING,
                          FLEEING_ROOM, WALL_FOLLOWING)

            # Safety net: if any civilians are still attached to this guide,
            # escort them to an exit rather than leave them as passengers.
            if self._group:
                self._start_escort(env, smoke, civilians)
                return

            any_needs_help = any(c.state in NEEDS_HELP for c in civilians)
            if not any_needs_help:
                # Mission complete — self-evacuate through nearest exit
                self._start_self_evacuation(env)
            else:
                # Still someone to help but no cluster — stay put, rescan next cycle.
                self.vel = np.zeros(2)

    def _do_responding(self, env, smoke=None, civilians=None):
        # max_group reached while walking to hotspot → escort immediately
        if len(self._group) >= self._effective_max_group():
            self._start_escort(env, smoke, civilians)
            return

        # DANGER CHECK: if group members in smoke/fire → emergency escort.
        # Same pattern as _do_scanning, _do_collecting, _do_returning.
        if self._group and smoke:
            danger = False
            sg = getattr(self, "_smoke_grid", None)
            fg = getattr(self, "_fire_grid", None)
            rows, cols = env.grid.shape
            gr, gc = env.world_to_grid(self.pos[0], self.pos[1])
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = gr + dr, gc + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if fg is not None and fg[nr, nc] > 0.1:
                            danger = True
                            break
                        if sg is not None and sg[nr, nc] > 0.4:
                            danger = True
                            break
                if danger:
                    break
            if not danger:
                for civ in self._group:
                    if smoke.get_density_at(civ.pos[0], civ.pos[1]) > 0.3:
                        danger = True
                        break
            if danger:
                self._start_escort(env, smoke, civilians)
                return

        if not self._waypoints:
            self._phase = self.COLLECTING
            self._collect_wait = 0
            self.vel = np.zeros(2)
        else:
            # Safety check: is our destination (hotspot) now dangerous?
            # FIX 1: lower smoke threshold when carrying civilians (they're vulnerable)
            destination = self._waypoints[-1]
            dr, dc = env.world_to_grid(destination[0], destination[1])
            fg = getattr(self, "_fire_grid", None)
            sg = getattr(self, "_smoke_grid", None)
            rows, cols = env.grid.shape
            smoke_limit = 0.3 if self._group else 0.5
            if fg is not None and 0 <= dr < rows and 0 <= dc < cols and fg[dr, dc] > 0.2:
                self._waypoints.clear()
                if self._group:
                    # Have civilians — escort them to safety instead of scanning
                    self._start_escort(env, smoke, civilians)
                else:
                    self._phase = self.SCANNING
                    self.vel = np.zeros(2)
                return
            if sg is not None and 0 <= dr < rows and 0 <= dc < cols and sg[dr, dc] > smoke_limit:
                self._waypoints.clear()
                if self._group:
                    self._start_escort(env, smoke, civilians)
                else:
                    self._phase = self.SCANNING
                    self.vel = np.zeros(2)
                return

            self._follow_path(env)
            if smoke and civilians:
                self._recruit_nearby(env, smoke, civilians)
                # FIX 3: also check max_group after recruiting
                if len(self._group) >= self._effective_max_group():
                    self._start_escort(env, smoke, civilians)

    def _do_collecting(self, env, smoke, civilians):
        """At hotspot: recruit civilians into group. Soft limit — collect
        as many as possible, but start escorting once we have enough or timeout.
        Includes danger detection (like Sweeper): if fire/smoke nearby,
        escort immediately regardless of group size."""
        # max_group reached → escort immediately
        if len(self._group) >= self._effective_max_group():
            self._start_escort(env, smoke, civilians)
            return

        # DANGER CHECK: fire/smoke near guide or group members (Sweeper pattern)
        if self._group:
            danger = False
            sg = getattr(self, "_smoke_grid", None)
            fg = getattr(self, "_fire_grid", None)
            rows, cols = env.grid.shape

            # Check 2-cell radius around guide (proactive — detect approaching hazards)
            gr, gc = env.world_to_grid(self.pos[0], self.pos[1])
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = gr + dr, gc + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if fg is not None and fg[nr, nc] > 0.1:
                            danger = True
                            break
                        if sg is not None and sg[nr, nc] > 0.4:
                            danger = True
                            break
                if danger:
                    break

            # Check if any group member is in smoke (they don't have masks)
            if not danger:
                for civ in self._group:
                    if smoke.get_density_at(civ.pos[0], civ.pos[1]) > 0.3:
                        danger = True
                        break

            if danger:
                self._start_escort(env, smoke, civilians)
                return

        self.vel = np.zeros(2)
        self._collect_wait += 1
        self._recruit_nearby(env, smoke, civilians)

        # Start escorting when: enough people OR timeout
        has_enough = len(self._group) >= self._effective_max_group()
        timed_out = self._collect_wait >= self._max_collect_wait
        if has_enough or timed_out:
            if self._group:
                if len(self._group) >= self._min_group:
                    self._start_escort(env, smoke, civilians)
                else:
                    # Small group — check if anyone nearby is worth waiting for
                    recruitable = self._count_recruitable(env, smoke, civilians)
                    if recruitable == 0:
                        # Nobody nearby — escort what we have. Duty of care
                        # over efficiency: never leave civilians as passengers
                        # waiting for more. Real fire marshals escort every
                        # person they've committed to, regardless of group size.
                        self._start_escort(env, smoke, civilians)
                    # else: keep collecting (reset wait timer for one more round)
                    # Don't reset indefinitely — if still small after second wait, escort
                    elif self._collect_wait >= self._max_collect_wait * 2:
                        # Still can't reach min_group after two waits — escort
                        # what we have rather than hold them longer.
                        self._start_escort(env, smoke, civilians)
            else:
                # No civilians left at collection point — rescan immediately
                self._blackboard.release(self._guide_id)
                self._phase = self.SCANNING
                self.vel = np.zeros(2)

    def _do_escorting(self, env, smoke, civilians):
        """Lead group to optimal exit. KEY: always pick up civilians on the
        way — no group limit while escorting. Maximize survivors per trip.

        Periodically re-evaluates exit choice during escort. If congestion
        shifts significantly (a much better exit appears), re-routes.
        This prevents committing to a bad exit for the entire trip."""
        if not self._group:
            # Everyone died — abort escort, rescan immediately
            self._blackboard.release(self._guide_id)
            self._phase = self.SCANNING
            self.vel = np.zeros(2)
            return

        # IMMEDIATE DANGER CHECK with cooldown: if guide or group members are
        # in/near fire, reroute to safest exit. Cooldown prevents recomputing
        # BFS every tick while in the danger zone (10-tick grace period).
        if self._tick - self._danger_reroute_tick >= 10:
            fg = getattr(self, "_fire_grid", None)
            sg = getattr(self, "_smoke_grid", None)
            guide_in_danger = False
            gr, gc = env.world_to_grid(self.pos[0], self.pos[1])
            rows, cols = env.grid.shape
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = gr + dr, gc + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if fg is not None and fg[nr, nc] > 0.1:
                            guide_in_danger = True
                            break
                        if sg is not None and sg[nr, nc] > 0.4:
                            guide_in_danger = True
                            break
                if guide_in_danger:
                    break
            if not guide_in_danger:
                for civ in self._group:
                    if smoke.get_density_at(civ.pos[0], civ.pos[1]) > 0.3:
                        guide_in_danger = True
                        break
            if guide_in_danger:
                self._danger_reroute_tick = self._tick  # Start cooldown
                if fg is not None and sg is not None:
                    danger = np.maximum(fg, sg * 1.0)
                elif fg is not None:
                    danger = fg
                elif sg is not None:
                    danger = sg * 1.0
                else:
                    danger = None
                best_exit, best_score = self._escort_exit, float("inf")
                for ex, ey in self.known_exits:
                    dist = np.hypot(self.pos[0] - ex, self.pos[1] - ey)
                    # A1 — effective congestion = current queue + incoming
                    # from OTHER Routers (self excluded). Stops emergency
                    # rerouting from stampeding everyone to the same exit.
                    current_queue = self._exit_counts.get((ex, ey), 0)
                    incoming = self._blackboard.get_incoming_count(
                        [ex, ey], self._tick,
                        exclude_guide=self._guide_id,
                    )
                    cong = (current_queue + incoming) * 30.0
                    s_pen = smoke.get_density_at(ex, ey) * 200.0
                    f_pen = 0
                    if fg is not None:
                        er, ec = env.world_to_grid(ex, ey)
                        if 0 <= er < fg.shape[0] and 0 <= ec < fg.shape[1]:
                            f_pen = fg[er, ec] * 500.0
                    score = dist + cong + s_pen + f_pen
                    if score < best_score:
                        best_score = score
                        best_exit = (ex, ey)
                self._escort_exit = best_exit
                # FIX 2 — use Dijkstra with safety validation instead of
                # plain BFS. BFS only checks "passable" (≤ 0.3) and picks
                # the shortest such path; Dijkstra's weighted cost prefers
                # genuinely cleaner routes, and the validator rejects any
                # path whose waypoints exceed the danger threshold.
                dij_path = dijkstra_congestion_path(
                    env, self.pos, np.array(best_exit),
                    fire_grid=fg, smoke_grid=sg,
                    civilian_density=self._density_grid,
                )
                dij_safe = True
                if dij_path and danger is not None:
                    for wp in dij_path:
                        wr, wc = env.world_to_grid(wp[0], wp[1])
                        if (0 <= wr < danger.shape[0]
                                and 0 <= wc < danger.shape[1]):
                            if danger[wr, wc] > 0.3:
                                dij_safe = False
                                break
                if dij_path and dij_safe:
                    self._waypoints = dij_path
                else:
                    # BFS fallback with danger grid (empty OR unsafe)
                    self._waypoints = bfs_path(
                        env, self.pos, np.array(best_exit),
                        fire_grid=danger,
                    )

        self._recruit_nearby_unlimited(env, smoke, civilians)

        # Re-evaluate exit every eval_interval ticks during escort.
        # If a significantly better exit appears (>30% better score),
        # switch target and recompute Dijkstra path.
        # COMMITMENT ZONE: if already close to current exit, do not switch —
        # civilians in FOLLOWING_GUIDE state pass through exits without
        # escaping (wrong state), so switching near the exit causes them to
        # walk past and take extra smoke damage heading to a new exit.
        near_current_exit = (
            self._escort_exit is not None
            and np.linalg.norm(self.pos - np.array(self._escort_exit))
                < env.cell_size * 5
        )
        if (
            self._eval_counter == 0
            and self._escort_exit is not None
            and not near_current_exit
        ):
            fg = getattr(self, "_fire_grid", None)
            sg = getattr(self, "_smoke_grid", None)
            # Score current exit
            cx, cy = self._escort_exit
            cur_dist = np.hypot(self.pos[0] - cx, self.pos[1] - cy)
            # A1 — effective congestion = current queue + incoming from
            # OTHER Routers (self excluded, so our own group is not counted
            # against our current target).
            cur_queue = self._exit_counts.get((cx, cy), 0)
            cur_incoming = self._blackboard.get_incoming_count(
                [cx, cy], self._tick, exclude_guide=self._guide_id,
            )
            cur_cong = (cur_queue + cur_incoming) * 30.0
            cur_smoke = smoke.get_density_at(cx, cy) * 200.0
            cur_score = cur_dist + cur_cong + cur_smoke

            # Check all exits for a better option
            for ex, ey in self.known_exits:
                if (ex, ey) == self._escort_exit:
                    continue
                dist = np.hypot(self.pos[0] - ex, self.pos[1] - ey)
                # A1 — same effective-congestion adjustment for candidates
                cand_queue = self._exit_counts.get((ex, ey), 0)
                cand_incoming = self._blackboard.get_incoming_count(
                    [ex, ey], self._tick, exclude_guide=self._guide_id,
                )
                cong = (cand_queue + cand_incoming) * 30.0
                s_pen = smoke.get_density_at(ex, ey) * 200.0
                f_pen = 0
                if fg is not None:
                    er, ec = env.world_to_grid(ex, ey)
                    if 0 <= er < fg.shape[0] and 0 <= ec < fg.shape[1]:
                        f_pen = fg[er, ec] * 500.0
                new_score = dist + cong + s_pen + f_pen
                # Only switch if >30% better (avoid oscillation)
                if new_score < cur_score * 0.7:
                    self._escort_exit = (ex, ey)
                    # Build danger grid up-front — used both by the safety
                    # validator below and by the BFS fallback.
                    if fg is not None and sg is not None:
                        danger = np.maximum(fg, sg * 1.0)
                    elif fg is not None:
                        danger = fg
                    elif sg is not None:
                        danger = sg * 1.0
                    else:
                        danger = None

                    dij_path = dijkstra_congestion_path(
                        env, self.pos, np.array((ex, ey)),
                        fire_grid=fg, smoke_grid=sg,
                        civilian_density=self._density_grid,
                    )

                    # FIX 1 — safety validation on the re-evaluation path,
                    # matching _start_escort's pattern. Dijkstra's weighted
                    # cost can still pick a shorter path through moderate
                    # smoke over a clean longer path; the validator rejects
                    # any route with a waypoint in danger > 0.3 and falls
                    # back to BFS with the danger grid.
                    dij_safe = True
                    if dij_path and danger is not None:
                        for wp in dij_path:
                            wr, wc = env.world_to_grid(wp[0], wp[1])
                            if (0 <= wr < danger.shape[0]
                                    and 0 <= wc < danger.shape[1]):
                                if danger[wr, wc] > 0.3:
                                    dij_safe = False
                                    break

                    if dij_path and dij_safe:
                        self._waypoints = dij_path
                    else:
                        # BFS fallback with danger grid (empty OR unsafe)
                        self._waypoints = bfs_path(
                            env, self.pos, np.array((ex, ey)),
                            fire_grid=danger,
                        )
                    break  # Switch once per re-evaluation

        if not self._waypoints:
            if self._escort_exit is not None:
                dist = np.linalg.norm(self.pos - np.array(self._escort_exit))
                if dist < env.cell_size * 2:
                    # A1 — arrived; group is now at the exit (joining the
                    # queue), no longer "incoming." Release the exit claim
                    # so other Routers stop penalising this exit.
                    self._blackboard.release_exit_claim(self._guide_id)
                    self._phase = self.RELEASING
                    self._release_wait = 0
                    self.vel = np.zeros(2)
                    return
                # STUCK FIX: waypoints empty but not near exit.
                # Recompute a path so the guide doesn't stand still mid-escort.
                # FIX 2 — upgraded from plain BFS to Dijkstra with safety
                # validation (matches _start_escort and re-eval pattern):
                # Dijkstra weights smoke/fire/density in cost; validator
                # rejects paths with any waypoint in danger > 0.3; BFS is
                # the last-resort fallback.
                fg = getattr(self, "_fire_grid", None)
                sg = getattr(self, "_smoke_grid", None)
                if fg is not None and sg is not None:
                    danger = np.maximum(fg, sg * 1.0)
                elif fg is not None:
                    danger = fg
                elif sg is not None:
                    danger = sg * 1.0
                else:
                    danger = None

                dij_path = dijkstra_congestion_path(
                    env, self.pos, np.array(self._escort_exit),
                    fire_grid=fg, smoke_grid=sg,
                    civilian_density=self._density_grid,
                )
                dij_safe = True
                if dij_path and danger is not None:
                    for wp in dij_path:
                        wr, wc = env.world_to_grid(wp[0], wp[1])
                        if (0 <= wr < danger.shape[0]
                                and 0 <= wc < danger.shape[1]):
                            if danger[wr, wc] > 0.3:
                                dij_safe = False
                                break
                if dij_path and dij_safe:
                    self._waypoints = dij_path
                else:
                    self._waypoints = bfs_path(
                        env, self.pos, np.array(self._escort_exit),
                        fire_grid=danger,
                    )
            if not self._waypoints:
                self.vel = np.zeros(2)
        else:
            self._follow_path(env)

    def _do_releasing(self, env, smoke=None):
        """Wait at exit for group to arrive and escape."""
        from evacsim.civilian import FOLLOWING_GUIDE, WANDERING

        self.vel = np.zeros(2)
        self._release_wait += 1

        still = []
        for civ in self._group:
            if civ.state != FOLLOWING_GUIDE:
                continue
            dist_exit = np.linalg.norm(civ.pos - np.array(self._escort_exit))
            dist_guide = np.linalg.norm(civ.pos - self.pos)
            if (
                dist_exit < env.cell_size * 5
                or dist_guide < env.cell_size * 3
            ):
                civ._set_exit_target_force(env, self._escort_exit)
            else:
                still.append(civ)
        self._group = still

        if not self._group or self._release_wait >= self._release_max_wait:
            # Timeout: give stragglers exit target instead of leaving them lost
            for civ in self._group:
                if civ.state == FOLLOWING_GUIDE:
                    civ._set_exit_target_force(env, self._escort_exit)
            self._group.clear()
            # Release blackboard claim — escort complete
            self._blackboard.release(self._guide_id)
            # Scan immediately from current position (no walking "home")
            self._phase = self.SCANNING
            self.vel = np.zeros(2)

    def _do_returning(self, env, smoke=None, civilians=None):
        """Return phase — scan for new hotspots while moving.
        If no waypoints, transition directly to SCANNING.
        Danger logic: if smoke/fire threatens group members, emergency escort."""
        if not smoke or not civilians:
            if not self._waypoints:
                self._phase = self.SCANNING
                self.vel = np.zeros(2)
            else:
                self._follow_path(env)
            return

        # Max group → escort immediately
        if len(self._group) >= self._effective_max_group():
            self._start_escort(env, smoke, civilians)
            return

        # DANGER CHECK (identical to Sweeper _do_returning):
        # if group members in smoke or fire/smoke near guide → emergency escort
        if self._group:
            danger = False
            # Check group members first (they're vulnerable, no masks)
            for civ in self._group:
                if smoke.get_density_at(civ.pos[0], civ.pos[1]) > 0.3:
                    danger = True
                    break
            # Check guide surroundings (2-cell radius)
            if not danger:
                sg = getattr(self, "_smoke_grid", None)
                fg = getattr(self, "_fire_grid", None)
                if sg is not None or fg is not None:
                    gr, gc = env.world_to_grid(self.pos[0], self.pos[1])
                    rows, cols = env.grid.shape
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            nr, nc = gr + dr, gc + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                if fg is not None and fg[nr, nc] > 0.1:
                                    danger = True
                                    break
                                if sg is not None and sg[nr, nc] > 0.4:
                                    danger = True
                                    break
                        if danger:
                            break
            if danger:
                self._start_escort(env, smoke, civilians)
                return

        # Scan for hotspots periodically while walking home (throttled —
        # DBSCAN is O(n²), don't run every tick). eval_counter resets every
        # eval_interval ticks in update(), so this fires every ~20 ticks.
        if self._eval_counter == 0:
            hotspot = self._find_hotspot(env, smoke, civilians)
        else:
            hotspot = None
        if hotspot is not None:
            fg = getattr(self, "_fire_grid", None)
            sg = getattr(self, "_smoke_grid", None)
            danger = self._build_solo_danger_grid(fg, sg)
            self._waypoints = bfs_path(env, self.pos, hotspot, fire_grid=danger)
            self._phase = self.RESPONDING
            return

        if not self._waypoints:
            # No destination — transition to scanning from current position
            self._blackboard.release(self._guide_id)
            self._phase = self.SCANNING
            self.vel = np.zeros(2)
        else:
            self._follow_path(env)
            # Recruit while moving
            self._recruit_nearby(env, smoke, civilians)
            # Only divert to escort if we have a decent group (min threshold)
            if len(self._group) >= self._min_group:
                self._start_escort(env, smoke, civilians)

    # ── Intelligence methods ──

    def _find_hotspot(self, env, smoke, civilians):
        """Two-phase adaptive search using DBSCAN (Ester et al., 1996).

        Phase 1 — DBSCAN cluster detection:
            Uses density-based spatial clustering to identify civilian groups.
            Finds natural clusters of arbitrary shape without boundary artifacts.
            Parameters: epsilon = 3 cells (~9m, room-sized), min_points = 3.
            Global scanning — no territory filter. Blackboard coordination
            prevents multiple Routers claiming the same hotspot.

        Phase 2 — Individual civilian search (fallback):
            When DBSCAN yields no clusters for N consecutive scans, the
            population has dispersed. Switches to targeting the nearest
            individual unhelped civilian. Prevents abandoning isolated
            survivors in late-stage evacuation.

        Cluster scoring: size × (1 - centroid_smoke × 0.5).
        Skips targets in fire or heavy smoke (unsafe to reach).
        Skips hotspots already claimed by other Routers (blackboard)."""
        from evacsim.civilian import ESCAPED, DEAD, MOVING_TO_EXIT, FOLLOWING_GUIDE, QUEUED

        # 1. Filter to civilians needing help — GLOBAL scan (no territory filter)
        candidates = []
        for civ in civilians:
            if civ.state in (ESCAPED, DEAD, MOVING_TO_EXIT, FOLLOWING_GUIDE, QUEUED):
                continue
            if civ.assigned_guide is not None:
                continue
            candidates.append(civ)

        if not candidates:
            return None

        fg = getattr(self, "_fire_grid", None)
        sg = getattr(self, "_smoke_grid", None)
        smoke_limit = 0.3 if self._group else 0.6
        rows, cols = env.grid.shape

        # 2. Phase 1: DBSCAN cluster detection
        if len(candidates) >= 3:
            positions = [c.pos for c in candidates]
            epsilon = env.cell_size * 3  # 3 cells (~9m) — room-sized
            clusters = dbscan_cluster(positions, epsilon, min_points=3)

            # Pre-compute data for inter-guide spread (cumulative repulsion
            # field). Each other Router's current position contributes an
            # inverse-distance repulsion; when multiple guides are clustered
            # near a candidate hotspot, their repulsions STACK — biasing this
            # guide toward distant, uncovered areas. Potential-field method,
            # lecture reference: intelligentAgents_coordination.pptx
            # (cell division vs. decentralised spreading via potentials).
            other_positions = self._blackboard.get_other_positions(
                exclude_guide=self._guide_id, current_tick=self._tick,
            )
            # Map diagonal used to normalise distances to [0, 1] so the
            # formula adapts to any map size.
            map_diag = float(np.hypot(
                rows * env.cell_size, cols * env.cell_size
            ))
            # Decay constant: repulsion drops to ~50% at 10% of map diag.
            REPULSION_DECAY = 10.0
            # Max bonus: a hotspot maximally far from all other guides
            # receives a 1.0 + MAX_SPREAD_BONUS score multiplier.
            # 0.3 → cluster ~23% smaller can be preferred if it's far away.
            MAX_SPREAD_BONUS = 0.3

            # Score each cluster and pick the best
            best_pos = None
            best_score = 0

            for cluster_indices in clusters:
                cx = sum(positions[i][0] for i in cluster_indices) / len(cluster_indices)
                cy = sum(positions[i][1] for i in cluster_indices) / len(cluster_indices)

                # Safety: skip clusters in fire/heavy smoke
                cr, cc = env.world_to_grid(cx, cy)
                if not (0 <= cr < rows and 0 <= cc < cols):
                    continue
                if fg is not None and fg[cr, cc] > 0.2:
                    continue
                if sg is not None and sg[cr, cc] > smoke_limit:
                    continue

                # Blackboard: skip hotspots already claimed by other Routers
                if self._blackboard.is_claimed(
                    [cx, cy], self._tick, exclude_guide=self._guide_id
                ):
                    continue

                size = len(cluster_indices)
                centroid_smoke = smoke.get_density_at(cx, cy) if smoke else 0
                safety_factor = 1.0 - centroid_smoke * 0.5

                # Cumulative repulsion: sum inverse-distance contributions
                # from every other Router's current position.
                #   repulsion_i = 1 / (1 + REPULSION_DECAY * d_norm_i)
                # where d_norm_i = distance / map_diagonal, in [0, 1].
                # Stacking is natural: 3 guides at distance d contribute 3×
                # the repulsion of 1 guide at that distance.
                if other_positions:
                    repulsion = 0.0
                    for p in other_positions:
                        d_norm = float(np.hypot(cx - p[0], cy - p[1])) / map_diag
                        repulsion += 1.0 / (1.0 + REPULSION_DECAY * d_norm)
                    # Normalise by number of other guides so the value is in
                    # [0, 1] regardless of guide count. 1.0 = maximum repulsion
                    # (all other guides adjacent). 0 = no repulsion.
                    normalized = repulsion / len(other_positions)
                    spread_bonus = 1.0 + MAX_SPREAD_BONUS * (1.0 - normalized)
                else:
                    spread_bonus = 1.0  # alone — no spread bias needed

                score = size * safety_factor * spread_bonus

                if score > best_score:
                    best_score = score
                    best_pos = np.array([cx, cy])

            if best_pos is not None:
                self._no_cluster_count = 0  # Reset fallback counter
                # Claim this hotspot on the blackboard
                self._blackboard.claim(self._guide_id, best_pos, self._tick)
                return self._snap_to_walkable(env, best_pos)

        # No clusters found — increment fallback counter
        self._no_cluster_count += 1

        # 3. Phase 2: Individual civilian search (fallback)
        # After N consecutive scans with no clusters, target nearest individual.
        if self._no_cluster_count >= self._no_cluster_threshold:
            best_civ = None
            best_dist = float("inf")
            for civ in candidates:
                # Safety: skip civilians in fire/heavy smoke zones
                cr, cc = env.world_to_grid(civ.pos[0], civ.pos[1])
                if fg is not None and 0 <= cr < rows and 0 <= cc < cols and fg[cr, cc] > 0.2:
                    continue
                if sg is not None and 0 <= cr < rows and 0 <= cc < cols and sg[cr, cc] > smoke_limit:
                    continue
                dist = np.linalg.norm(self.pos - civ.pos)
                if dist < best_dist:
                    best_dist = dist
                    best_civ = civ

            if best_civ is not None:
                return self._snap_to_walkable(env, best_civ.pos.copy())

        return None

    def _snap_to_walkable(self, env, pos):
        """Snap a position to the nearest walkable cell if it's in a wall."""
        gr, gc = env.world_to_grid(pos[0], pos[1])
        rows, cols = env.grid.shape
        if 0 <= gr < rows and 0 <= gc < cols and env.grid[gr, gc] != CellType.WALL:
            return pos
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = gr + dr, gc + dc
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and env.grid[nr, nc] != CellType.WALL
                ):
                    return np.array(env.grid_to_world(nr, nc), dtype=float)
        return None

    def _recruit_nearby(self, env, smoke, civilians):
        """Recruit nearby civilians using visual range (Jin, 1978) + LOS."""
        if len(self._group) >= self._effective_max_group():
            return
        radius = self.get_visual_recruit_radius(smoke)
        from evacsim.civilian import (
            MOVING_TO_EXIT,
            ESCAPED,
            DEAD,
            IDLE,
            REACTING,
            FOLLOWING_GUIDE,
        )

        for civ in civilians:
            if len(self._group) >= self._effective_max_group():
                break
            if civ.state in (MOVING_TO_EXIT, ESCAPED, DEAD, FOLLOWING_GUIDE):
                continue
            if civ.assigned_guide is not None:
                continue
            dist = np.linalg.norm(civ.pos - self.pos)
            if dist < radius and self._has_line_of_sight(civ.pos, env):
                if civ.state == IDLE:
                    civ.state = REACTING
                    civ._reaction_counter = 0
                    civ._reaction_delay = min(civ._reaction_delay, 5)
                elif civ.state == REACTING:
                    civ._reaction_delay = min(civ._reaction_delay, 5)
                    continue
                else:
                    self._group.append(civ)
                    civ.assigned_guide = self
                    civ.state = FOLLOWING_GUIDE

    def _recruit_nearby_unlimited(self, env, smoke, civilians):
        """Recruit ANY nearby civilian while escorting — no group limit.
        Uses Jin (1978) visual range + LOS. Picks up everyone on the way."""
        radius = self.get_visual_recruit_radius(smoke)
        from evacsim.civilian import (
            MOVING_TO_EXIT,
            ESCAPED,
            DEAD,
            IDLE,
            REACTING,
            FOLLOWING_GUIDE,
        )

        for civ in civilians:
            if civ.state in (MOVING_TO_EXIT, ESCAPED, DEAD, FOLLOWING_GUIDE):
                continue
            if civ.assigned_guide is not None:
                continue
            dist = np.linalg.norm(civ.pos - self.pos)
            if dist < radius and self._has_line_of_sight(civ.pos, env):
                if civ.state == IDLE:
                    civ.state = REACTING
                    civ._reaction_counter = 0
                    civ._reaction_delay = min(civ._reaction_delay, 5)
                elif civ.state == REACTING:
                    civ._reaction_delay = min(civ._reaction_delay, 5)
                    continue
                else:
                    self._group.append(civ)
                    civ.assigned_guide = self
                    civ.state = FOLLOWING_GUIDE

    def _count_recruitable(self, env, smoke, civilians):
        """Count civilians that can ACTUALLY be recruited (visual range + LOS).
        Must match what _recruit_nearby can reach — no false positives.
        Used in _do_collecting to decide: keep waiting or move on?"""
        radius = self.get_visual_recruit_radius(smoke)
        from evacsim.civilian import (
            MOVING_TO_EXIT,
            ESCAPED,
            DEAD,
            FOLLOWING_GUIDE,
        )

        count = 0
        for civ in civilians:
            if civ.state in (MOVING_TO_EXIT, ESCAPED, DEAD, FOLLOWING_GUIDE):
                continue
            if civ.assigned_guide is not None:
                continue
            dist = np.linalg.norm(civ.pos - self.pos)
            if dist < radius and self._has_line_of_sight(civ.pos, env):
                count += 1
        return count

    def _start_escort(self, env, smoke, civilians):
        """Deliberative escort planning: predict fire/smoke spread, then route.

        Hybrid reactive-deliberative architecture:
          - REACTIVE: uses current congestion data for exit selection
          - DELIBERATIVE: simulates fire/smoke CA forward to predict future
            danger, then plans path through corridors that will STILL be safe
            when the group passes through (not just safe NOW).

        This avoids the common failure mode where a corridor is clear when
        the escort starts but fills with smoke by the time the group arrives.
        """
        self._phase = self.ESCORTING
        fg = getattr(self, "_fire_grid", None)
        sg = getattr(self, "_smoke_grid", None)

        # ── DELIBERATIVE: predict fire/smoke spread ──
        # Estimate escort duration from straight-line distance
        # Use predicted danger for both exit scoring and path planning
        fire_sys = getattr(self, "_fire_system", None)
        smoke_sys = getattr(self, "_smoke_system", None)
        predicted_fg = fg
        predicted_sg = sg

        if fire_sys is not None and smoke_sys is not None:
            # Estimate ticks to reach each exit (rough: straight-line / group speed)
            avg_dist = np.mean([np.hypot(self.pos[0] - ex, self.pos[1] - ey)
                                for ex, ey in self.known_exits])
            group_speed = max(self.speed * 0.75, 0.1)
            est_ticks = int(avg_dist / group_speed)
            est_ticks = min(est_ticks, 200)  # Cap at 200 ticks

            if est_ticks > 10:  # Only predict if escort will take meaningful time
                predicted_fg = predict_spread(
                    fire_sys.density, fire_sys._burnable,
                    fire_sys._fire_cells, fire_sys.spread_rate, est_ticks)
                predicted_sg = predict_spread(
                    smoke_sys.density, smoke_sys._smokeable,
                    smoke_sys._fire_cells, smoke_sys.spread_rate, est_ticks)

        # ── Score each exit using PREDICTED conditions ──
        best_exit, best_score = None, float("inf")
        for ex, ey in self.known_exits:
            dist = np.hypot(self.pos[0] - ex, self.pos[1] - ey)
            # A1 — effective congestion = current queue + incoming from
            # OTHER Routers (self excluded). Router about to pick an exit
            # sees pending arrivals from peers and avoids pile-ups.
            current_queue = self._exit_counts.get((ex, ey), 0)
            incoming = self._blackboard.get_incoming_count(
                [ex, ey], self._tick, exclude_guide=self._guide_id,
            )
            congestion = (current_queue + incoming) * 30.0
            # Use predicted smoke/fire at exit (will it be safe when we arrive?)
            er, ec = env.world_to_grid(ex, ey)
            smoke_pen = 0
            fire_pen = 0
            if predicted_sg is not None and 0 <= er < predicted_sg.shape[0] and 0 <= ec < predicted_sg.shape[1]:
                smoke_pen = predicted_sg[er, ec] * 200.0
            if predicted_fg is not None and 0 <= er < predicted_fg.shape[0] and 0 <= ec < predicted_fg.shape[1]:
                fire_pen = predicted_fg[er, ec] * 500.0
            score = dist + congestion + smoke_pen + fire_pen
            if score < best_score:
                best_score = score
                best_exit = (ex, ey)

        self._escort_exit = best_exit
        if best_exit:
            # Build PREDICTED danger grid for path validation
            if predicted_fg is not None and predicted_sg is not None:
                danger = np.maximum(predicted_fg, predicted_sg * 1.0)
            elif predicted_fg is not None:
                danger = predicted_fg
            elif predicted_sg is not None:
                danger = predicted_sg * 1.0
            else:
                danger = None

            # Primary: Dijkstra with predicted density + congestion
            dij_path = dijkstra_congestion_path(
                env,
                self.pos,
                np.array(best_exit),
                fire_grid=predicted_fg,
                smoke_grid=predicted_sg,
                civilian_density=self._density_grid,
            )

            # Validate: reject Dijkstra path if it goes through predicted danger
            dij_safe = True
            if dij_path and danger is not None:
                for wp in dij_path:
                    wr, wc = env.world_to_grid(wp[0], wp[1])
                    if 0 <= wr < danger.shape[0] and 0 <= wc < danger.shape[1]:
                        if danger[wr, wc] > 0.3:
                            dij_safe = False
                            break

            if dij_path and dij_safe:
                self._waypoints = dij_path
            else:
                # BFS fallback with predicted danger grid
                self._waypoints = bfs_path(
                    env, self.pos, np.array(best_exit), fire_grid=danger
                )

    def _update_exit_counts(self, civilians):
        """Count civilians heading to each exit (confirmed + predicted).

        Confirmed: civilians in MOVING_TO_EXIT or QUEUED with a known target.
        Predicted: all other active civilians counted toward their nearest exit
        at half weight — they WILL likely go there eventually.

        This gives Router early congestion awareness even before civilians
        have found exits, enabling smarter exit selection from tick 1."""
        from evacsim.civilian import MOVING_TO_EXIT, QUEUED, ESCAPED, DEAD, FOLLOWING_GUIDE

        self._exit_counts = {(ex, ey): 0.0 for ex, ey in self.known_exits}
        for civ in civilians:
            if civ.state in (ESCAPED, DEAD):
                continue

            if civ.state in (MOVING_TO_EXIT, QUEUED) and civ._target_exit is not None:
                # Confirmed: heading to specific exit (weight 1.0)
                tx, ty = civ._target_exit
                best, best_d = None, float("inf")
                for ex, ey in self.known_exits:
                    d = np.hypot(tx - ex, ty - ey)
                    if d < best_d:
                        best_d = d
                        best = (ex, ey)
                if best and best_d < 20:
                    self._exit_counts[best] += 1.0
            elif civ.state != FOLLOWING_GUIDE and civ.assigned_guide is None:
                # Predicted: will likely go to nearest exit (weight 0.5)
                best, best_d = None, float("inf")
                for ex, ey in self.known_exits:
                    d = np.hypot(civ.pos[0] - ex, civ.pos[1] - ey)
                    if d < best_d:
                        best_d = d
                        best = (ex, ey)
                if best:
                    self._exit_counts[best] += 0.5

    def _build_solo_danger_grid(self, fg, sg):
        """Build a danger grid for solo movement (no group).
        Guides have masks so smoke is less dangerous — use 0.3x weight.
        Still avoid fire fully. This prevents guides repeatedly walking
        through smoke to territory center and dying."""
        if fg is not None and sg is not None:
            return np.maximum(fg, sg * 0.3)
        elif fg is not None:
            return fg
        elif sg is not None:
            return sg * 0.3
        return None

    def _release_group_wandering(self):
        from evacsim.civilian import WANDERING, FOLLOWING_GUIDE

        for civ in self._group:
            if civ.state == FOLLOWING_GUIDE:
                civ.assigned_guide = None
                civ.state = WANDERING
        self._group.clear()
        # Release blackboard claim (guide died or self-evacuating)
        self._blackboard.release(self._guide_id)

    def _clean_group(self):
        from evacsim.civilian import ESCAPED, DEAD

        self._group = [c for c in self._group if c.state not in (ESCAPED, DEAD)]


def spawn_router_guides(env, n_guides, speed=2.0, seed=77, n_civilians=100):
    """Spawn router guides at exits with shared blackboard coordination.

    No territory assignment — Routers scan globally and coordinate via
    blackboard to avoid claiming the same hotspot. Each Router spawns
    at a different exit for spatial distribution."""
    rng = np.random.default_rng(seed)
    exits = list(env.exits)

    # Shared blackboard for all Routers in this simulation
    blackboard = RouterBlackboard()

    guides = []
    for i in range(n_guides):
        ex, ey = exits[i % len(exits)]
        guides.append(RouterGuide(ex, ey, speed, env, rng, blackboard=blackboard,
                                  n_civilians=n_civilians, n_guides=n_guides))
    return guides
