"""RouterGuide and RouterBlackboard: hybrid reactive-deliberative policy.

The guide scans globally for clusters of civilians needing assistance,
collects them into a group, and escorts them to the exit chosen by a
multi-criteria score that combines distance, congestion, fire, and
smoke. Forecasting is used both for the dynamic commitment ceiling and
for path planning: the guide simulates the fire and smoke automata
forward by a horizon proportional to the expected escort time and
plans against those predicted grids rather than the current ones.

Coordination across multiple Routers goes through a shared blackboard
with three channels: hotspot claims, live positions (used for a
distance-based repulsion field that spreads guides out), and exit
claims (used to fold incoming civilians from peers into the exit
score, preventing pile-ups at the same exit).
"""

import math
import numpy as np
from evacsim.environment import CellType
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

    Three channels:

    1. CLAIMS. Routers write claimed hotspot positions when committing
       to a rescue. Other Routers skip claimed hotspots within a radius
       to avoid overlap. Claims auto-expire after CLAIM_EXPIRY ticks so
       that guide death or rerouting frees the hotspot. Claims persist
       longer than positions because they cover travel plus collection
       time, not just current location.

    2. POSITIONS. Each Router posts its current world position every
       tick. Other Routers read these positions when scoring hotspots
       and apply a distance-based repulsion: hotspots near clusters of
       other guides receive stacked penalties, encouraging
       decentralised spread without hard territory assignment.
       Positions expire faster than claims because they describe
       'where I am now', not 'where I am committed to going'.

    3. EXIT CLAIMS. While ESCORTING, each Router posts its destination
       exit and current group size. Peers fold this into the exit
       score as 'incoming traffic' to avoid independently picking the
       same cleanest exit and piling up. Refreshed every tick, so the
       short TTL only acts as silent-death cleanup.

    Implements decentralised coordination via shared spatial awareness,
    using an inter-agent potential-field approach.
    """

    CLAIM_EXPIRY = 60      # Ticks before a hotspot claim expires.
    POSITION_EXPIRY = 10   # Ticks before a position becomes stale.
    EXIT_CLAIM_EXPIRY = 10  # Refreshed each tick during ESCORTING; the
    # TTL only catches silent-death cleanup, matching POSITION_EXPIRY.

    def __init__(self):
        self._claims = {}       # guide_id -> (position, tick_claimed)
        self._positions = {}    # guide_id -> (position, tick_updated)
        self._exit_claims = {}  # guide_id -> (exit_pos, group_size, tick)

    # -- Hotspot claims --

    def claim(self, guide_id, position, tick):
        """Claim a hotspot position."""
        self._claims[guide_id] = (np.array(position), tick)

    def release(self, guide_id):
        """Release a claim and any associated exit claim.

        Called when the guide finishes escorting or dies. The exit
        claim is also cleared defensively: any cleanup path that
        releases a hotspot claim should also release the exit claim.
        """
        self._claims.pop(guide_id, None)
        self._exit_claims.pop(guide_id, None)

    def is_claimed(self, position, current_tick, exclude_guide=None, radius=30.0):
        """Return True if the position is claimed by another guide.

        Uses radius matching: positions within radius world units count
        as the same hotspot.
        """
        for gid, (pos, tick) in list(self._claims.items()):
            if gid == exclude_guide:
                continue
            if current_tick - tick > self.CLAIM_EXPIRY:
                del self._claims[gid]  # Expired.
                continue
            if np.linalg.norm(pos - np.array(position)) < radius:
                return True
        return False

    # -- Position channel (inter-guide spread) --

    def update_position(self, guide_id, position, tick):
        """Record the current position of a Router. Called every tick."""
        self._positions[guide_id] = (np.array(position, dtype=float), tick)

    def get_other_positions(self, exclude_guide, current_tick):
        """Return the fresh positions of other Routers.

        Stale positions are filtered out and lazily cleaned up so dead
        or departed guides do not keep holding territory.
        """
        result = []
        for gid, (pos, tick) in list(self._positions.items()):
            if gid == exclude_guide:
                continue
            if current_tick - tick > self.POSITION_EXPIRY:
                del self._positions[gid]  # Lazy cleanup of stale entries.
                continue
            result.append(pos)
        return result

    # -- Exit claims (destination coordination) --

    def update_exit_claim(self, guide_id, exit_pos, group_size, tick):
        """Refresh the exit claim. Called every tick while ESCORTING.

        Same pattern as update_position. If the guide dies silently the
        entry auto-expires after EXIT_CLAIM_EXPIRY ticks.
        """
        self._exit_claims[guide_id] = (
            np.array(exit_pos, dtype=float),
            int(group_size),
            tick,
        )

    def release_exit_claim(self, guide_id):
        """Release an exit claim on arrival at the exit.

        The group is now in the queue and no longer 'incoming', so
        peers should stop penalising this exit. This avoids
        double-counting in the brief window between arrival and
        natural TTL expiry.
        """
        self._exit_claims.pop(guide_id, None)

    def get_incoming_count(
        self, exit_pos, current_tick, exclude_guide=None, radius=15.0
    ):
        """Sum of group sizes across other Routers escorting toward this exit.

        Matching is positional within radius world units to handle
        minor coordinate variation. Used in exit scoring as the
        'incoming traffic' term so a newly arriving group sees
        current_queue + incoming and avoids piling up at exits already
        targeted by peers.
        """
        total = 0
        target = np.array(exit_pos, dtype=float)
        for gid, (pos, size, tick) in list(self._exit_claims.items()):
            if gid == exclude_guide:
                continue
            if current_tick - tick > self.EXIT_CLAIM_EXPIRY:
                del self._exit_claims[gid]  # Lazy cleanup.
                continue
            if np.linalg.norm(pos - target) < radius:
                total += size
        return total

    def reset(self):
        """Clear all blackboard state for a new simulation."""
        self._claims.clear()
        self._positions.clear()
        self._exit_claims.clear()


class RouterGuide(Guide):
    """Hybrid reactive-deliberative guide that uses congestion-aware routing.

    Routes civilians on the least-congested path to the least-congested
    exit, using forecast fire and smoke grids for both the dynamic
    commitment ceiling and the path-planning step. Each tick the guide
    is in one of six phases:

        1. SCANNING. Look for a hotspot of unhelped civilians using
           DBSCAN. If none found, transition to RESPONDING.
        2. RESPONDING. Walk toward the chosen hotspot.
        3. COLLECTING. At the hotspot, recruit civilians within the
           visual range. Move to ESCORTING when the group is large
           enough, the timeout fires, or no more civilians are
           recruitable.
        4. ESCORTING. Lead the group toward the chosen exit, picking
           up any civilian passed on the way (no group limit during
           escort), and re-evaluating the exit choice every twenty
           ticks if a substantially better option appears.
        5. RELEASING. Stand at the exit and forward civilians into the
           queue.
        6. RETURNING (kept for future phase reuse; the guide
           transitions directly back to SCANNING after RELEASING).

    Architectural elements that distinguish this policy from the
    others: a global hotspot scan rather than a fixed territory, a
    forecast-driven exit and path planner, density-weighted Dijkstra
    cost, and the three-channel blackboard described above.
    """

    SCANNING = "scanning"
    RESPONDING = "responding"
    COLLECTING = "collecting"
    ESCORTING = "escorting"
    RELEASING = "releasing"
    RETURNING = "returning"

    def __init__(
        self,
        x,
        y,
        speed,
        env,
        rng,
        blackboard=None,
        n_civilians: int = 100,
        n_guides: int = 4,
    ):
        super().__init__(x, y, speed, env, rng)

        # No fixed territory. Coordination goes through the blackboard.
        self._blackboard = blackboard or RouterBlackboard()
        self._guide_id = id(self)  # Unique ID for blackboard entries.

        self._spawn_pos = np.array([x, y], dtype=float)
        self._waypoints = []
        self._phase = self.SCANNING

        # Dynamic commitment ceiling. Same closed-form derivation as
        # Sweeper: minimising C(n) = P*beta/n + P*h*n yields an optimal
        # batch size proportional to sqrt(territory_population). The
        # asymmetry between policies is in danger sensing: this guide
        # samples PREDICTED danger (deliberative), Sweeper samples
        # current danger (reactive).
        self._group = []
        self._base_max_group = round(3 * math.sqrt(n_civilians / max(1, n_guides)))
        self._floor = 5
        self._min_group = 5  # Below this, escort overhead dominates.
        self._cached_effective_max = self._base_max_group
        self._cached_max_tick = -1

        # Congestion data.
        self._exit_counts = {}
        self._density_grid = None
        self._eval_interval = 20
        self._eval_counter = 0

        # Escort target.
        self._escort_exit = None

        # Releasing state.
        self._release_wait = 0
        self._release_max_wait = 50

        # Collecting wait.
        self._collect_wait = 0
        self._max_collect_wait = 30

        # Internal tick counter and danger-reroute cooldown.
        self._tick = 0
        self._danger_reroute_tick = -999

        # Adaptive search: DBSCAN with an individual-civilian fallback
        # after several consecutive empty scans.
        self._no_cluster_count = 0
        self._no_cluster_threshold = 3  # 3 empty scans x 20 ticks = 60 ticks.

    def _effective_max_group(self) -> int:
        """Return the dynamic commitment ceiling for the current tick.

        Same closed-form derivation as Sweeper, but danger is sampled
        from PREDICTED grids 30 ticks ahead via predict_spread rather
        than from the current grids. This is the deliberative
        component: the guide foresees danger and reduces commitment
        before it arrives.

        Zero-order fuzzy interpolation rule:
            effective = base - d * (base - floor)
        where d is the normalised peak danger in [0, 1]. Cached per
        tick because each phase method may call this several times.
        """
        if self._cached_max_tick == self._tick:
            return self._cached_effective_max

        fire_sys = getattr(self, "_fire_system", None)
        smoke_sys = getattr(self, "_smoke_system", None)

        if fire_sys is None and smoke_sys is None:
            self._cached_effective_max = self._base_max_group
            self._cached_max_tick = self._tick
            return self._base_max_group

        # Predict danger 30 ticks ahead.
        HORIZON = 30
        predicted_fg = None
        predicted_sg = None
        if fire_sys is not None:
            predicted_fg = predict_spread(
                fire_sys.density,
                fire_sys._burnable,
                fire_sys._fire_cells,
                fire_sys.spread_rate,
                HORIZON,
            )
        if smoke_sys is not None:
            predicted_sg = predict_spread(
                smoke_sys.density,
                smoke_sys._smokeable,
                smoke_sys._fire_cells,
                smoke_sys.spread_rate,
                HORIZON,
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

        # Normalise to [0, 1] using the existing emergency thresholds.
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

        # Cache references for the deliberative planner.
        self._fire_system = fire
        self._smoke_system = smoke

        self._tick += 1

        # Publish current position so peers can apply distance-based
        # hotspot scoring (see RouterBlackboard.get_other_positions).
        self._blackboard.update_position(self._guide_id, self.pos, self._tick)

        # While escorting, publish destination and group size so peers
        # see incoming traffic on this exit and can prefer alternatives.
        if self._phase == self.ESCORTING and self._escort_exit is not None:
            self._blackboard.update_exit_claim(
                self._guide_id,
                self._escort_exit,
                len(self._group),
                self._tick,
            )

        # Refresh density and exit-count data on the eval interval.
        self._eval_counter += 1
        if self._eval_counter >= self._eval_interval:
            self._eval_counter = 0
            self._density_grid = build_density_grid(env, civilians)
            self._update_exit_counts(civilians)

        # Router does not broadcast; it physically escorts. This is
        # the key behavioural difference from Beacon.

        self._clean_group()

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

    # -- Phase behaviours --

    def _do_scanning(self, env, smoke, civilians):
        # If the guide is carrying civilians and danger is nearby,
        # escort immediately rather than wait for min_group. Same
        # pattern as _do_collecting and _do_returning.
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
            # Reached max_group while walking. Escort immediately.
            if len(self._group) >= self._effective_max_group():
                self._start_escort(env, smoke, civilians)
            return
        self.vel = np.zeros(2)
        # Scan for a congestion hotspot.
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
            # No hotspot found. Two cases:
            # 1. Safety net: civilians are still attached to this guide.
            #    Escort them. Should not happen after _do_collecting
            #    fixes, but guard anyway: never leave civilians as
            #    passengers attached to an idle guide.
            # 2. Mission complete: nobody anywhere needs help.
            #    Self-evacuate through the nearest exit.
            from evacsim.civilian import (
                IDLE,
                REACTING,
                WANDERING,
                FOLLOWING,
                FLEEING_ROOM,
                WALL_FOLLOWING,
            )

            NEEDS_HELP = (
                IDLE,
                REACTING,
                WANDERING,
                FOLLOWING,
                FLEEING_ROOM,
                WALL_FOLLOWING,
            )

            if self._group:
                self._start_escort(env, smoke, civilians)
                return

            any_needs_help = any(c.state in NEEDS_HELP for c in civilians)
            if not any_needs_help:
                # Nobody to help; self-evacuate.
                self._start_self_evacuation(env)
            else:
                # Still someone to help but no cluster. Stay put and rescan.
                self.vel = np.zeros(2)

    def _do_responding(self, env, smoke=None, civilians=None):
        # Reached max_group while walking to the hotspot. Escort now.
        if len(self._group) >= self._effective_max_group():
            self._start_escort(env, smoke, civilians)
            return

        # Same danger pattern as the other phases.
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
            # Is the destination hotspot now dangerous? Use a tighter
            # smoke threshold when carrying civilians (vulnerable).
            destination = self._waypoints[-1]
            dr, dc = env.world_to_grid(destination[0], destination[1])
            fg = getattr(self, "_fire_grid", None)
            sg = getattr(self, "_smoke_grid", None)
            rows, cols = env.grid.shape
            smoke_limit = 0.3 if self._group else 0.5
            if (
                fg is not None
                and 0 <= dr < rows
                and 0 <= dc < cols
                and fg[dr, dc] > 0.2
            ):
                self._waypoints.clear()
                if self._group:
                    # With civilians, escort to safety instead of rescanning.
                    self._start_escort(env, smoke, civilians)
                else:
                    self._phase = self.SCANNING
                    self.vel = np.zeros(2)
                return
            if (
                sg is not None
                and 0 <= dr < rows
                and 0 <= dc < cols
                and sg[dr, dc] > smoke_limit
            ):
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
                # Re-check max_group after recruiting.
                if len(self._group) >= self._effective_max_group():
                    self._start_escort(env, smoke, civilians)

    def _do_collecting(self, env, smoke, civilians):
        """Recruit civilians at the hotspot until full or timed out.

        Soft limit: collect as many as possible, but escort once
        max_group is reached or the wait timer expires. Includes the
        same proactive danger detection as the other phases: if fire
        or smoke is nearby, escort immediately regardless of group
        size.
        """
        if len(self._group) >= self._effective_max_group():
            self._start_escort(env, smoke, civilians)
            return

        # Proactive danger detection on the current group.
        if self._group:
            danger = False
            sg = getattr(self, "_smoke_grid", None)
            fg = getattr(self, "_fire_grid", None)
            rows, cols = env.grid.shape

            # Two-cell radius around the guide.
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

            # Group members in heavy smoke (no breathing protection).
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

        # Start escorting on enough people or timeout.
        has_enough = len(self._group) >= self._effective_max_group()
        timed_out = self._collect_wait >= self._max_collect_wait
        if has_enough or timed_out:
            if self._group:
                if len(self._group) >= self._min_group:
                    self._start_escort(env, smoke, civilians)
                else:
                    # Small group. Check whether anyone nearby is worth
                    # waiting for, otherwise escort what we have.
                    recruitable = self._count_recruitable(env, smoke, civilians)
                    if recruitable == 0:
                        # No one nearby. Escort what we have. Duty of
                        # care over efficiency: never leave civilians
                        # attached to an idle guide waiting for more.
                        self._start_escort(env, smoke, civilians)
                    elif self._collect_wait >= self._max_collect_wait * 2:
                        # Still below min_group after two waits. Escort
                        # what we have rather than holding them longer.
                        self._start_escort(env, smoke, civilians)
            else:
                # No civilians left at the collection point. Rescan.
                self._blackboard.release(self._guide_id)
                self._phase = self.SCANNING
                self.vel = np.zeros(2)

    def _do_escorting(self, env, smoke, civilians):
        """Lead the group to the chosen exit, picking up anyone passed.

        No group limit applies during escort. Periodically
        re-evaluates the exit choice; if a substantially better exit
        appears, switches and recomputes the path. This prevents
        committing to a poor exit for the entire trip.
        """
        if not self._group:
            # Group emptied (everyone died or was lost). Rescan.
            self._blackboard.release(self._guide_id)
            self._phase = self.SCANNING
            self.vel = np.zeros(2)
            return

        # Immediate danger reroute with cooldown. If the guide or
        # group members are in or near fire, reroute to the safest
        # exit. The 10-tick cooldown prevents recomputing BFS every
        # tick while still inside a danger zone.
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
                self._danger_reroute_tick = self._tick  # Start cooldown.
                # Danger grid: fire x3 so the 0.3 threshold matches the
                # 0.1 fire-damage onset for civilians.
                if fg is not None and sg is not None:
                    danger = np.maximum(fg * 3, sg * 1.0)
                elif fg is not None:
                    danger = fg * 3
                elif sg is not None:
                    danger = sg * 1.0
                else:
                    danger = None
                best_exit, best_score = self._escort_exit, float("inf")
                for ex, ey in self.known_exits:
                    dist = np.hypot(self.pos[0] - ex, self.pos[1] - ey)
                    # Effective congestion = current queue + incoming
                    # from other Routers. Stops emergency rerouting
                    # from stampeding everyone to the same exit.
                    current_queue = self._exit_counts.get((ex, ey), 0)
                    incoming = self._blackboard.get_incoming_count(
                        [ex, ey],
                        self._tick,
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
                # Use Dijkstra with safety validation rather than plain
                # BFS. BFS only checks 'passable' (at most 0.3) and picks
                # the shortest such path; Dijkstra's weighted cost
                # prefers genuinely cleaner routes, and the validator
                # rejects any path whose waypoints exceed 0.3 danger.
                dij_path = dijkstra_congestion_path(
                    env,
                    self.pos,
                    np.array(best_exit),
                    fire_grid=fg,
                    smoke_grid=sg,
                    civilian_density=self._density_grid,
                )
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
                    # BFS fallback with the danger grid.
                    self._waypoints = bfs_path(
                        env,
                        self.pos,
                        np.array(best_exit),
                        fire_grid=danger,
                    )

        self._recruit_nearby_unlimited(env, smoke, civilians)

        # Periodic exit re-evaluation. If a substantially better exit
        # appears (more than 30 percent better score), switch target
        # and recompute the path.
        # Commitment zone: do not switch once close to the current
        # exit. Civilians in FOLLOWING_GUIDE state pass through exits
        # without escaping (wrong state), so switching near the exit
        # would cause them to walk past and take extra smoke damage.
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
            # Score the current exit.
            cx, cy = self._escort_exit
            cur_dist = np.hypot(self.pos[0] - cx, self.pos[1] - cy)
            # Effective congestion = current queue + incoming from
            # other Routers. Excluding self ensures the current group
            # is not counted against the current target.
            cur_queue = self._exit_counts.get((cx, cy), 0)
            cur_incoming = self._blackboard.get_incoming_count(
                [cx, cy],
                self._tick,
                exclude_guide=self._guide_id,
            )
            cur_cong = (cur_queue + cur_incoming) * 30.0
            cur_smoke = smoke.get_density_at(cx, cy) * 200.0
            cur_score = cur_dist + cur_cong + cur_smoke

            for ex, ey in self.known_exits:
                if (ex, ey) == self._escort_exit:
                    continue
                dist = np.hypot(self.pos[0] - ex, self.pos[1] - ey)
                # Same effective-congestion adjustment for candidates.
                cand_queue = self._exit_counts.get((ex, ey), 0)
                cand_incoming = self._blackboard.get_incoming_count(
                    [ex, ey],
                    self._tick,
                    exclude_guide=self._guide_id,
                )
                cong = (cand_queue + cand_incoming) * 30.0
                s_pen = smoke.get_density_at(ex, ey) * 200.0
                f_pen = 0
                if fg is not None:
                    er, ec = env.world_to_grid(ex, ey)
                    if 0 <= er < fg.shape[0] and 0 <= ec < fg.shape[1]:
                        f_pen = fg[er, ec] * 500.0
                new_score = dist + cong + s_pen + f_pen
                # Switch only if the new score is at least 30 percent
                # better (avoids oscillation).
                if new_score < cur_score * 0.7:
                    self._escort_exit = (ex, ey)
                    # Build the danger grid up front. It is used both
                    # by the safety validator and by the BFS fallback.
                    # Fire is scaled x3 so the 0.3 threshold rejects
                    # fire above 0.1 (matches civilian damage onset).
                    if fg is not None and sg is not None:
                        danger = np.maximum(fg * 3, sg * 1.0)
                    elif fg is not None:
                        danger = fg * 3
                    elif sg is not None:
                        danger = sg * 1.0
                    else:
                        danger = None

                    dij_path = dijkstra_congestion_path(
                        env,
                        self.pos,
                        np.array((ex, ey)),
                        fire_grid=fg,
                        smoke_grid=sg,
                        civilian_density=self._density_grid,
                    )

                    # Safety validation on the re-evaluation path,
                    # matching _start_escort's pattern. Dijkstra's
                    # weighted cost can still pick a shorter path
                    # through moderate smoke over a clean longer path;
                    # the validator rejects any route with a waypoint
                    # in danger > 0.3 and falls back to BFS over the
                    # danger grid.
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
                        # BFS fallback with the danger grid.
                        self._waypoints = bfs_path(
                            env,
                            self.pos,
                            np.array((ex, ey)),
                            fire_grid=danger,
                        )
                    break  # Switch once per re-evaluation.

        if not self._waypoints:
            if self._escort_exit is not None:
                dist = np.linalg.norm(self.pos - np.array(self._escort_exit))
                if dist < env.cell_size * 2:
                    # Arrived. The group is now at the exit (joining
                    # the queue) and no longer 'incoming'. Release the
                    # exit claim so peers stop penalising this exit.
                    self._blackboard.release_exit_claim(self._guide_id)
                    self._phase = self.RELEASING
                    self._release_wait = 0
                    self.vel = np.zeros(2)
                    return
                # Stuck: waypoints empty but not near the exit.
                # Recompute a path so the guide does not stand still
                # mid-escort. Same three-layer cascade as _start_escort.
                fg = getattr(self, "_fire_grid", None)
                sg = getattr(self, "_smoke_grid", None)
                # Fire x3 so the 0.3 threshold rejects fire above 0.1.
                if fg is not None and sg is not None:
                    danger = np.maximum(fg * 3, sg * 1.0)
                elif fg is not None:
                    danger = fg * 3
                elif sg is not None:
                    danger = sg * 1.0
                else:
                    danger = None

                dij_path = dijkstra_congestion_path(
                    env,
                    self.pos,
                    np.array(self._escort_exit),
                    fire_grid=fg,
                    smoke_grid=sg,
                    civilian_density=self._density_grid,
                )
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
                    self._waypoints = bfs_path(
                        env,
                        self.pos,
                        np.array(self._escort_exit),
                        fire_grid=danger,
                    )
            if not self._waypoints:
                self.vel = np.zeros(2)
        else:
            self._follow_path(env)

    def _do_releasing(self, env, smoke=None):
        """Wait at the exit for the group to arrive and escape."""
        from evacsim.civilian import FOLLOWING_GUIDE, WANDERING

        self.vel = np.zeros(2)
        self._release_wait += 1

        still = []
        for civ in self._group:
            if civ.state != FOLLOWING_GUIDE:
                continue
            dist_exit = np.linalg.norm(civ.pos - np.array(self._escort_exit))
            dist_guide = np.linalg.norm(civ.pos - self.pos)
            if dist_exit < env.cell_size * 5 or dist_guide < env.cell_size * 3:
                civ._set_exit_target_force(env, self._escort_exit)
            else:
                still.append(civ)
        self._group = still

        if not self._group or self._release_wait >= self._release_max_wait:
            # On timeout, give stragglers an exit target rather than
            # leaving them lost.
            for civ in self._group:
                if civ.state == FOLLOWING_GUIDE:
                    civ._set_exit_target_force(env, self._escort_exit)
            self._group.clear()
            # Release the blackboard claim; the escort is complete.
            self._blackboard.release(self._guide_id)
            # Scan from the current position; no walking 'home'.
            self._phase = self.SCANNING
            self.vel = np.zeros(2)

    def _do_returning(self, env, smoke=None, civilians=None):
        """Return phase: scan for new hotspots while moving back.

        If no waypoints, transition directly to SCANNING. Same
        proactive danger handling as the other phases.
        """
        if not smoke or not civilians:
            if not self._waypoints:
                self._phase = self.SCANNING
                self.vel = np.zeros(2)
            else:
                self._follow_path(env)
            return

        if len(self._group) >= self._effective_max_group():
            self._start_escort(env, smoke, civilians)
            return

        # Proactive danger check.
        if self._group:
            danger = False
            for civ in self._group:
                if smoke.get_density_at(civ.pos[0], civ.pos[1]) > 0.3:
                    danger = True
                    break
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

        # Periodic hotspot scan while walking back. DBSCAN is O(n^2),
        # so it runs only on the eval boundary, every 20 ticks.
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
            # No destination. Transition to scanning.
            self._blackboard.release(self._guide_id)
            self._phase = self.SCANNING
            self.vel = np.zeros(2)
        else:
            self._follow_path(env)
            self._recruit_nearby(env, smoke, civilians)
            # Divert to escort only if the group meets the floor.
            if len(self._group) >= self._min_group:
                self._start_escort(env, smoke, civilians)

    # -- Intelligence methods --

    def _find_hotspot(self, env, smoke, civilians):
        """Find the best hotspot via DBSCAN with an individual fallback.

        Phase 1: DBSCAN cluster detection. Density-based spatial
        clustering identifies civilian groups of arbitrary shape.
        Parameters: epsilon = 3 cells (about 9 m, room-sized) and
        min_points = 3. Global scanning is used: blackboard
        coordination prevents multiple Routers from claiming the same
        hotspot.

        Phase 2: individual-civilian fallback. After several
        consecutive empty DBSCAN scans the population has dispersed.
        Switches to targeting the nearest unhelped civilian, which
        prevents abandoning isolated survivors late in the evacuation.

        Cluster scoring: size * (1 - 0.5 * centroid_smoke) *
        spread_bonus. The spread_bonus is a cumulative inverse-distance
        repulsion from peer positions, biasing the guide toward
        regions other guides are not covering.

        Skips clusters in fire or heavy smoke (unsafe to reach) and
        hotspots already claimed by other Routers.
        """
        from evacsim.civilian import (
            ESCAPED,
            DEAD,
            MOVING_TO_EXIT,
            FOLLOWING_GUIDE,
            QUEUED,
        )

        # 1. Collect civilians needing help with a global scan.
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

        # 2. Phase 1: DBSCAN cluster detection.
        if len(candidates) >= 3:
            positions = [c.pos for c in candidates]
            epsilon = env.cell_size * 3  # 3 cells, room-sized.
            clusters = dbscan_cluster(positions, epsilon, min_points=3)

            # Pre-compute data for inter-guide spread (cumulative
            # repulsion field). Each peer Router's current position
            # contributes an inverse-distance repulsion; when several
            # peers are clustered near a candidate hotspot, their
            # repulsions stack, biasing this guide toward distant,
            # uncovered areas.
            other_positions = self._blackboard.get_other_positions(
                exclude_guide=self._guide_id,
                current_tick=self._tick,
            )
            # Map diagonal normalises distances to [0, 1] so the
            # formula adapts to any map size.
            map_diag = float(np.hypot(rows * env.cell_size, cols * env.cell_size))
            # Decay constant: repulsion drops to about 50 percent at 10
            # percent of the map diagonal.
            REPULSION_DECAY = 10.0
            # Maximum spread bonus. A hotspot maximally far from every
            # peer receives a 1.0 + MAX_SPREAD_BONUS multiplier. With
            # 0.3, a cluster about 23 percent smaller can be preferred
            # if it is far from peers.
            MAX_SPREAD_BONUS = 0.3

            best_pos = None
            best_score = 0

            for cluster_indices in clusters:
                cx = sum(positions[i][0] for i in cluster_indices) / len(
                    cluster_indices
                )
                cy = sum(positions[i][1] for i in cluster_indices) / len(
                    cluster_indices
                )

                # Skip clusters in fire or heavy smoke.
                cr, cc = env.world_to_grid(cx, cy)
                if not (0 <= cr < rows and 0 <= cc < cols):
                    continue
                if fg is not None and fg[cr, cc] > 0.2:
                    continue
                if sg is not None and sg[cr, cc] > smoke_limit:
                    continue

                # Skip hotspots already claimed by other Routers.
                if self._blackboard.is_claimed(
                    [cx, cy], self._tick, exclude_guide=self._guide_id
                ):
                    continue

                size = len(cluster_indices)
                centroid_smoke = smoke.get_density_at(cx, cy) if smoke else 0
                safety_factor = 1.0 - centroid_smoke * 0.5

                # Cumulative repulsion: sum of inverse-distance
                # contributions from every peer's current position.
                #     repulsion_i = 1 / (1 + REPULSION_DECAY * d_norm_i)
                # where d_norm_i = distance / map_diagonal in [0, 1].
                # Stacking is natural: three peers at distance d
                # contribute three times the repulsion of one peer at
                # the same distance.
                if other_positions:
                    repulsion = 0.0
                    for p in other_positions:
                        d_norm = float(np.hypot(cx - p[0], cy - p[1])) / map_diag
                        repulsion += 1.0 / (1.0 + REPULSION_DECAY * d_norm)
                    # Normalise by the number of peers so the value is
                    # in [0, 1] regardless of guide count. 1.0 means
                    # maximum repulsion (all peers adjacent); 0 means
                    # no repulsion.
                    normalized = repulsion / len(other_positions)
                    spread_bonus = 1.0 + MAX_SPREAD_BONUS * (1.0 - normalized)
                else:
                    spread_bonus = 1.0  # Alone; no spread bias needed.

                score = size * safety_factor * spread_bonus

                if score > best_score:
                    best_score = score
                    best_pos = np.array([cx, cy])

            if best_pos is not None:
                self._no_cluster_count = 0  # Reset the fallback counter.
                # Claim this hotspot on the blackboard.
                self._blackboard.claim(self._guide_id, best_pos, self._tick)
                return self._snap_to_walkable(env, best_pos)

        # No clusters found. Increment the fallback counter.
        self._no_cluster_count += 1

        # 3. Phase 2: individual-civilian fallback.
        if self._no_cluster_count >= self._no_cluster_threshold:
            best_civ = None
            best_dist = float("inf")
            for civ in candidates:
                # Skip civilians in fire or heavy smoke zones.
                cr, cc = env.world_to_grid(civ.pos[0], civ.pos[1])
                if (
                    fg is not None
                    and 0 <= cr < rows
                    and 0 <= cc < cols
                    and fg[cr, cc] > 0.2
                ):
                    continue
                if (
                    sg is not None
                    and 0 <= cr < rows
                    and 0 <= cc < cols
                    and sg[cr, cc] > smoke_limit
                ):
                    continue
                dist = np.linalg.norm(self.pos - civ.pos)
                if dist < best_dist:
                    best_dist = dist
                    best_civ = civ

            if best_civ is not None:
                return self._snap_to_walkable(env, best_civ.pos.copy())

        return None

    def _snap_to_walkable(self, env, pos):
        """Snap a position to the nearest walkable cell when it lies in a wall."""
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
        """Recruit civilians within visual range, requiring line of sight."""
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
        """Recruit any civilian within visual range during escort.

        No group limit applies. Every civilian passed on the way to
        the exit is picked up provided line of sight allows.
        """
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
        """Count civilians actually recruitable by _recruit_nearby.

        Uses identical visual-range and line-of-sight checks so the
        count never reports false positives.
        """
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
        """Plan a deliberative escort: predict spread, then choose path.

        Reactive component: uses current congestion data for exit
        selection inputs.
        Deliberative component: simulates the fire and smoke automata
        forward by an estimated escort duration, then plans against
        the predicted grids so the chosen path is safe when the group
        arrives, not just safe now.

        This avoids the failure mode where a corridor is clear at the
        start of the escort but fills with smoke by the time the
        group passes through.
        """
        self._phase = self.ESCORTING
        fg = getattr(self, "_fire_grid", None)
        sg = getattr(self, "_smoke_grid", None)

        # Deliberative step: predict fire and smoke spread.
        # Estimate escort duration from straight-line distance and use
        # the predicted danger for both exit scoring and path planning.
        fire_sys = getattr(self, "_fire_system", None)
        smoke_sys = getattr(self, "_smoke_system", None)
        predicted_fg = fg
        predicted_sg = sg

        if fire_sys is not None and smoke_sys is not None:
            # Estimate ticks to reach each exit using straight-line
            # distance over the group speed.
            avg_dist = np.mean(
                [
                    np.hypot(self.pos[0] - ex, self.pos[1] - ey)
                    for ex, ey in self.known_exits
                ]
            )
            group_speed = max(self.speed * 0.75, 0.1)
            est_ticks = int(avg_dist / group_speed)
            est_ticks = min(est_ticks, 200)  # Cap at 200 ticks.

            if est_ticks > 10:  # Predict only if escort will take meaningful time.
                predicted_fg = predict_spread(
                    fire_sys.density,
                    fire_sys._burnable,
                    fire_sys._fire_cells,
                    fire_sys.spread_rate,
                    est_ticks,
                )
                predicted_sg = predict_spread(
                    smoke_sys.density,
                    smoke_sys._smokeable,
                    smoke_sys._fire_cells,
                    smoke_sys.spread_rate,
                    est_ticks,
                )

        # Score each exit using predicted conditions.
        best_exit, best_score = None, float("inf")
        for ex, ey in self.known_exits:
            dist = np.hypot(self.pos[0] - ex, self.pos[1] - ey)
            # Effective congestion = current queue + incoming from
            # other Routers (self excluded). The Router about to pick
            # an exit sees pending arrivals from peers and avoids
            # pile-ups.
            current_queue = self._exit_counts.get((ex, ey), 0)
            incoming = self._blackboard.get_incoming_count(
                [ex, ey],
                self._tick,
                exclude_guide=self._guide_id,
            )
            congestion = (current_queue + incoming) * 30.0
            # Use predicted smoke and fire at the exit (will it be
            # safe when we arrive?).
            er, ec = env.world_to_grid(ex, ey)
            smoke_pen = 0
            fire_pen = 0
            if (
                predicted_sg is not None
                and 0 <= er < predicted_sg.shape[0]
                and 0 <= ec < predicted_sg.shape[1]
            ):
                smoke_pen = predicted_sg[er, ec] * 200.0
            if (
                predicted_fg is not None
                and 0 <= er < predicted_fg.shape[0]
                and 0 <= ec < predicted_fg.shape[1]
            ):
                fire_pen = predicted_fg[er, ec] * 500.0
            score = dist + congestion + smoke_pen + fire_pen
            if score < best_score:
                best_score = score
                best_exit = (ex, ey)

        self._escort_exit = best_exit
        if best_exit:
            # Build the predicted danger grid for path validation.
            # Fire is scaled x3 so the 0.3 threshold rejects predicted
            # fire above 0.1 (matches civilian damage onset).
            if predicted_fg is not None and predicted_sg is not None:
                danger = np.maximum(predicted_fg * 3, predicted_sg * 1.0)
            elif predicted_fg is not None:
                danger = predicted_fg * 3
            elif predicted_sg is not None:
                danger = predicted_sg * 1.0
            else:
                danger = None

            # Primary plan: Dijkstra over predicted grids and density.
            dij_path = dijkstra_congestion_path(
                env,
                self.pos,
                np.array(best_exit),
                fire_grid=predicted_fg,
                smoke_grid=predicted_sg,
                civilian_density=self._density_grid,
            )

            # Validate: reject the Dijkstra path if any waypoint sits in
            # predicted danger above 0.3.
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
                # BFS fallback over the predicted danger grid.
                self._waypoints = bfs_path(
                    env, self.pos, np.array(best_exit), fire_grid=danger
                )

    def _update_exit_counts(self, civilians):
        """Estimate the number of civilians heading to each exit.

        Confirmed civilians (in MOVING_TO_EXIT or QUEUED with a known
        target) contribute weight 1.0. Other active civilians without a
        guide contribute weight 0.5 toward their nearest exit because
        they will likely go there eventually.

        This gives the Router early congestion awareness even before
        civilians have explicitly chosen exits.
        """
        from evacsim.civilian import (
            MOVING_TO_EXIT,
            QUEUED,
            ESCAPED,
            DEAD,
            FOLLOWING_GUIDE,
        )

        self._exit_counts = {(ex, ey): 0.0 for ex, ey in self.known_exits}
        for civ in civilians:
            if civ.state in (ESCAPED, DEAD):
                continue

            if civ.state in (MOVING_TO_EXIT, QUEUED) and civ._target_exit is not None:
                # Confirmed: heading to a specific exit (weight 1.0).
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
                # Predicted: will likely go to the nearest exit (weight 0.5).
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

        The guide has a mask, so smoke is weighted at 0.3 rather than
        1.0. Fire is scaled x3 so the 0.3 threshold matches the 0.1
        fire-damage onset for civilians, keeping consistency with the
        civilian damage model. This prevents guides repeatedly walking
        through smoke to a territory centre and dying.
        """
        if fg is not None and sg is not None:
            return np.maximum(fg * 3, sg * 0.3)
        elif fg is not None:
            return fg * 3
        elif sg is not None:
            return sg * 0.3
        return None

    def _release_group_wandering(self):
        """Release the group when the guide dies or self-evacuates."""
        from evacsim.civilian import WANDERING, FOLLOWING_GUIDE

        for civ in self._group:
            if civ.state == FOLLOWING_GUIDE:
                civ.assigned_guide = None
                civ.state = WANDERING
        self._group.clear()
        # Release the blackboard claim.
        self._blackboard.release(self._guide_id)

    def _clean_group(self):
        """Remove escaped or dead civilians from the group list."""
        from evacsim.civilian import ESCAPED, DEAD

        self._group = [c for c in self._group if c.state not in (ESCAPED, DEAD)]


def spawn_router_guides(env, n_guides, speed=2.0, seed=77, n_civilians=100):
    """Spawn Router guides at exits with a shared coordination blackboard.

    No territory is assigned; Routers scan globally and coordinate via
    the blackboard to avoid claiming the same hotspot. Each Router
    spawns at a different exit to give some initial spatial spread.
    """
    rng = np.random.default_rng(seed)
    exits = list(env.exits)

    blackboard = RouterBlackboard()

    guides = []
    for i in range(n_guides):
        ex, ey = exits[i % len(exits)]
        guides.append(
            RouterGuide(
                ex,
                ey,
                speed,
                env,
                rng,
                blackboard=blackboard,
                n_civilians=n_civilians,
                n_guides=n_guides,
            )
        )
    return guides
