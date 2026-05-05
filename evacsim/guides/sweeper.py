"""SweeperGuide: reactive patrol-and-escort policy.

The guide patrols an assigned territory, recruits civilians as it moves,
and periodically escorts a group to the nearest safe exit before
returning to its patrol route. The policy is purely reactive: every
decision is based on the current grid state with no forward prediction
or congestion inference.
"""

import math
import numpy as np
from evacsim.environment import Environment, CellType
from evacsim.smoke import SmokeSystem
from evacsim.fire import FireSystem
from evacsim.guides.base import Guide
from evacsim.guides.common import bfs_path, dijkstra_congestion_path


class SweeperBlackboard:
    """Shared coordination blackboard for Sweeper guides.

    Strictly reactive: every channel stores current state read for
    current decisions. No prediction, no multi-step planning, no
    congestion inference. Provides a uniform coordination mechanism
    that mirrors RouterBlackboard structurally while preserving the
    Sweeper's reactive identity.

    Four channels:

    1. POSITIONS (TTL 10 ticks). Each Sweeper posts its current world
       position every tick. Parallels RouterBlackboard's position
       channel. The field is passive in this implementation (written
       but not read by the patrol FSM); included for architectural
       symmetry and diagnostic visibility.

    2. PATROL TARGETS (TTL 30 ticks). Each Sweeper posts the patrol
       cell it is currently walking toward. Peers consult this channel
       during _advance_patrol and skip candidate cells currently claimed
       by another guide. Prevents duplicate visits after territory
       reallocation, when adopted territories can overlap live peer
       routes. The 30-tick TTL covers a typical walk-to-point; if the
       claiming guide dies silently, the claim auto-expires.

    3. CLEARED CELLS (persistent for the simulation). Each Sweeper
       posts the world position of every patrol cell it has marked
       cleared. Peers consult this channel in _advance_patrol and skip
       candidate cells near positions a peer has already visited.
       Persistence is correct because cleared cells cannot become
       re-occupied by unassisted civilians during a single run.

    4. WORKLOAD (TTL 30 ticks). Each Sweeper posts its uncleared
       patrol-point count every tick alongside a self-reference. Used
       by _try_reallocate_to_peer to pick the busiest peer without
       iterating a class-level registry.

    Lifetime: one instance per simulation, created in
    spawn_sweeper_guides and handed to each SweeperGuide through the
    constructor. A fresh instance per spawn call handles cross-run
    isolation under multiprocessing without a class-level reset.
    """

    POSITION_EXPIRY = 10
    TARGET_EXPIRY = 30
    # Workload is republished every tick by alive guides, so the TTL
    # only acts as silent-death cleanup, matching POSITION_EXPIRY.
    WORKLOAD_EXPIRY = 10

    def __init__(self):
        self._positions = {}  # guide_id -> (np.array pos, tick)
        self._patrol_targets = {}  # guide_id -> (np.array pos, tick)
        self._cleared_cells = []  # list of np.array world positions
        self._workload = {}  # guide_id -> (count, guide_ref, tick)

    def update_position(self, guide_id, position, tick):
        """Record the current position of a Sweeper. Called every tick."""
        self._positions[guide_id] = (np.array(position, dtype=float), tick)

    def get_other_positions(self, exclude_guide, current_tick):
        """Return fresh positions of other Sweepers.

        Lazy-cleans stale entries so dead or departed guides do not keep
        holding territory.
        """
        result = []
        for gid, (pos, tick) in list(self._positions.items()):
            if gid == exclude_guide:
                continue
            if current_tick - tick > self.POSITION_EXPIRY:
                del self._positions[gid]
                continue
            result.append(pos)
        return result

    def claim_patrol_target(self, guide_id, position, tick):
        """Publish the cell this Sweeper is walking toward."""
        self._patrol_targets[guide_id] = (
            np.array(position, dtype=float),
            tick,
        )

    def is_target_claimed(
        self, position, current_tick, exclude_guide=None, radius=20.0
    ):
        """Return True if any other guide has a fresh claim within radius
        world units. Lazy-cleans stale entries on the fly.
        """
        target = np.array(position, dtype=float)
        for gid, (pos, tick) in list(self._patrol_targets.items()):
            if gid == exclude_guide:
                continue
            if current_tick - tick > self.TARGET_EXPIRY:
                del self._patrol_targets[gid]
                continue
            if np.linalg.norm(pos - target) < radius:
                return True
        return False

    def post_cleared(self, position):
        """Post a cell as cleared. Persistent for the simulation (no TTL):
        once a patrol point is confirmed empty, unassisted civilians do
        not re-occupy it within one run.
        """
        self._cleared_cells.append(np.array(position, dtype=float))

    def is_cleared(self, position, radius=20.0):
        """Return True if any peer has marked a cell within radius world
        units cleared.
        """
        target = np.array(position, dtype=float)
        for pos in self._cleared_cells:
            if np.linalg.norm(pos - target) < radius:
                return True
        return False

    def update_workload(self, guide_id, count, guide, tick):
        """Publish uncleared patrol count and a self-reference so peers
        can identify the busiest guide without iterating a class registry.
        """
        self._workload[guide_id] = (int(count), guide, tick)

    def get_busiest_peer(self, exclude_guide, current_tick, min_count=3):
        """Return the peer guide with the highest uncleared count, provided
        it meets min_count. Filters dead and stale entries.
        """
        best = None
        best_count = 0
        for gid, (count, guide, tick) in list(self._workload.items()):
            if gid == exclude_guide:
                continue
            if current_tick - tick > self.WORKLOAD_EXPIRY:
                del self._workload[gid]
                continue
            if not getattr(guide, "alive", False):
                continue
            if count > best_count:
                best_count = count
                best = guide
        if best is None or best_count < min_count:
            return None
        return best

    def release(self, guide_id):
        """Remove this guide's live-state entries from every channel.

        Called when a Sweeper dies or self-evacuates, mirroring
        RouterBlackboard.release. All channels store current state per
        guide except _cleared_cells, which is a persistent coverage log
        contributed to across the simulation; those entries stay (a
        dead guide's earlier coverage knowledge is still true).

        Reads in every channel already filter stale or dead entries
        (alive check plus TTL), so this method is structural tidiness:
        it wipes the dict immediately instead of waiting 10 ticks.
        """
        self._positions.pop(guide_id, None)
        self._patrol_targets.pop(guide_id, None)
        self._workload.pop(guide_id, None)

    def reset(self):
        """Clear all blackboard state for a new simulation."""
        self._positions.clear()
        self._patrol_targets.clear()
        self._cleared_cells.clear()
        self._workload.clear()


def _find_patrol_points(env: Environment):
    """Build the patrol-point list using room detection plus a grid scan.

    The two strategies complement each other:

        Bottleneck maps (many rooms): use room centres as patrol points
        and add a sparse grid scan to cover corridors.

        Open-plan maps (few or no rooms): use a denser grid scan for
        full coverage.

    Duplicates within three cells of each other are removed at the end.
    """
    rows, cols = env.grid.shape
    points = []

    # Try room detection first.
    rooms = []
    try:
        rooms = env.detect_rooms()
        for room_cells in rooms:
            if not room_cells:
                continue
            avg_r = sum(r for r, c in room_cells) / len(room_cells)
            avg_c = sum(c for r, c in room_cells) / len(room_cells)
            best, best_d = None, float("inf")
            for r, c in room_cells:
                d = abs(r - avg_r) + abs(c - avg_c)
                if d < best_d:
                    best_d = d
                    best = (r, c)
            if best:
                wx, wy = env.grid_to_world(best[0], best[1])
                points.append(np.array([wx, wy], dtype=float))
    except Exception:
        pass

    # Grid-scan density depends on whether rooms were found.
    if len(rooms) >= 2:
        # Bottleneck map. Add sparse corridor points as a supplement.
        row_step, col_step = 5, 10
    else:
        # Open-plan map. Use a denser grid scan for full coverage.
        row_step, col_step = 4, 6

    for r in range(2, rows - 2, row_step):
        for c in range(2, cols - 2, col_step):
            if env.grid[r, c] != CellType.WALL:
                wx, wy = env.grid_to_world(r, c)
                points.append(np.array([wx, wy], dtype=float))

    # Remove points within three cells of each other.
    if not points:
        return points
    filtered = [points[0]]
    min_dist = env.cell_size * 3
    for p in points[1:]:
        if all(np.linalg.norm(p - f) >= min_dist for f in filtered):
            filtered.append(p)
    return filtered


def _balanced_patrol_order(start_pos, waypoints):
    """Order waypoints so the guide alternates between halves of the map.

    Points are split into 'above start', 'below start' (by y-coordinate),
    and 'near start', then interleaved. Within each group points are
    sorted greedy-nearest. This ensures the guide visits both halves of
    the building rather than only the nearest half.
    """
    if not waypoints:
        return []

    sy = start_pos[1]
    above = [p for p in waypoints if p[1] < sy - 5]
    below = [p for p in waypoints if p[1] > sy + 5]
    middle = [p for p in waypoints if abs(p[1] - sy) <= 5]

    def greedy_sort(pts, start):
        remaining = list(pts)
        ordered = []
        current = start.copy()
        while remaining:
            best_idx = min(
                range(len(remaining)),
                key=lambda i: np.linalg.norm(current - remaining[i]),
            )
            ordered.append(remaining.pop(best_idx))
            current = ordered[-1]
        return ordered

    above_sorted = greedy_sort(above, start_pos)
    below_sorted = greedy_sort(below, start_pos)
    middle_sorted = greedy_sort(middle, start_pos)

    # Start with the middle points, then alternate above and below.
    result = list(middle_sorted)
    for i in range(max(len(above_sorted), len(below_sorted))):
        if i < len(above_sorted):
            result.append(above_sorted[i])
        if i < len(below_sorted):
            result.append(below_sorted[i])

    return result


class SweeperGuide(Guide):
    """Patrol-and-escort guide that walks an assigned territory, collects
    civilians into a group, and escorts them to the nearest safe exit.

    Phase loop:
        1. COLLECTING: walk to the next patrol point, recruit nearby
           civilians, wait briefly, then advance.
        2. ESCORTING: lead the group to the nearest safe exit, picking
           up any civilian passed on the way.
        3. RELEASING: stand at the exit and direct stragglers to it,
           then return to patrolling.
        4. RETURNING: walk back toward the patrol route and resume
           collecting.

    Coordination: when a Sweeper finishes clearing its entire territory,
    it adopts uncleared patrol points from a peer Sweeper that still has
    work, mirroring real fire-brigade radio coordination ('Zone 1 clear,
    moving to assist Zone 2'). This preserves the policy's systematic
    identity while preventing idle guides on unbalanced territories.

    Spawns from exits, treating them as ingress points.
    """

    COLLECTING = "collecting"
    ESCORTING = "escorting"
    RELEASING = "releasing"  # Waiting at the exit for the group to arrive.
    RETURNING = "returning"

    def __init__(
        self,
        x: float,
        y: float,
        speed: float,
        env: Environment,
        rng: np.random.Generator,
        patrol_points: list = None,
        n_civilians: int = 100,
        n_guides: int = 4,
        blackboard: "SweeperBlackboard" = None,
    ):
        super().__init__(x, y, speed, env, rng)

        # Shared coordination blackboard. A fresh per-guide instance is
        # used when no shared board is provided, so stand-alone
        # construction (for example in unit tests) still works.
        self._blackboard = blackboard or SweeperBlackboard()
        self._guide_id = id(self)

        self._group = []

        # Dynamic commitment ceiling. The base value is the integer
        # closest to 3 * sqrt(n_civilians / n_guides), derived from
        # minimising the cost C(n) = P*beta/n + P*h*n where beta is
        # fixed trip overhead and h*n captures quadratic danger
        # exposure. The floor of 5 matches the min_group efficiency
        # threshold below which escort overhead dominates.
        self._base_max_group = round(3 * math.sqrt(n_civilians / max(1, n_guides)))
        self._floor = 5
        self._tick = 0
        self._cached_effective_max = self._base_max_group
        self._cached_max_tick = -1

        # Build the patrol route.
        all_pts = patrol_points if patrol_points else _find_patrol_points(env)
        self._patrol_route = _balanced_patrol_order(self.pos, all_pts)
        self._patrol_idx = 0

        # Track cleared rooms so they are skipped until a full cycle completes.
        self._cleared_points = set()
        self._cycle_start_idx = 0  # Where the current patrol cycle started.

        self._waypoints = []
        self._phase = self.COLLECTING

        # Brief wait at each patrol point during recruitment.
        self._wait_ticks = 15
        self._wait_counter = 0
        self._waiting = False

        # Target exit for escorting.
        self._escort_exit = None

        # Releasing: wait at the exit for stragglers.
        self._release_wait = 0
        self._release_max_wait = 30

        # Start the patrol.
        if self._patrol_route:
            self._waypoints = bfs_path(env, self.pos, self._patrol_route[0])
            # Publish the initial target claim so peers see it from tick 0.
            self._blackboard.claim_patrol_target(
                self._guide_id,
                self._patrol_route[0],
                self._tick,
            )

    def _effective_max_group(self) -> int:
        """Return the dynamic commitment ceiling for the current tick.

        Uses a zero-order fuzzy interpolation rule with two triangular
        membership functions (SAFE and DANGEROUS). The closed-form
        result is a linear interpolation:

            effective = base - d * (base - floor)

        where d is the normalised peak danger in a three-cell radius
        around the guide and is therefore in [0, 1]. Sweeper samples
        the current fire and smoke grids (reactive sensing).
        Cached per tick because each phase method may call this several
        times per tick.
        """
        if self._cached_max_tick == self._tick:
            return self._cached_effective_max

        sg = getattr(self, "_smoke_grid", None)
        fg = getattr(self, "_fire_grid", None)

        if sg is None and fg is None:
            self._cached_effective_max = self._base_max_group
            self._cached_max_tick = self._tick
            return self._base_max_group

        gr, gc = getattr(self, "_last_grid_pos", (0, 0))
        rows, cols = (sg if sg is not None else fg).shape

        peak_fire = 0.0
        peak_smoke = 0.0
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                nr, nc = gr + dr, gc + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if fg is not None and fg[nr, nc] > peak_fire:
                        peak_fire = fg[nr, nc]
                    if sg is not None and sg[nr, nc] > peak_smoke:
                        peak_smoke = sg[nr, nc]

        # Normalise to [0, 1] using the existing emergency thresholds.
        fire_factor = min(1.0, peak_fire / 0.1)
        smoke_factor = min(1.0, peak_smoke / 0.4)
        d = max(fire_factor, smoke_factor)

        base = self._base_max_group
        effective = max(self._floor, round(base - d * (base - self._floor)))

        self._cached_effective_max = effective
        self._cached_max_tick = self._tick
        return effective

    def update(
        self, env: Environment, smoke: SmokeSystem, fire: FireSystem, civilians: list
    ) -> None:
        super().update(env, smoke, fire, civilians)
        self._tick += 1

        if not self.alive:
            self._release_group_no_exit()
            return

        if self._do_self_evacuation(env):
            self._release_group_no_exit()
            return

        # Publish current position to the shared blackboard (diagnostic
        # and symmetry channel; not read by the patrol FSM).
        self._blackboard.update_position(self._guide_id, self.pos, self._tick)

        # Publish workload (uncleared patrol count) so peers can identify
        # the busiest Sweeper without iterating a class-level registry.
        if self._patrol_route:
            uncleared = sum(
                1
                for i in range(len(self._patrol_route))
                if i not in self._cleared_points
            )
        else:
            uncleared = 0
        self._blackboard.update_workload(
            self._guide_id,
            uncleared,
            self,
            self._tick,
        )

        self._clean_group()

        # Always recruit nearby civilians while moving.
        self._recruit_nearby(env, smoke, civilians)

        if self._phase == self.COLLECTING:
            self._do_collecting(env, smoke, civilians)
        elif self._phase == self.ESCORTING:
            self._do_escorting(env, smoke, civilians)
        elif self._phase == self.RELEASING:
            self._do_releasing(env)
        elif self._phase == self.RETURNING:
            self._do_returning(env, smoke, civilians)

    def _do_collecting(self, env, smoke, civilians):
        """Walk the patrol route, recruit civilians, escort if full or in danger."""
        if len(self._group) >= self._effective_max_group():
            self._start_escort(env)
            return

        # Proactive danger detection on the current group.
        if self._group:
            danger = False
            sg = getattr(self, "_smoke_grid", None)
            fg = getattr(self, "_fire_grid", None)
            rows, cols = env.grid.shape

            # Two-cell radius around the guide (proactive detection).
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
                self._start_escort(env)
                return

        if self._waiting:
            self._wait_counter += 1
            if self._wait_counter >= self._wait_ticks:
                self._waiting = False
                recruitable = self._count_recruitable(env, smoke, civilians)
                if recruitable == 0:
                    # Empty patrol point. Mark cleared locally and on the
                    # shared blackboard so peers can skip it.
                    self._cleared_points.add(self._patrol_idx)
                    if 0 <= self._patrol_idx < len(self._patrol_route):
                        self._blackboard.post_cleared(
                            self._patrol_route[self._patrol_idx]
                        )
                if self._group and recruitable == 0:
                    # Empty point with a small group already collected.
                    # Escort if the group is at least at the floor,
                    # otherwise advance to the next patrol point.
                    if len(self._group) >= 5:
                        self._start_escort(env)
                        return
                    # Small group: keep them and move to the next point.
                self._advance_patrol(env)
            return

        if not self._waypoints:
            self._waiting = True
            self._wait_counter = 0
            self.vel = np.zeros(2)
        else:
            self._follow_path(env)

    def _do_escorting(self, env, smoke, civilians):
        """Lead the group to an exit, picking up any civilian passed.

        No upper group limit applies during escorting. If fire or smoke
        threatens the guide or any group member, the escort is rerouted
        to the nearest safe exit immediately.
        """
        if not self._group:
            self._phase = self.RETURNING
            self._advance_patrol(env)
            return

        # Reroute to the nearest safe exit if danger is detected.
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
            self._start_escort(env)

        # Pick up any civilian along the way (no group limit during escort).
        self._recruit_nearby_escort(env, smoke, civilians)
        if not self._waypoints:
            if self._escort_exit is not None:
                dist = np.linalg.norm(self.pos - np.array(self._escort_exit))
                if dist < env.cell_size * 2:
                    # Arrived at the exit. Begin RELEASING.
                    self._phase = self.RELEASING
                    self._release_wait = 0
                    self.vel = np.zeros(2)
                    return
            self.vel = np.zeros(2)
        else:
            self._follow_path(env)

    def _do_releasing(self, env):
        """Stand at the exit and direct group members through it.

        Civilians within five cells of the exit or three cells of the
        guide receive a forced exit target. After max_wait ticks any
        remaining group members are forced through and the guide
        returns to patrol.
        """
        from evacsim.civilian import FOLLOWING_GUIDE

        self.vel = np.zeros(2)
        self._release_wait += 1

        # Active release: forward each civilian to the exit.
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

        # End the release phase when the group is empty or on timeout.
        if not self._group or self._release_wait >= self._release_max_wait:
            # On timeout, give stragglers an exit target rather than
            # leaving them lost.
            for civ in self._group:
                if civ.state == FOLLOWING_GUIDE:
                    civ._set_exit_target_force(env, self._escort_exit)
            self._group.clear()
            self._phase = self.RETURNING
            self._advance_patrol(env)

    def _do_returning(self, env, smoke, civilians):
        """Walk back along the patrol route to resume collecting."""
        if len(self._group) >= self._effective_max_group():
            self._start_escort(env)
            return

        # Civilians picked up during return: check group safety.
        if self._group:
            danger = False
            for civ in self._group:
                if smoke.get_density_at(civ.pos[0], civ.pos[1]) > 0.3:
                    danger = True
                    break
            if not danger:
                sg = getattr(self, "_smoke_grid", None)
                if sg is not None:
                    gr, gc = env.world_to_grid(self.pos[0], self.pos[1])
                    rows, cols = sg.shape
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            nr, nc = gr + dr, gc + dc
                            if 0 <= nr < rows and 0 <= nc < cols and sg[nr, nc] > 0.4:
                                danger = True
                                break
                        if danger:
                            break
            if danger:
                self._start_escort(env)
                return

        if not self._waypoints:
            self._phase = self.COLLECTING
            self._waiting = True
            self._wait_counter = 0
            self.vel = np.zeros(2)
        else:
            self._follow_path(env)

    def _recruit_nearby(self, env, smoke, civilians):
        """Recruit civilians within visual range, requiring line of sight.

        Walls block recruitment. Civilians who are already on the way to
        an exit, queueing, or attached to another guide are skipped.
        IDLE and REACTING civilians have their reaction shortened so
        they will be ready for recruitment on a later tick.
        """
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
            QUEUED,
        )

        for civ in civilians:
            if len(self._group) >= self._effective_max_group():
                break
            # Skip QUEUED civilians: they are already at an exit and
            # about to escape. Pulling them out of the queue would be
            # unrealistic and creates an asymmetry with Router.
            if civ.state in (MOVING_TO_EXIT, ESCAPED, DEAD, FOLLOWING_GUIDE, QUEUED):
                continue
            if civ.assigned_guide is not None:
                continue

            dist = np.linalg.norm(civ.pos - self.pos)
            if dist < radius and self._has_line_of_sight(civ.pos, env):
                # Wake IDLE or REACTING civilians.
                if civ.state == IDLE:
                    civ.state = REACTING
                    civ._reaction_counter = 0
                    civ._reaction_delay = min(civ._reaction_delay, 5)
                    continue  # Wait for the reaction to finish.
                elif civ.state == REACTING:
                    civ._reaction_delay = min(civ._reaction_delay, 5)
                    continue  # Still reacting.
                else:
                    self._group.append(civ)
                    civ.assigned_guide = self
                    civ.state = FOLLOWING_GUIDE

    def _recruit_nearby_escort(self, env, smoke, civilians):
        """Recruit any civilian along the escort route without a group limit.

        Same selection rules as _recruit_nearby (visual range, line of
        sight, skip QUEUED) but without the maximum-group cap.
        """
        radius = self.get_visual_recruit_radius(smoke)
        from evacsim.civilian import (
            MOVING_TO_EXIT,
            ESCAPED,
            DEAD,
            IDLE,
            REACTING,
            FOLLOWING_GUIDE,
            QUEUED,
        )

        for civ in civilians:
            if civ.state in (MOVING_TO_EXIT, ESCAPED, DEAD, FOLLOWING_GUIDE, QUEUED):
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

    def _start_escort(self, env):
        """Switch to ESCORTING and plan a path to the nearest safe exit.

        Exit score = distance + 500 * exit_fire + 200 * exit_smoke.
        The chosen path is the safe Dijkstra route when the validator
        accepts it (no waypoint above 0.3 danger), otherwise BFS over a
        combined danger grid.
        """
        self._phase = self.ESCORTING
        fg = getattr(self, "_fire_grid", None)

        best_exit, best_score = None, float("inf")
        for ex, ey in self.known_exits:
            dist = np.hypot(self.pos[0] - ex, self.pos[1] - ey)
            danger_penalty = 0.0
            if fg is not None:
                er, ec = env.world_to_grid(ex, ey)
                if 0 <= er < fg.shape[0] and 0 <= ec < fg.shape[1]:
                    danger_penalty += fg[er, ec] * 500.0
            sg = getattr(self, "_smoke_grid", None)
            if sg is not None:
                er, ec = env.world_to_grid(ex, ey)
                if 0 <= er < sg.shape[0] and 0 <= ec < sg.shape[1]:
                    danger_penalty += sg[er, ec] * 200.0
            score = dist + danger_penalty
            if score < best_score:
                best_score = score
                best_exit = (ex, ey)

        self._escort_exit = best_exit
        if best_exit:
            # When escorting, smoke is at full danger weight (civilians
            # are vulnerable). This must match _follow_path's reroute
            # weight (smoke x 1.0 when has_group). Fire is scaled x3 so
            # the 0.3 threshold rejects fire above 0.1, matching civilian
            # fire-damage onset.
            sg = getattr(self, "_smoke_grid", None)
            if fg is not None and sg is not None:
                danger = np.maximum(fg * 3, sg * 1.0)
            elif fg is not None:
                danger = fg * 3
            elif sg is not None:
                danger = sg * 1.0
            else:
                danger = None
            # Three-layer cascade: Dijkstra cost weighting (fire x 20,
            # smoke x 5), safety validator that rejects any path with a
            # waypoint above 0.3 danger, BFS fallback with the danger
            # grid. civilian_density stays None: density-aware routing
            # is Router-only.
            dij_path = dijkstra_congestion_path(
                env,
                self.pos,
                np.array(best_exit),
                fire_grid=fg,
                smoke_grid=sg,
                civilian_density=None,
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
                    np.array(best_exit),
                    fire_grid=danger,
                )

    def _release_group_no_exit(self):
        """Release the group when the guide dies or self-evacuates.

        Civilians attached to the group are returned to WANDERING so
        they can resume autonomous evacuation. The guide's live-state
        entries are also released from the shared blackboard so peers
        see immediate cleanup rather than waiting for TTL expiry.
        """
        from evacsim.civilian import WANDERING, FOLLOWING_GUIDE

        for civ in self._group:
            if civ.state == FOLLOWING_GUIDE:
                civ.assigned_guide = None
                civ.state = WANDERING
        self._group.clear()
        self._blackboard.release(self._guide_id)

    def _clean_group(self):
        """Remove escaped or dead civilians from the group list."""
        from evacsim.civilian import ESCAPED, DEAD

        self._group = [c for c in self._group if c.state not in (ESCAPED, DEAD)]

    def _try_reallocate_to_peer(self, env):
        """Adopt half of the busiest peer's uncleared patrol points.

        Triggered when this guide has cleared its own territory.
        Returns True if reallocation succeeded (patrol_route and
        waypoints updated). Peer selection goes through the shared
        blackboard's workload channel; a defensive re-check guards
        against stale workload signals.
        """
        best_peer = self._blackboard.get_busiest_peer(
            exclude_guide=self._guide_id,
            current_tick=self._tick,
            min_count=3,
        )
        if best_peer is None:
            return False

        # Defensive re-check against the peer's live state. The workload
        # signal may be up to WORKLOAD_EXPIRY ticks stale, so the peer's
        # current uncleared count could be lower than reported.
        peer_live_uncleared = sum(
            1
            for i in range(len(best_peer._patrol_route))
            if i not in best_peer._cleared_points
        )
        if peer_live_uncleared < 3:
            return False

        # Take half of the peer's uncleared points so both guides keep working.
        peer_uncleared_indices = [
            i
            for i in range(len(best_peer._patrol_route))
            if i not in best_peer._cleared_points
        ]
        n_take = max(1, len(peer_uncleared_indices) // 2)
        taken_indices = peer_uncleared_indices[:n_take]

        # Copy those patrol points into a new route for this guide.
        new_points = [best_peer._patrol_route[i] for i in taken_indices]

        # Mark them cleared in the peer's set so the peer does not revisit.
        for idx in taken_indices:
            best_peer._cleared_points.add(idx)

        # Replace this guide's patrol route with the adopted territory.
        self._patrol_route = new_points
        self._cleared_points = set()
        self._patrol_idx = 0
        self._cycle_start_idx = 0

        # Compute waypoints to the new first point with a danger-aware grid.
        fg = getattr(self, "_fire_grid", None)
        sg = getattr(self, "_smoke_grid", None)
        if self._group:
            if fg is not None and sg is not None:
                danger = np.maximum(fg * 3, sg * 1.0)
            elif fg is not None:
                danger = fg * 3
            elif sg is not None:
                danger = sg * 1.0
            else:
                danger = None
        else:
            if fg is not None and sg is not None:
                danger = np.maximum(fg * 3, sg * 0.3)
            elif fg is not None:
                danger = fg * 3
            elif sg is not None:
                danger = sg * 0.3
            else:
                danger = None

        if self._group:
            # With civilians in tow, use the three-layer cascade.
            dij_path = dijkstra_congestion_path(
                env,
                self.pos,
                self._patrol_route[0],
                fire_grid=fg,
                smoke_grid=sg,
                civilian_density=None,
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
                    self._patrol_route[0],
                    fire_grid=danger,
                )
        else:
            # Solo movement: BFS is fast and sufficient.
            self._waypoints = bfs_path(
                env, self.pos, self._patrol_route[0], fire_grid=danger
            )

        # Publish the adopted territory's first target so other peers
        # treat it as claimed by this guide going forward.
        self._blackboard.claim_patrol_target(
            self._guide_id,
            self._patrol_route[0],
            self._tick,
        )
        return True

    def _advance_patrol(self, env):
        """Advance to the next patrol point, skipping cleared and unsafe ones."""
        if not self._patrol_route:
            return

        fg = getattr(self, "_fire_grid", None)
        sg = getattr(self, "_smoke_grid", None)

        # Track whether this call has wrapped to cycle-complete without a
        # successful reallocation. When that happens, peer-gating is
        # relaxed so the guide can re-walk its own territory rather than
        # stall: the persistent CLEARED CELLS channel would otherwise
        # block every candidate. IDLE civilians who wander into
        # previously empty cells are the main reason re-walking has
        # residual value.
        cycle_restarted = False

        # Try up to len(route) candidate points.
        for _ in range(len(self._patrol_route)):
            self._patrol_idx = (self._patrol_idx + 1) % len(self._patrol_route)

            # Full cycle completed. Try territory reallocation before
            # resetting; if a peer's points can be adopted, do that
            # instead of re-walking already-empty territory.
            if self._patrol_idx == self._cycle_start_idx and not cycle_restarted:
                if self._try_reallocate_to_peer(env):
                    return  # Waypoints already set by reallocation.
                self._cleared_points.clear()
                self._cycle_start_idx = self._patrol_idx
                cycle_restarted = True

            # Skip points already cleared by this guide.
            if self._patrol_idx in self._cleared_points:
                continue

            target = self._patrol_route[self._patrol_idx]

            # Peer-gating applies to the first-pass coverage assignment,
            # not to the post-exhaustion re-walk. Skipping the two
            # blackboard checks on a restarted cycle restores the
            # pre-blackboard fallback behaviour: walk own territory
            # while picking up any civilian who has drifted in.
            if not cycle_restarted:
                # Skip cells another peer is currently walking toward.
                # TTL 30 ticks (TARGET_EXPIRY) auto-recovers from silent
                # peer death.
                if self._blackboard.is_target_claimed(
                    target,
                    self._tick,
                    exclude_guide=self._guide_id,
                ):
                    continue

                # Skip cells a peer has already visited and confirmed
                # empty. The cleared-cells channel is persistent because
                # unassisted civilians cannot re-occupy a cell within
                # one run.
                if self._blackboard.is_cleared(target):
                    continue

            tr, tc = env.world_to_grid(target[0], target[1])

            # Skip patrol points that are now in fire or heavy smoke.
            is_dangerous = False
            if fg is not None and 0 <= tr < fg.shape[0] and 0 <= tc < fg.shape[1]:
                if fg[tr, tc] > 0.2:
                    is_dangerous = True
            if sg is not None and 0 <= tr < sg.shape[0] and 0 <= tc < sg.shape[1]:
                if sg[tr, tc] > 0.5:
                    is_dangerous = True

            if not is_dangerous:
                break  # Found a safe patrol point.

        # Publish the new patrol target so peers skip it. Claimed even
        # when the loop exhausted and landed on a degraded candidate:
        # stale claims auto-expire after TARGET_EXPIRY ticks.
        self._blackboard.claim_patrol_target(
            self._guide_id,
            self._patrol_route[self._patrol_idx],
            self._tick,
        )

        # Compute the path. Fire is scaled x3 so the 0.3 threshold
        # rejects fire above 0.1. Smoke weight depends on whether the
        # guide is escorting civilians.
        if self._group:
            if fg is not None and sg is not None:
                danger = np.maximum(fg * 3, sg * 1.0)
            elif fg is not None:
                danger = fg * 3
            elif sg is not None:
                danger = sg * 1.0
            else:
                danger = None
        else:
            # Solo: the guide is masked, so smoke weight is 0.3.
            if fg is not None and sg is not None:
                danger = np.maximum(fg * 3, sg * 0.3)
            elif fg is not None:
                danger = fg * 3
            elif sg is not None:
                danger = sg * 0.3
            else:
                danger = None

        if self._group:
            # With civilians in tow, use the three-layer cascade.
            target = self._patrol_route[self._patrol_idx]
            dij_path = dijkstra_congestion_path(
                env,
                self.pos,
                target,
                fire_grid=fg,
                smoke_grid=sg,
                civilian_density=None,
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
                    target,
                    fire_grid=danger,
                )
        else:
            # Solo movement: BFS is fast and sufficient.
            self._waypoints = bfs_path(
                env,
                self.pos,
                self._patrol_route[self._patrol_idx],
                fire_grid=danger,
            )


def spawn_sweeper_guides(
    env: Environment,
    n_guides: int,
    speed: float = 2.0,
    seed: int = 77,
    n_civilians: int = 100,
) -> list[SweeperGuide]:
    """Spawn n_guides Sweepers from exit positions and split coverage.

    Patrol points are partitioned geographically so guides do not
    overlap:
        1 guide:  full map
        2 guides: top half and bottom half (y-coordinate split)
        3+ guides: quadrants based on the map centre

    A single SweeperBlackboard is created per call and shared among
    every spawned guide so the four coordination channels stay
    isolated across simulations under multiprocessing.
    """
    rng = np.random.default_rng(seed)
    exits = list(env.exits)
    all_points = _find_patrol_points(env)

    if not all_points:
        return []

    center_y = env.height / 2.0
    center_x = env.width / 2.0

    # Split points into territories based on the number of guides.
    territories = [[] for _ in range(n_guides)]

    if n_guides == 1:
        territories[0] = all_points
    elif n_guides == 2:
        for p in all_points:
            if p[1] < center_y:
                territories[0].append(p)
            else:
                territories[1].append(p)
    elif n_guides >= 3:
        # Quadrants: top-left, top-right, bottom-left, bottom-right.
        for p in all_points:
            if p[1] < center_y:
                if p[0] < center_x:
                    territories[0 % n_guides].append(p)
                else:
                    territories[1 % n_guides].append(p)
            else:
                if p[0] < center_x:
                    territories[2 % n_guides].append(p)
                else:
                    territories[3 % n_guides].append(p)

    # Shared coordination blackboard for every Sweeper in this run.
    blackboard = SweeperBlackboard()

    guides = []
    for i in range(n_guides):
        ex, ey = exits[i % len(exits)]
        # Use this guide's territory; fall back to all points if empty.
        pts = territories[i] if territories[i] else all_points
        # Order farthest first so the guide moves deep into the building.
        pts_sorted = sorted(pts, key=lambda p: -np.linalg.norm(p - np.array([ex, ey])))
        guide = SweeperGuide(
            ex,
            ey,
            speed,
            env,
            rng,
            patrol_points=pts_sorted,
            n_civilians=n_civilians,
            n_guides=n_guides,
            blackboard=blackboard,
        )
        guides.append(guide)

    return guides
