"""SweeperGuide — reactive patrol + escort policy (L2 reactive)."""
import math
import numpy as np
from collections import deque
from evacsim.environment import Environment, CellType
from evacsim.smoke import SmokeSystem
from evacsim.fire import FireSystem
from evacsim.guides.base import Guide
from evacsim.guides.common import bfs_path


def _find_patrol_points(env: Environment):
    """Find patrol points using hybrid room detection + grid scan.

    Strategy:
      - Bottleneck maps (many rooms): use room centers as patrol points
      - Open plan maps (few/no rooms): use denser grid scan for coverage
      - Always de-duplicate to avoid overlapping patrol points
    """
    rows, cols = env.grid.shape
    points = []

    # 1. Try room detection first
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

    # 2. Grid scan — density depends on whether rooms were found
    if len(rooms) >= 2:
        # Bottleneck: rooms found. Add sparse corridor points as supplement
        row_step, col_step = 5, 10
    else:
        # Open plan: no/few rooms. Use DENSE grid scan for full coverage
        row_step, col_step = 4, 6

    for r in range(2, rows - 2, row_step):
        for c in range(2, cols - 2, col_step):
            if env.grid[r, c] != CellType.WALL:
                wx, wy = env.grid_to_world(r, c)
                points.append(np.array([wx, wy], dtype=float))

    # De-duplicate: remove points within 3 cells of each other
    if not points:
        return points
    filtered = [points[0]]
    min_dist = env.cell_size * 3
    for p in points[1:]:
        if all(np.linalg.norm(p - f) >= min_dist for f in filtered):
            filtered.append(p)
    return filtered


def _balanced_patrol_order(start_pos, waypoints):
    """Order waypoints so the guide alternates between different areas.

    Split points into "above start" and "below start" (by y-coordinate),
    then interleave them. Within each group, order by nearest-first.
    This ensures the guide visits both halves of the building, not just
    the nearest half."""
    if not waypoints:
        return []

    sy = start_pos[1]
    above = [p for p in waypoints if p[1] < sy - 5]  # Above start (lower y)
    below = [p for p in waypoints if p[1] > sy + 5]  # Below start (higher y)
    middle = [p for p in waypoints if abs(p[1] - sy) <= 5]  # Near start

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

    # Interleave: above point, below point, above, below...
    # Start with middle points, then alternate
    result = list(middle_sorted)
    for i in range(max(len(above_sorted), len(below_sorted))):
        if i < len(above_sorted):
            result.append(above_sorted[i])
        if i < len(below_sorted):
            result.append(below_sorted[i])

    return result


class SweeperGuide(Guide):
    """
    Sweeper policy: collect civilians into a group, escort them to exit.

    1. PATROL: walk through building visiting rooms via BFS
    2. COLLECT: at each room, recruit nearby civilians (up to max_group)
    3. ESCORT: lead group to nearest exit; civilians use boids to follow
    4. RELEASE: at exit, civilians get BFS path to exit and leave group
    5. RETURN: go back for more

    Coordination: when a Sweeper finishes clearing its entire territory,
    it adopts uncleared patrol points from a peer Sweeper that still has
    work. This mirrors real fire brigade radio coordination: 'Zone 1 clear,
    moving to assist Zone 2.' Preserves Sweeper's systematic identity while
    preventing idle guides when territories are unbalanced.

    Spawns from exits (like real firefighters entering the building).
    """

    # Class-level registry of all active Sweepers (populated by spawn function).
    # Used for territory reallocation between peer Sweepers.
    _all_sweepers: list = []

    COLLECTING = "collecting"
    ESCORTING = "escorting"
    RELEASING = "releasing"  # Waiting at exit for group to arrive and escape
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
    ):
        super().__init__(x, y, speed, env, rng)

        self._group = []

        # Dynamic commitment ceiling (first-principles derivation):
        # Optimal batch size n* ∝ √(territory_population) from minimizing
        # C(n) = Pβ/n + P·h·n (fixed trip overhead vs quadratic danger exposure).
        # α=3 calibrated at crowd=100 via McCormack & Chen (2014) 1:20 ratio.
        # Floor=5 matches min_group efficiency threshold (escort overhead).
        self._base_max_group = round(3 * math.sqrt(n_civilians / max(1, n_guides)))
        self._floor = 5
        self._tick = 0
        self._cached_effective_max = self._base_max_group
        self._cached_max_tick = -1

        # Build patrol route
        all_pts = patrol_points if patrol_points else _find_patrol_points(env)
        self._patrol_route = _balanced_patrol_order(self.pos, all_pts)
        self._patrol_idx = 0

        # Track cleared rooms — skip them until full cycle completes
        self._cleared_points = set()
        self._cycle_start_idx = 0  # Where this patrol cycle started

        self._waypoints = []
        self._phase = self.COLLECTING

        # Wait at each patrol point to collect civilians
        self._wait_ticks = 15  # Quick check — civilians spot exits themselves now
        self._wait_counter = 0
        self._waiting = False

        # Target exit for escorting
        self._escort_exit = None

        # Releasing: wait at exit for stragglers (shorter — civilians self-release at exits)
        self._release_wait = 0
        self._release_max_wait = 30

        # Start patrol
        if self._patrol_route:
            self._waypoints = bfs_path(env, self.pos, self._patrol_route[0])

    def _effective_max_group(self) -> int:
        """Dynamic commitment ceiling via T-S fuzzy inference (reactive).

        Derived from first-principles optimization of collection-escort cycle:
        fixed trip overhead β vs quadratic danger exposure h·n² yields
        n* ∝ √(territory_population). T-S fuzzy adjusts this base using
        CURRENT local danger (reactive architecture — Brooks 1991).

        Two triangular membership functions (SAFE, DANGEROUS) with zero-order
        T-S consequents collapse to linear interpolation:
            effective = base − d × (base − floor)
        where d ∈ [0, 1] is normalized peak danger in 3-cell radius.
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

        # Normalize to [0, 1] using existing emergency thresholds
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

        self._clean_group()

        # Always recruit nearby civilians while moving
        self._recruit_nearby(env, smoke, civilians)

        if self._phase == self.COLLECTING:
            self._do_collecting(env, smoke, civilians)
        elif self._phase == self.ESCORTING:
            self._do_escorting(env, smoke, civilians)
        elif self._phase == self.RELEASING:
            self._do_releasing(env)
        elif self._phase == self.RETURNING:
            self._do_returning(env, smoke, civilians)

    # ── Phase behaviors ──

    def _do_collecting(self, env, smoke, civilians):
        """Patrol rooms, collect civilians."""
        if len(self._group) >= self._effective_max_group():
            self._start_escort(env)
            return

        # DANGER CHECK: proactive detection — check guide surroundings + group members
        if self._group:
            danger = False
            sg = getattr(self, "_smoke_grid", None)
            fg = getattr(self, "_fire_grid", None)
            rows, cols = env.grid.shape

            # Check 2-cell radius around guide (proactive — detect approaching smoke)
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
                self._start_escort(env)
                return

        if self._waiting:
            self._wait_counter += 1
            if self._wait_counter >= self._wait_ticks:
                self._waiting = False
                recruitable = self._count_recruitable(env, smoke, civilians)
                if recruitable == 0:
                    # Nobody here — mark this point as cleared
                    self._cleared_points.add(self._patrol_idx)
                if self._group and recruitable == 0:
                    # Nobody else here. Escort or keep collecting?
                    # Only escort if group has decent size (5+).
                    # Otherwise continue to next patrol point — collect more.
                    if len(self._group) >= 5:
                        self._start_escort(env)
                        return
                    # Small group — keep them, move to next point
                self._advance_patrol(env)
            return

        if not self._waypoints:
            self._waiting = True
            self._wait_counter = 0
            self.vel = np.zeros(2)
        else:
            self._follow_path(env)

    def _do_escorting(self, env, smoke, civilians):
        """Lead group to exit. Pick up anyone on the way (no limit during escort).
        Includes danger detection: if fire/smoke threatens group, reroute to
        nearest safe exit immediately."""
        if not self._group:
            self._phase = self.RETURNING
            self._advance_patrol(env)
            return

        # DANGER CHECK: if fire/smoke near guide or group members, reroute
        # to nearest safe exit. Prevents leading civilians through fire.
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
            # Reroute to nearest safe exit (same logic as _start_escort)
            self._start_escort(env)

        # Pick up anyone we pass — no group limit during escort
        self._recruit_nearby_escort(env, smoke, civilians)
        if not self._waypoints:
            if self._escort_exit is not None:
                dist = np.linalg.norm(self.pos - np.array(self._escort_exit))
                if dist < env.cell_size * 2:
                    # Arrived at exit — start RELEASING (wait for group)
                    self._phase = self.RELEASING
                    self._release_wait = 0
                    self.vel = np.zeros(2)
                    return
            self.vel = np.zeros(2)
        else:
            self._follow_path(env)

    def _do_releasing(self, env):
        """Stand at exit and actively direct civilians to exit.
        Guide verbally tells nearby civilians where the exit is,
        then gives final directions to stragglers before departing."""
        from evacsim.civilian import FOLLOWING_GUIDE

        self.vel = np.zeros(2)
        self._release_wait += 1

        # Active release: direct civilians to exit (verbal communication)
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

        # Done waiting? (all released OR timeout)
        if not self._group or self._release_wait >= self._release_max_wait:
            # Timeout: give stragglers exit target instead of leaving them lost
            for civ in self._group:
                if civ.state == FOLLOWING_GUIDE:
                    civ._set_exit_target_force(env, self._escort_exit)
            self._group.clear()
            self._phase = self.RETURNING
            self._advance_patrol(env)

    def _do_returning(self, env, smoke, civilians):
        """Go back for more civilians."""
        if len(self._group) >= self._effective_max_group():
            self._start_escort(env)
            return

        # If we picked up civilians — check danger (guide surroundings + group members)
        if self._group:
            danger = False
            # Check group members first (they're vulnerable)
            for civ in self._group:
                if smoke.get_density_at(civ.pos[0], civ.pos[1]) > 0.3:
                    danger = True
                    break
            # Check guide surroundings (2-cell radius)
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

    # ── Recruit & release ──

    def _recruit_nearby(self, env, smoke, civilians):
        """Recruit nearby civilians using visual range (Jin, 1978).
        Range based on visibility of light-emitting objects in smoke.
        Requires line of sight — walls block recruitment."""
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
                # Wake up idle/reacting civilians
                if civ.state == IDLE:
                    civ.state = REACTING
                    civ._reaction_counter = 0
                    civ._reaction_delay = min(civ._reaction_delay, 5)
                    continue  # Wait for reaction to finish
                elif civ.state == REACTING:
                    civ._reaction_delay = min(civ._reaction_delay, 5)
                    continue  # Still reacting, not ready yet
                else:
                    # Active civilian — add to group
                    self._group.append(civ)
                    civ.assigned_guide = self
                    civ.state = FOLLOWING_GUIDE

    def _recruit_nearby_escort(self, env, smoke, civilians):
        """Recruit anyone nearby during escort — NO group limit.
        If you're leading people to the exit and pass more civilians, take them."""
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
        Must match what _recruit_nearby can reach — no false positives."""
        radius = self.get_visual_recruit_radius(smoke)
        from evacsim.civilian import (
            MOVING_TO_EXIT,
            ESCAPED,
            DEAD,
            IDLE,
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
        """Begin leading group to nearest SAFE exit.
        Avoids exits that are in or near fire."""
        self._phase = self.ESCORTING
        fg = getattr(self, "_fire_grid", None)

        best_exit, best_score = None, float("inf")
        for ex, ey in self.known_exits:
            dist = np.hypot(self.pos[0] - ex, self.pos[1] - ey)
            # Penalize exits near fire or heavy smoke
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
            # Escorting group — smoke is full danger (civilians vulnerable)
            # Must match _follow_path's reroute weight (sg * 1.0 when has_group)
            sg = getattr(self, "_smoke_grid", None)
            if fg is not None and sg is not None:
                danger = np.maximum(fg, sg * 1.0)
            elif fg is not None:
                danger = fg
            elif sg is not None:
                danger = sg * 1.0
            else:
                danger = None
            self._waypoints = bfs_path(
                env, self.pos, np.array(best_exit), fire_grid=danger
            )

    # _release_group_at_exit removed — replaced by _do_releasing phase
    # Guide now waits at exit and releases civilians one by one as they arrive

    def _release_group_no_exit(self):
        """Release group without exit info (guide died/evacuating)."""
        from evacsim.civilian import WANDERING, FOLLOWING_GUIDE

        for civ in self._group:
            if civ.state == FOLLOWING_GUIDE:
                civ.assigned_guide = None
                civ.state = WANDERING
        self._group.clear()

    def _clean_group(self):
        """Remove dead/escaped civilians from group."""
        from evacsim.civilian import ESCAPED, DEAD

        self._group = [c for c in self._group if c.state not in (ESCAPED, DEAD)]

    def _try_reallocate_to_peer(self, env):
        """Territory reallocation: when this guide's own territory is fully
        cleared, adopt uncleared patrol points from a peer Sweeper that still
        has work. Mirrors fire brigade radio coordination. Returns True if
        reallocation succeeded (patrol_route and waypoints updated)."""
        best_peer = None
        best_uncleared = 0
        for peer in SweeperGuide._all_sweepers:
            if peer is self or not peer.alive:
                continue
            uncleared = sum(
                1 for i in range(len(peer._patrol_route))
                if i not in peer._cleared_points
            )
            if uncleared > best_uncleared:
                best_uncleared = uncleared
                best_peer = peer

        # Only reallocate if peer has enough work to share (avoid ping-pong)
        if best_peer is None or best_uncleared < 3:
            return False

        # Take half of peer's uncleared points (both guides keep working)
        peer_uncleared_indices = [
            i for i in range(len(best_peer._patrol_route))
            if i not in best_peer._cleared_points
        ]
        n_take = max(1, len(peer_uncleared_indices) // 2)
        taken_indices = peer_uncleared_indices[:n_take]

        # Copy those patrol points to our new route
        new_points = [best_peer._patrol_route[i] for i in taken_indices]

        # Mark them as cleared in peer's set so peer doesn't revisit
        for idx in taken_indices:
            best_peer._cleared_points.add(idx)

        # Replace our patrol route with the new (adopted) territory
        self._patrol_route = new_points
        self._cleared_points = set()
        self._patrol_idx = 0
        self._cycle_start_idx = 0

        # Compute waypoints to new first point (danger-aware)
        fg = getattr(self, "_fire_grid", None)
        sg = getattr(self, "_smoke_grid", None)
        if self._group:
            if fg is not None and sg is not None:
                danger = np.maximum(fg, sg * 1.0)
            elif fg is not None:
                danger = fg
            elif sg is not None:
                danger = sg * 1.0
            else:
                danger = None
        else:
            if fg is not None and sg is not None:
                danger = np.maximum(fg, sg * 0.3)
            elif fg is not None:
                danger = fg
            elif sg is not None:
                danger = sg * 0.3
            else:
                danger = None
        self._waypoints = bfs_path(
            env, self.pos, self._patrol_route[0], fire_grid=danger
        )
        return True

    def _advance_patrol(self, env):
        """Move to next patrol point. Skip cleared and dangerous points."""
        if not self._patrol_route:
            return

        fg = getattr(self, "_fire_grid", None)
        sg = getattr(self, "_smoke_grid", None)

        # Try up to len(route) points to find a valid one
        for _ in range(len(self._patrol_route)):
            self._patrol_idx = (self._patrol_idx + 1) % len(self._patrol_route)

            # Full cycle completed? Try territory reallocation before reset.
            # If we can adopt a peer's uncleared points, do that instead of
            # re-walking our own (already empty) territory.
            if self._patrol_idx == self._cycle_start_idx:
                if self._try_reallocate_to_peer(env):
                    return  # Waypoints already set by reallocation
                self._cleared_points.clear()
                self._cycle_start_idx = self._patrol_idx

            # Skip cleared points (visited, nobody there)
            if self._patrol_idx in self._cleared_points:
                continue

            target = self._patrol_route[self._patrol_idx]
            tr, tc = env.world_to_grid(target[0], target[1])

            # Skip patrol points that are now in fire or heavy smoke
            is_dangerous = False
            if fg is not None and 0 <= tr < fg.shape[0] and 0 <= tc < fg.shape[1]:
                if fg[tr, tc] > 0.2:
                    is_dangerous = True
            if sg is not None and 0 <= tr < sg.shape[0] and 0 <= tc < sg.shape[1]:
                if sg[tr, tc] > 0.5:
                    is_dangerous = True

            if not is_dangerous:
                break  # Found a safe patrol point

        # Compute path — avoid smoke if carrying group, light smoke avoidance solo
        if self._group:
            if fg is not None and sg is not None:
                danger = np.maximum(fg, sg * 1.0)
            elif fg is not None:
                danger = fg
            elif sg is not None:
                danger = sg * 1.0
            else:
                danger = None
        else:
            # Solo: avoid fire + light smoke (guides take smoke damage too)
            if fg is not None and sg is not None:
                danger = np.maximum(fg, sg * 0.3)
            elif fg is not None:
                danger = fg
            elif sg is not None:
                danger = sg * 0.3
            else:
                danger = None
        self._waypoints = bfs_path(
            env, self.pos, self._patrol_route[self._patrol_idx], fire_grid=danger
        )


def spawn_sweeper_guides(
    env: Environment, n_guides: int, speed: float = 2.0, seed: int = 77,
    n_civilians: int = 100
) -> list[SweeperGuide]:
    """Spawn sweeper guides from exits. Each covers a different territory.

    Splits patrol points geographically so guides don't overlap:
    - 2 guides: one covers top half, one covers bottom half
    - 4 guides: each covers a quadrant
    This ensures the whole building gets coverage."""
    rng = np.random.default_rng(seed)
    exits = list(env.exits)
    all_points = _find_patrol_points(env)

    if not all_points:
        return []

    # Find map center
    center_y = env.height / 2.0
    center_x = env.width / 2.0

    # Split points into territories based on number of guides
    territories = [[] for _ in range(n_guides)]

    if n_guides == 1:
        territories[0] = all_points
    elif n_guides == 2:
        # Top half / bottom half
        for p in all_points:
            if p[1] < center_y:
                territories[0].append(p)
            else:
                territories[1].append(p)
    elif n_guides >= 3:
        # Quadrants: top-left, top-right, bottom-left, bottom-right
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

    # Reset the class-level registry for this new simulation (prevents
    # pollution between experimental runs).
    SweeperGuide._all_sweepers = []

    guides = []
    for i in range(n_guides):
        ex, ey = exits[i % len(exits)]
        # Get this guide's territory, fall back to all points if empty
        pts = territories[i] if territories[i] else all_points
        # Order: farthest points first (so guide goes deep into building first)
        pts_sorted = sorted(pts, key=lambda p: -np.linalg.norm(p - np.array([ex, ey])))
        guide = SweeperGuide(ex, ey, speed, env, rng, patrol_points=pts_sorted,
                             n_civilians=n_civilians, n_guides=n_guides)
        guides.append(guide)
        SweeperGuide._all_sweepers.append(guide)

    return guides
