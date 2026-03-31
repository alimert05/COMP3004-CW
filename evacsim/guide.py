"""
Guide agents — intelligent agents that help civilians find exits.

Guide types:
  - BeaconGuide: Stands near an exit and broadcasts position.
    Civilians within hearing range detect the guide even through smoke.
    Think of it as a firefighter with a megaphone standing by the exit.

Guides are the "intelligent agents" in this simulation.
They have full knowledge of the map (know where exits are),
are immune to smoke damage (trained + equipment), and can
see through smoke better than civilians.
"""

import numpy as np
from collections import deque
from evacsim.environment import Environment, CellType
from evacsim.smoke import SmokeSystem
from evacsim.fire import FireSystem


def bfs_path(env: Environment, start_world, goal_world, fire_grid=None):
    """Find a path from start to goal using BFS on the grid.

    If fire_grid is provided, first tries to find a path avoiding fire
    cells (density > 0.3). If no safe path exists, retries allowing fire.
    This way agents avoid fire when possible but can still escape through
    it as a last resort.

    Returns a list of world-coordinate waypoints, or empty list if no path."""

    def _bfs(avoid_fire):
        sr, sc = env.world_to_grid(start_world[0], start_world[1])
        gr, gc = env.world_to_grid(goal_world[0], goal_world[1])
        rows, cols = env.grid.shape

        if sr == gr and sc == gc:
            return []

        visited = {(sr, sc): None}
        queue = deque([(sr, sc)])

        while queue:
            r, c = queue.popleft()

            if r == gr and c == gc:
                path = []
                current = (r, c)
                while current is not None:
                    path.append(current)
                    current = visited[current]
                path.reverse()

                waypoints = []
                for pr, pc in path[1:]:
                    wx, wy = env.grid_to_world(pr, pc)
                    waypoints.append(np.array([wx, wy], dtype=float))
                return waypoints

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    if env.grid[nr, nc] == CellType.WALL:
                        continue
                    # Avoid fire cells if requested
                    if avoid_fire and fire_grid is not None:
                        if fire_grid[nr, nc] > 0.3:
                            continue
                    visited[(nr, nc)] = (r, c)
                    queue.append((nr, nc))

        return None  # No path (None = truly no path, [] = already there)

    # First try: avoid fire
    if fire_grid is not None:
        result = _bfs(avoid_fire=True)
        if result is not None:
            return result
        # No safe path — retry allowing fire (last resort)
        result = _bfs(avoid_fire=False)
        return result if result is not None else []
    else:
        result = _bfs(avoid_fire=False)
        return result if result is not None else []


class Guide:
    """Base class for all guide types."""

    def __init__(self, x: float, y: float, speed: float,
                 env: Environment, rng: np.random.Generator):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.array([0.0, 0.0])
        self.speed = speed
        self.env = env
        self._rng = rng

        # Guides know ALL exit locations (trained, have map)
        self.known_exits = list(env.exits)

        # Guides can broadcast their position to nearby civilians
        # This is much larger than civilian vision (megaphone, whistle, torch)
        self.broadcast_radius = 80.0  # 8 cells

        # Guides are alive unless fire kills them (they have gas masks for smoke)
        self.alive = True
        self._fire_damage = 0.0
        self._max_fire_damage = 300.0  # Much tougher than civilians

        # Also take smoke damage (less than civilians, but still accumulates)
        self._smoke_damage = 0.0
        self._max_smoke_damage = 250.0

        # Total health = fire + smoke combined
        # When health drops below threshold, guide abandons mission and escapes
        self._health_threshold = 0.6  # 60% — start escaping at 40% health remaining

        # Self-evacuation state
        self._self_evacuating = False
        self._escape_waypoints = []

        # Which exit this guide is assigned to
        self.assigned_exit = None

    @property
    def health(self) -> float:
        """Current health as 0.0-1.0 ratio. 1.0 = full health, 0.0 = dead."""
        fire_ratio = self._fire_damage / self._max_fire_damage
        smoke_ratio = self._smoke_damage / self._max_smoke_damage
        # Worst of the two damage types determines health
        return max(0.0, 1.0 - max(fire_ratio, smoke_ratio))

    def update(self, env: Environment, smoke: SmokeSystem,
               fire: FireSystem, civilians: list) -> None:
        """Base update — handles damage and self-evacuation.
        Override in subclasses but call super().update() first."""
        if not self.alive:
            return

        # Store grids for danger-aware BFS
        self._fire_grid = fire.density
        self._smoke_grid = smoke.density

        # Check fire damage (guides resist smoke but not fire)
        local_fire = fire.get_density_at(self.pos[0], self.pos[1])
        local_smoke = smoke.get_density_at(self.pos[0], self.pos[1])

        if local_fire > 0.3:
            self._fire_damage += 2.0
        if local_smoke > 0.3:
            self._smoke_damage += local_smoke * 0.5  # Half damage vs civilians

        # Check for death
        if self._fire_damage >= self._max_fire_damage or self._smoke_damage >= self._max_smoke_damage:
            self.alive = False
            return

        # Check health threshold — start self-evacuating
        if not self._self_evacuating and self.health < self._health_threshold:
            self._start_self_evacuation(env)

    def _start_self_evacuation(self, env: Environment) -> None:
        """Guide's health is critical — abandon mission, head to nearest exit."""
        self._self_evacuating = True

        # Find nearest exit
        best_exit = None
        best_dist = float("inf")
        for ex, ey in self.known_exits:
            d = np.sqrt((self.pos[0] - ex) ** 2 + (self.pos[1] - ey) ** 2)
            if d < best_dist:
                best_dist = d
                best_exit = np.array([ex, ey], dtype=float)

        if best_exit is not None:
            fg = getattr(self, '_fire_grid', None)
            self._escape_waypoints = bfs_path(env, self.pos, best_exit, fire_grid=fg)

    def _do_self_evacuation(self, env: Environment) -> bool:
        """Follow escape path to exit. Returns True if currently self-evacuating."""
        if not self._self_evacuating:
            return False

        # Check if reached exit
        for ex, ey in self.known_exits:
            dist = np.sqrt((self.pos[0] - ex) ** 2 + (self.pos[1] - ey) ** 2)
            if dist < env.cell_size:
                self.alive = False  # Guide escaped (remove from simulation)
                return True

        # Follow escape waypoints
        if self._escape_waypoints:
            target = self._escape_waypoints[0]
            diff = target - self.pos
            dist = np.linalg.norm(diff)

            if dist < env.cell_size * 0.5:
                self._escape_waypoints.pop(0)
                if self._escape_waypoints:
                    target = self._escape_waypoints[0]
                    diff = target - self.pos
                    dist = np.linalg.norm(diff)

            if dist > 0:
                direction = diff / dist
                self.vel = direction * self.speed * 1.5  # Sprint to exit

            # Move
            vx, vy = self.vel
            x, y = self.pos
            if env.is_walkable(x + vx, y + vy):
                self.pos = np.array([x + vx, y + vy])
            elif env.is_walkable(x + vx, y):
                self.pos = np.array([x + vx, y])
            elif env.is_walkable(x, y + vy):
                self.pos = np.array([x, y + vy])

        return True

    def get_broadcast_radius(self, smoke: SmokeSystem) -> float:
        """Broadcast radius reduced by smoke but still much better than civilian vision.
        Even in heavy smoke, guide can be heard from 3 cells away."""
        local_smoke = smoke.get_density_at(self.pos[0], self.pos[1])
        # Minimum 30 units (3 cells) even in full smoke
        return max(30.0, self.broadcast_radius * (1.0 - local_smoke * 0.5))

    def _follow_path(self, env: Environment) -> None:
        """Follow the pre-computed waypoint path.
        Checks ahead for fire AND heavy smoke — reroutes if dangerous.
        Guides resist smoke but their civilian group does not."""
        if not self._waypoints:
            self.vel = np.array([0.0, 0.0])
            return

        # Check next 10 waypoints for fire or heavy smoke
        fg = getattr(self, '_fire_grid', None)
        sg = getattr(self, '_smoke_grid', None)
        if fg is not None or sg is not None:
            needs_reroute = False
            for wp in self._waypoints[:10]:
                wr, wc = env.world_to_grid(wp[0], wp[1])
                # Fire check
                if fg is not None and 0 <= wr < fg.shape[0] and 0 <= wc < fg.shape[1]:
                    if fg[wr, wc] > 0.2:
                        needs_reroute = True
                        break
                # Heavy smoke check (dangerous for civilians in group)
                if sg is not None and 0 <= wr < sg.shape[0] and 0 <= wc < sg.shape[1]:
                    if sg[wr, wc] > 0.6:
                        needs_reroute = True
                        break
            if needs_reroute:
                # Combine fire + smoke into danger grid for BFS avoidance
                danger = fg if fg is not None else np.zeros_like(env.grid, dtype=float)
                if sg is not None:
                    danger = np.maximum(danger, sg * 0.5)  # Smoke counts as half-fire
                final = self._waypoints[-1]
                self._waypoints = bfs_path(env, self.pos, final, fire_grid=danger)
                if not self._waypoints:
                    self.vel = np.array([0.0, 0.0])
                    return

        target = self._waypoints[0]

        diff = target - self.pos
        dist = np.linalg.norm(diff)

        if dist < self.env.cell_size * 0.5:
            self._waypoints.pop(0)
            if not self._waypoints:
                self.vel = np.array([0.0, 0.0])
                return
            target = self._waypoints[0]
            diff = target - self.pos
            dist = np.linalg.norm(diff)

        if dist > 0:
            direction = diff / dist
            self.vel = direction * self.speed

        # Move
        vx, vy = self.vel
        x, y = self.pos
        if env.is_walkable(x + vx, y + vy):
            self.pos = np.array([x + vx, y + vy])
        elif env.is_walkable(x + vx, y):
            self.pos = np.array([x + vx, y])
        elif env.is_walkable(x, y + vy):
            self.pos = np.array([x, y + vy])


class BeaconGuide(Guide):
    """
    Beacon policy: stand near an assigned exit and broadcast.

    Behaviour:
      1. On creation, assigned to an exit
      2. Uses BFS pathfinding to navigate through corridors to the exit
      3. Stands 2-3 cells in front of exit and broadcasts
      4. Any civilian within hearing range learns the exit location

    Why it helps:
      - Civilians with 1-cell vision would never find exits on their own
      - Guide's broadcast (8 cells) reaches civilians in corridors
      - Even in smoke (3-cell broadcast), much better than blind civilians
    """

    def __init__(self, x: float, y: float, speed: float,
                 env: Environment, rng: np.random.Generator,
                 assigned_exit: tuple = None):
        super().__init__(x, y, speed, env, rng)

        # Assign to nearest exit if not specified
        if assigned_exit is not None:
            self.assigned_exit = np.array(assigned_exit, dtype=float)
        else:
            self.assigned_exit = self._find_nearest_exit()

        # Find beacon position (2-3 cells inward from exit)
        self._beacon_pos = self._find_beacon_position()

        # Compute BFS path from spawn to beacon position
        self._waypoints = bfs_path(env, self.pos, self._beacon_pos)

        self._in_position = False

    def _find_nearest_exit(self) -> np.ndarray:
        """Find the exit closest to spawn position."""
        best = None
        best_dist = float("inf")
        for ex, ey in self.known_exits:
            dist = np.sqrt((self.pos[0] - ex) ** 2 + (self.pos[1] - ey) ** 2)
            if dist < best_dist:
                best_dist = dist
                best = np.array([ex, ey], dtype=float)
        return best

    def _find_beacon_position(self) -> np.ndarray:
        """Find a walkable position 2-3 cells inward from the exit.
        Uses BFS from the exit cell to find the best spot inside the building."""
        ex, ey = self.assigned_exit
        gr, gc = self.env.world_to_grid(ex, ey)
        rows, cols = self.env.grid.shape

        visited = set()
        queue = deque()
        queue.append((gr, gc, 0))
        visited.add((gr, gc))

        best_pos = self.assigned_exit.copy()  # fallback
        best_dist = 0

        while queue:
            r, c, d = queue.popleft()
            if d > 4:
                continue

            # We want cells 2-3 steps away that are FLOOR (not exit, not wall)
            if 2 <= d <= 3 and self.env.grid[r, c] == CellType.FLOOR:
                if d > best_dist:
                    best_dist = d
                    bx, by = self.env.grid_to_world(r, c)
                    best_pos = np.array([bx, by])

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    if self.env.grid[nr, nc] != CellType.WALL:
                        visited.add((nr, nc))
                        queue.append((nr, nc, d + 1))

        return best_pos

    def update(self, env: Environment, smoke: SmokeSystem,
               fire: FireSystem, civilians: list) -> None:
        """Move to beacon position via BFS path, then stand and broadcast."""
        super().update(env, smoke, fire, civilians)
        if not self.alive:
            return

        # Self-evacuation overrides everything
        if self._do_self_evacuation(env):
            return

        # --- Phase 1: Follow path to beacon position ---
        if not self._in_position:
            if not self._waypoints:
                # Path done — lock in position
                self._in_position = True
                self.vel = np.array([0.0, 0.0])
            else:
                self._follow_path(env)
            return

        # --- Phase 2: In position — broadcast to nearby civilians ---
        radius = self.get_broadcast_radius(smoke)

        from evacsim.civilian import MOVING_TO_EXIT, ESCAPED, DEAD, IDLE, REACTING

        exit_pos = (self.assigned_exit[0], self.assigned_exit[1])

        for civ in civilians:
            if civ.state in (MOVING_TO_EXIT, ESCAPED, DEAD):
                continue

            dist = np.linalg.norm(civ.pos - self.pos)
            if dist < radius:
                if civ.state == IDLE:
                    # Fix #1: Wake up idle civilians — guide is shouting!
                    civ.state = REACTING
                    civ._reaction_counter = 0
                    civ._reaction_delay = min(civ._reaction_delay, 10)
                    civ._guide_told_exit = exit_pos  # Remember exit for after reaction
                elif civ.state == REACTING:
                    # Fix #2: Speed up reaction — hearing a guide snaps you out faster
                    civ._reaction_delay = min(civ._reaction_delay, 5)
                    civ._guide_told_exit = exit_pos
                else:
                    # Active civilian — give them the exit location
                    civ._set_exit_target(env, exit_pos)


def spawn_beacon_guides(env: Environment, n_guides: int, speed: float = 2.0,
                        seed: int = 77) -> list[BeaconGuide]:
    """Spawn beacon guides AT the exits (realistic — firefighters enter from outside).

    Each guide starts at an assigned exit and walks inward to their beacon position.
    If more guides than exits, multiple guides share exits (round-robin).
    """
    rng = np.random.default_rng(seed)
    exits = list(env.exits)

    guides = []
    for i in range(n_guides):
        # Round-robin assign exits
        assigned_exit = exits[i % len(exits)]
        ex, ey = assigned_exit
        # Spawn AT the exit position
        guide = BeaconGuide(ex, ey, speed, env, rng, assigned_exit=assigned_exit)
        guides.append(guide)

    return guides


# ════════════════════════════════════════════════════════════════════
# SWEEPER GUIDE — collect-and-escort with boids
# ════════════════════════════════════════════════════════════════════

def _find_patrol_points(env: Environment):
    """Find patrol points using room detection + grid scan.

    Uses env.detect_rooms() to get actual room centers, then adds
    corridor points via grid scan. This ensures every room in the
    building gets a patrol point regardless of grid alignment."""
    rows, cols = env.grid.shape
    points = []

    # 1. Add center of each detected room
    try:
        rooms = env.detect_rooms()
        for room_cells in rooms:
            if not room_cells:
                continue
            # Find center of room
            avg_r = sum(r for r, c in room_cells) / len(room_cells)
            avg_c = sum(c for r, c in room_cells) / len(room_cells)
            # Find the walkable cell nearest to center
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
        pass  # Fall back to grid scan only

    # 2. Add corridor points via grid scan (every 5 rows for better coverage)
    for r in range(2, rows - 2, 5):
        for c in range(2, cols - 2, 10):
            if env.grid[r, c] != CellType.WALL:
                wx, wy = env.grid_to_world(r, c)
                points.append(np.array([wx, wy], dtype=float))

    # De-duplicate: remove points within 4 cells of each other
    if not points:
        return points
    filtered = [points[0]]
    min_dist = env.cell_size * 4
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
    below = [p for p in waypoints if p[1] > sy + 5]   # Below start (higher y)
    middle = [p for p in waypoints if abs(p[1] - sy) <= 5]  # Near start

    def greedy_sort(pts, start):
        remaining = list(pts)
        ordered = []
        current = start.copy()
        while remaining:
            best_idx = min(range(len(remaining)),
                           key=lambda i: np.linalg.norm(current - remaining[i]))
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

    Spawns from exits (like real firefighters entering the building).
    """

    COLLECTING = "collecting"
    ESCORTING = "escorting"
    RELEASING = "releasing"   # Waiting at exit for group to arrive and escape
    RETURNING = "returning"

    def __init__(self, x: float, y: float, speed: float,
                 env: Environment, rng: np.random.Generator,
                 patrol_points: list = None, max_group: int = 8):
        super().__init__(x, y, speed, env, rng)

        self._group = []
        self._max_group = max_group

        # Build patrol route
        all_pts = patrol_points if patrol_points else _find_patrol_points(env)
        self._patrol_route = _balanced_patrol_order(self.pos, all_pts)
        self._patrol_idx = 0

        self._waypoints = []
        self._phase = self.COLLECTING

        # Wait at each patrol point to collect civilians
        self._wait_ticks = 20
        self._wait_counter = 0
        self._waiting = False

        # Target exit for escorting
        self._escort_exit = None

        # Releasing: wait at exit for stragglers
        self._release_wait = 0
        self._release_max_wait = 60  # Max 60 ticks (2 seconds) at exit

        # Start patrol
        if self._patrol_route:
            self._waypoints = bfs_path(env, self.pos, self._patrol_route[0])

    def update(self, env: Environment, smoke: SmokeSystem,
               fire: FireSystem, civilians: list) -> None:
        super().update(env, smoke, fire, civilians)
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
        if len(self._group) >= self._max_group:
            self._start_escort(env)
            return

        if self._waiting:
            self._wait_counter += 1
            if self._wait_counter >= self._wait_ticks:
                self._waiting = False
                if self._group:
                    # Got some people — check if more nearby
                    if self._count_recruitable(env, smoke, civilians) == 0:
                        self._start_escort(env)
                        return
                # Next patrol point
                self._advance_patrol(env)
            return

        if not self._waypoints:
            self._waiting = True
            self._wait_counter = 0
            self.vel = np.zeros(2)
        else:
            self._follow_path(env)

    def _do_escorting(self, env, smoke, civilians):
        """Lead group to exit."""
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
        """Stand at exit, release group members as they arrive.
        Like a firefighter holding the door open, counting heads."""
        import numpy as np
        from evacsim.civilian import FOLLOWING_GUIDE, WANDERING

        self.vel = np.zeros(2)  # Stand still at exit
        self._release_wait += 1

        # Each tick: release any group member close enough to the exit
        still_in_group = []
        for civ in self._group:
            if civ.state != FOLLOWING_GUIDE:
                continue
            dist_to_exit = np.linalg.norm(civ.pos - np.array(self._escort_exit))
            dist_to_guide = np.linalg.norm(civ.pos - self.pos)
            if dist_to_exit < env.cell_size * 5 or dist_to_guide < env.cell_size * 3:
                # Close enough — give them the exit
                civ._set_exit_target_force(env, self._escort_exit)
            else:
                still_in_group.append(civ)

        self._group = still_in_group

        # Done waiting? (all released OR timeout)
        if not self._group or self._release_wait >= self._release_max_wait:
            # Drop any remaining stragglers back to wandering
            for civ in self._group:
                if civ.state == FOLLOWING_GUIDE:
                    civ.assigned_guide = None
                    civ.state = WANDERING
            self._group.clear()
            # Go back for more
            self._phase = self.RETURNING
            self._advance_patrol(env)

    def _do_returning(self, env, smoke, civilians):
        """Go back for more civilians."""
        if len(self._group) >= self._max_group:
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
        """Add nearby civilians to group.
        Uses SMALLER radius than broadcast — guide needs to be close enough
        that civilians can physically follow through doorways."""
        if len(self._group) >= self._max_group:
            return
        # Recruit radius: 4 cells (40 units) — much smaller than broadcast (80)
        # Civilians need to be close enough to actually follow
        radius = min(env.cell_size * 4, self.get_broadcast_radius(smoke))

        from evacsim.civilian import (MOVING_TO_EXIT, ESCAPED, DEAD,
                                       IDLE, REACTING, FOLLOWING_GUIDE)

        for civ in civilians:
            if len(self._group) >= self._max_group:
                break
            if civ.state in (MOVING_TO_EXIT, ESCAPED, DEAD, FOLLOWING_GUIDE):
                continue
            if civ.assigned_guide is not None:
                continue

            dist = np.linalg.norm(civ.pos - self.pos)
            if dist < radius:
                # Wake up idle/reacting civilians
                if civ.state == IDLE:
                    civ.state = REACTING
                    civ._reaction_counter = 0
                    civ._reaction_delay = min(civ._reaction_delay, 5)
                if civ.state == REACTING:
                    # Speed up reaction and wait for them
                    civ._reaction_delay = min(civ._reaction_delay, 5)
                    continue  # Don't add to group until they're ready to move

                # Active civilian — add to group
                self._group.append(civ)
                civ.assigned_guide = self
                civ.state = FOLLOWING_GUIDE

    def _count_recruitable(self, env, smoke, civilians):
        """Count recruitable civilians nearby."""
        radius = self.get_broadcast_radius(smoke)
        from evacsim.civilian import (MOVING_TO_EXIT, ESCAPED, DEAD,
                                       IDLE, FOLLOWING_GUIDE)
        count = 0
        for civ in civilians:
            if civ.state in (MOVING_TO_EXIT, ESCAPED, DEAD, FOLLOWING_GUIDE):
                continue
            if civ.assigned_guide is not None:
                continue
            if np.linalg.norm(civ.pos - self.pos) < radius:
                count += 1
        return count

    def _start_escort(self, env):
        """Begin leading group to nearest SAFE exit.
        Avoids exits that are in or near fire."""
        self._phase = self.ESCORTING
        fg = getattr(self, '_fire_grid', None)

        best_exit, best_score = None, float("inf")
        for ex, ey in self.known_exits:
            dist = np.hypot(self.pos[0] - ex, self.pos[1] - ey)
            # Penalize exits near fire or heavy smoke
            danger_penalty = 0.0
            if fg is not None:
                er, ec = env.world_to_grid(ex, ey)
                if 0 <= er < fg.shape[0] and 0 <= ec < fg.shape[1]:
                    danger_penalty += fg[er, ec] * 500.0
            sg = getattr(self, '_smoke_grid', None)
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
            self._waypoints = bfs_path(env, self.pos, np.array(best_exit), fire_grid=fg)

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

    def _advance_patrol(self, env):
        """Move to next patrol point."""
        if not self._patrol_route:
            return
        self._patrol_idx = (self._patrol_idx + 1) % len(self._patrol_route)
        fg = getattr(self, '_fire_grid', None)
        self._waypoints = bfs_path(env, self.pos, self._patrol_route[self._patrol_idx], fire_grid=fg)


def spawn_sweeper_guides(env: Environment, n_guides: int, speed: float = 2.0,
                         seed: int = 77) -> list[SweeperGuide]:
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

    guides = []
    for i in range(n_guides):
        ex, ey = exits[i % len(exits)]
        # Get this guide's territory, fall back to all points if empty
        pts = territories[i] if territories[i] else all_points
        # Order: farthest points first (so guide goes deep into building first)
        pts_sorted = sorted(pts, key=lambda p: -np.linalg.norm(p - np.array([ex, ey])))
        guide = SweeperGuide(ex, ey, speed, env, rng, patrol_points=pts_sorted)
        guides.append(guide)

    return guides


# ════════════════════════════════════════════════════════════════════
# CONGESTION-AWARE ROUTER — corridor junction controller
# ════════════════════════════════════════════════════════════════════

def _find_junctions(env: Environment):
    """Find corridor junction cells — cells where 3+ cardinal directions
    are walkable. These are T-junctions, crossroads, and doorway openings.

    Junctions are the decision points where civilians choose which way
    to go. Router positions here to direct traffic."""
    rows, cols = env.grid.shape
    junctions = []

    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if env.grid[r, c] == CellType.WALL:
                continue
            # Count walkable cardinal neighbours
            open_dirs = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if env.grid[r + dr, c + dc] != CellType.WALL:
                    open_dirs += 1
            # Junction = 3+ open directions (T-junction or crossroad)
            if open_dirs >= 3:
                wx, wy = env.grid_to_world(r, c)
                junctions.append(np.array([wx, wy], dtype=float))

    # De-duplicate: keep junctions at least 3 cells apart
    if not junctions:
        return junctions
    filtered = [junctions[0]]
    min_dist = env.cell_size * 3
    for j in junctions[1:]:
        if all(np.linalg.norm(j - f) >= min_dist for f in filtered):
            filtered.append(j)
    return filtered


def _rank_junctions_by_centrality(junctions, exits, env):
    """Rank junctions by how centrally located they are.
    Centrality = sum of BFS distances to all exits (lower = more central).
    Most central junctions see the most traffic."""
    if not junctions:
        return []

    scored = []
    for j in junctions:
        total_dist = 0
        for ex, ey in exits:
            total_dist += np.hypot(j[0] - ex, j[1] - ey)
        scored.append((total_dist, j))

    scored.sort(key=lambda x: x[0])  # Most central first (lowest total distance)
    return [j for _, j in scored]


class RouterGuide(Guide):
    """
    Congestion-Aware Router: positions at corridor JUNCTIONS and
    directs traffic based on real-time exit congestion.

    Fundamentally different from other policies:
      - Beacon: STATIC at exits
      - Sweeper: MOBILE through rooms, collects groups
      - Router: STRATEGIC at junctions, manages corridor FLOW

    Behaviour:
      1. Find all corridor junctions (3+ open directions)
      2. Walk to most central junction (decision point with most traffic)
      3. Stand there, broadcast per-civilian optimal exit
      4. Every 30 ticks: re-evaluate — if congestion shifts, move to a
         better junction closer to the congested area
      5. Can REDIRECT civilians already heading to wrong exit

    Intelligence:
      - Per-civilian exit scoring: distance + congestion_penalty + smoke_penalty
      - Dynamic repositioning based on real-time flow data
      - Smoke-aware: penalizes exits with heavy smoke on the path
    """

    def __init__(self, x: float, y: float, speed: float,
                 env: Environment, rng: np.random.Generator,
                 assigned_junctions: list = None):
        super().__init__(x, y, speed, env, rng)

        # Get junctions for this guide
        if assigned_junctions:
            self._junctions = assigned_junctions
        else:
            all_j = _find_junctions(env)
            self._junctions = _rank_junctions_by_centrality(
                all_j, env.exits, env)

        # Current junction we're standing at
        self._current_junction_idx = 0
        self._waypoints = []
        self._at_junction = False

        # Congestion data
        self._exit_counts = {}
        self._eval_interval = 80  # Stay at junction long before reconsidering
        self._eval_counter = 0

        # Walk to first (most central) junction
        if self._junctions:
            self._waypoints = bfs_path(env, self.pos, self._junctions[0])

    def update(self, env, smoke, fire, civilians):
        super().update(env, smoke, fire, civilians)
        if not self.alive:
            return
        if self._do_self_evacuation(env):
            return

        # Periodic: update congestion + maybe reposition
        self._eval_counter += 1
        if self._eval_counter >= self._eval_interval:
            self._eval_counter = 0
            self._update_congestion(civilians)
            self._maybe_reposition(env, smoke, civilians)

        # ALWAYS broadcast to nearby civilians — even while walking to junction
        # This ensures we help every civilian we pass, not just at junctions
        if self._eval_counter % 3 == 0:
            self._broadcast_optimal_exits(env, smoke, civilians)

        # Movement — walk to junction, then STAY there
        if not self._at_junction:
            if not self._waypoints:
                self._at_junction = True
                self.vel = np.zeros(2)
            else:
                self._follow_path(env)
        else:
            self.vel = np.zeros(2)  # Stand at junction

    def _update_congestion(self, civilians):
        """Count civilians heading to each exit + queued at each exit."""
        from evacsim.civilian import MOVING_TO_EXIT, QUEUED
        self._exit_counts = {(ex, ey): 0 for ex, ey in self.known_exits}
        for civ in civilians:
            if civ.state not in (MOVING_TO_EXIT, QUEUED) or civ._target_exit is None:
                continue
            tx, ty = civ._target_exit
            best_exit, best_d = None, float("inf")
            for ex, ey in self.known_exits:
                d = np.hypot(tx - ex, ty - ey)
                if d < best_d:
                    best_d = d
                    best_exit = (ex, ey)
            if best_exit and best_d < 20:
                self._exit_counts[best_exit] += 1

    def _maybe_reposition(self, env, smoke, civilians):
        """Check if we should move to a different junction.

        Move if:
          - Current junction is in heavy smoke
          - A different junction is closer to the congestion hotspot
          - A different junction has more un-helped civilians nearby
        """
        if not self._junctions or len(self._junctions) < 2:
            return

        # Score each junction
        best_idx = self._current_junction_idx
        best_score = float("inf")

        # Find the most congested exit
        if self._exit_counts:
            most_congested = max(self._exit_counts, key=self._exit_counts.get)
            max_congestion = self._exit_counts[most_congested]
        else:
            most_congested = None
            max_congestion = 0

        from evacsim.civilian import ESCAPED, DEAD, IDLE, MOVING_TO_EXIT, QUEUED

        for i, junc in enumerate(self._junctions):
            # Penalty for smoke at junction
            smoke_at = smoke.get_density_at(junc[0], junc[1])
            smoke_penalty = smoke_at * 200

            # Bonus for being near congested exit (to intercept traffic)
            congestion_bonus = 0
            if most_congested and max_congestion > 3:
                dist_to_congested = np.hypot(
                    junc[0] - most_congested[0], junc[1] - most_congested[1])
                congestion_bonus = -dist_to_congested * 0.5  # Negative = good

            # Bonus for un-helped civilians nearby
            nearby_unhelpful = 0
            for civ in civilians:
                if civ.state in (ESCAPED, DEAD, IDLE, MOVING_TO_EXIT, QUEUED):
                    continue
                d = np.linalg.norm(civ.pos - junc)
                if d < self.broadcast_radius:
                    nearby_unhelpful += 1
            civilian_bonus = -nearby_unhelpful * 50  # More civilians = better

            score = smoke_penalty + congestion_bonus + civilian_bonus

            if score < best_score:
                best_score = score
                best_idx = i

        # Reposition if a significantly better junction exists
        if best_idx != self._current_junction_idx:
            self._current_junction_idx = best_idx
            self._waypoints = bfs_path(
                env, self.pos, self._junctions[best_idx])
            self._at_junction = False

    def _best_exit_for(self, civ_pos, smoke):
        """Find optimal exit for a civilian. Considers:
        1. BFS distance from civilian to exit
        2. Congestion at exit (how many already heading there)
        3. Smoke on the path (penalize smoky exits)"""
        best_exit, best_score = None, float("inf")

        for ex, ey in self.known_exits:
            # Distance
            dist = np.hypot(civ_pos[0] - ex, civ_pos[1] - ey)

            # Congestion penalty
            congestion = self._exit_counts.get((ex, ey), 0)
            congestion_penalty = congestion * 30.0

            # Smoke penalty: sample smoke at exit location
            smoke_at_exit = smoke.get_density_at(ex, ey)
            smoke_penalty = smoke_at_exit * 100.0

            score = dist + congestion_penalty + smoke_penalty

            if score < best_score:
                best_score = score
                best_exit = (ex, ey)

        return best_exit

    def _broadcast_optimal_exits(self, env, smoke, civilians):
        """Tell each nearby civilian their personal optimal exit."""
        radius = self.get_broadcast_radius(smoke)

        from evacsim.civilian import (MOVING_TO_EXIT, ESCAPED, DEAD,
                                       IDLE, REACTING, FOLLOWING_GUIDE, QUEUED)

        for civ in civilians:
            if civ.state in (ESCAPED, DEAD, FOLLOWING_GUIDE):
                continue
            dist = np.linalg.norm(civ.pos - self.pos)
            if dist >= radius:
                continue

            optimal = self._best_exit_for(civ.pos, smoke)
            if optimal is None:
                continue

            if civ.state == IDLE:
                civ.state = REACTING
                civ._reaction_counter = 0
                civ._reaction_delay = min(civ._reaction_delay, 10)
                civ._guide_told_exit = optimal
            elif civ.state == REACTING:
                civ._reaction_delay = min(civ._reaction_delay, 5)
                civ._guide_told_exit = optimal
            elif civ.state in (MOVING_TO_EXIT, QUEUED):
                # Redirect only if new exit is 20%+ better
                if civ._target_exit is not None:
                    cur = civ._target_exit
                    cur_s = (np.hypot(civ.pos[0] - cur[0], civ.pos[1] - cur[1])
                             + self._exit_counts.get((cur[0], cur[1]), 0) * 30.0)
                    opt_s = (np.hypot(civ.pos[0] - optimal[0], civ.pos[1] - optimal[1])
                             + self._exit_counts.get(optimal, 0) * 30.0)
                    if opt_s < cur_s * 0.8:
                        civ._set_exit_target_force(env, optimal)
            else:
                civ._set_exit_target(env, optimal)


def spawn_router_guides(env, n_guides, speed=2.0, seed=77):
    """Spawn router guides at exits, each assigned to different junctions.

    Junctions are split by territory so guides don't overlap.
    Each guide gets the most central junctions in their territory."""
    rng = np.random.default_rng(seed)
    exits = list(env.exits)
    all_junctions = _find_junctions(env)

    if not all_junctions:
        # Fallback: no junctions found, use patrol points
        all_junctions = _find_patrol_points(env)

    ranked = _rank_junctions_by_centrality(all_junctions, env.exits, env)

    # Split junctions into territories
    center_y = env.height / 2.0
    center_x = env.width / 2.0

    territories = [[] for _ in range(n_guides)]
    if n_guides == 1:
        territories[0] = ranked
    elif n_guides == 2:
        for j in ranked:
            territories[0 if j[1] < center_y else 1].append(j)
    else:
        for j in ranked:
            idx = (0 if j[0] < center_x else 1) + (0 if j[1] < center_y else 2)
            territories[idx % n_guides].append(j)

    guides = []
    for i in range(n_guides):
        ex, ey = exits[i % len(exits)]
        juncs = territories[i] if territories[i] else ranked
        # Re-rank by centrality within territory
        juncs = _rank_junctions_by_centrality(juncs, env.exits, env)
        guide = RouterGuide(ex, ey, speed, env, rng, assigned_junctions=juncs)
        guides.append(guide)
    return guides
