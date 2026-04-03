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


def dijkstra_congestion_path(env, start_world, goal_world,
                             fire_grid=None, smoke_grid=None,
                             civilian_density=None):
    """Congestion-aware pathfinding using Dijkstra's algorithm.
    Inspired by Zhang et al. (2020) IEEE ICRA.

    Cost of traversing a cell = base_cost + fire_penalty + smoke_penalty + density_penalty

    This finds paths that AVOID congested areas, smoky corridors, and fire —
    not just the shortest geometric path. A longer path through empty corridors
    is preferred over a shorter path through a jammed doorway.

    civilian_density: 2D numpy array counting civilians per cell.
    """
    import heapq

    sr, sc = env.world_to_grid(start_world[0], start_world[1])
    gr, gc = env.world_to_grid(goal_world[0], goal_world[1])
    rows, cols = env.grid.shape

    if sr == gr and sc == gc:
        return []

    # Cost grid: infinity everywhere
    cost = np.full((rows, cols), float('inf'))
    cost[sr, sc] = 0.0
    parent = {}
    parent[(sr, sc)] = None

    # Priority queue: (cost, row, col)
    pq = [(0.0, sr, sc)]

    while pq:
        c, r, cc = heapq.heappop(pq)

        if r == gr and cc == gc:
            # Reconstruct path
            path = []
            current = (r, cc)
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()
            waypoints = []
            for pr, pc in path[1:]:
                wx, wy = env.grid_to_world(pr, pc)
                waypoints.append(np.array([wx, wy], dtype=float))
            return waypoints

        if c > cost[r, cc]:
            continue  # Stale entry

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, cc + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if env.grid[nr, nc] == CellType.WALL:
                continue

            # Base movement cost
            move_cost = 1.0

            # Fire penalty (impassable if > 0.5, expensive if > 0.2)
            if fire_grid is not None:
                fire_val = fire_grid[nr, nc]
                if fire_val > 0.5:
                    continue  # Impassable
                move_cost += fire_val * 20.0

            # Smoke penalty
            if smoke_grid is not None:
                smoke_val = smoke_grid[nr, nc]
                move_cost += smoke_val * 5.0

            # Congestion penalty (Zhang et al. 2020 density map)
            if civilian_density is not None:
                density = civilian_density[nr, nc]
                move_cost += density * 8.0  # Each civilian on cell adds 8 cost

            new_cost = cost[r, cc] + move_cost
            if new_cost < cost[nr, nc]:
                cost[nr, nc] = new_cost
                parent[(nr, nc)] = (r, cc)
                heapq.heappush(pq, (new_cost, nr, nc))

    return []  # No path found


def build_density_grid(env, civilians):
    """Build a 2D grid of civilian density per cell.
    Used by Router for congestion-aware pathfinding (Zhang et al. 2020)."""
    from evacsim.civilian import ESCAPED, DEAD
    grid = np.zeros(env.grid.shape, dtype=float)
    for civ in civilians:
        if civ.state in (ESCAPED, DEAD):
            continue
        r, c = env.world_to_grid(civ.pos[0], civ.pos[1])
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
            grid[r, c] += 1.0
    return grid


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

        # Beacon broadcast radius (sound-based: megaphone/whistle)
        # 100 units = 10 cells = ~30m at 3m/cell scale
        self.broadcast_radius = 100.0  # 10 cells

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

    def get_visual_recruit_radius(self, smoke: SmokeSystem) -> float:
        """Visual recruit range based on Jin (1978) visibility model.
        Guide is a light-emitting object (hi-vis gear, flashlight).
        Guide wears breathing apparatus — smoke reduces vision by 60%
        of what it does to unprotected civilians (mask prevents eye
        irritation, Jin 1978 showed irritation is the main vision reducer).
        Used by Sweeper and Router (physical contact, not broadcast)."""
        local_smoke = smoke.get_density_at(self.pos[0], self.pos[1])
        base_range = 70.0  # 7 cells in clear air (~20m at 3m/cell)
        # Mask reduces smoke's effect by 40% (only particle obscuration remains)
        smoke_effect = local_smoke * 0.6
        return base_range * max(0.15, 1.0 - smoke_effect)

    def _has_line_of_sight(self, target_pos, env) -> bool:
        """Ray-cast: can we see the target without a wall blocking?
        Used for visual recruit — Sweeper/Router can't recruit through walls."""
        diff = target_pos - self.pos
        dist = np.linalg.norm(diff)
        if dist < env.cell_size:
            return True
        direction = diff / dist
        steps = int(dist / env.cell_size) + 1
        for i in range(1, min(steps + 1, 25)):
            cx = self.pos[0] + direction[0] * env.cell_size * i
            cy = self.pos[1] + direction[1] * env.cell_size * i
            if not env.is_walkable(cx, cy):
                return False
        return True

    def _follow_path(self, env: Environment) -> None:
        """Follow the pre-computed waypoint path.
        Checks ahead for fire AND heavy smoke — reroutes if dangerous.
        Guides resist smoke but their civilian group does not."""
        if not self._waypoints:
            self.vel = np.array([0.0, 0.0])
            return

        # Check next 10 waypoints for fire or smoke
        fg = getattr(self, '_fire_grid', None)
        sg = getattr(self, '_smoke_grid', None)
        # Lower smoke threshold when carrying civilians (they're vulnerable)
        has_group = hasattr(self, '_group') and len(self._group) > 0
        smoke_threshold = 0.3 if has_group else 0.6

        if fg is not None or sg is not None:
            needs_reroute = False
            for wp in self._waypoints[:10]:
                wr, wc = env.world_to_grid(wp[0], wp[1])
                if fg is not None and 0 <= wr < fg.shape[0] and 0 <= wc < fg.shape[1]:
                    if fg[wr, wc] > 0.2:
                        needs_reroute = True
                        break
                if sg is not None and 0 <= wr < sg.shape[0] and 0 <= wc < sg.shape[1]:
                    if sg[wr, wc] > smoke_threshold:
                        needs_reroute = True
                        break
            if needs_reroute:
                # Combine fire + smoke into danger grid for BFS avoidance
                # When carrying a group: smoke is MORE dangerous (civilians vulnerable)
                danger = fg if fg is not None else np.zeros_like(env.grid, dtype=float)
                if sg is not None:
                    smoke_weight = 1.0 if has_group else 0.5
                    danger = np.maximum(danger, sg * smoke_weight)
                final = self._waypoints[-1]
                self._waypoints = bfs_path(env, self.pos, final, fire_grid=danger)
                if not self._waypoints:
                    # No safe path — try without smoke avoidance as fallback
                    self._waypoints = bfs_path(env, self.pos, final, fire_grid=fg)
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
            # Slow down when carrying a group (pace with civilians)
            if has_group:
                move_speed = self.speed * 0.75  # ~1.5 — close to civilian speed
            else:
                move_speed = self.speed
            self.vel = direction * move_speed

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

        # --- Phase 2: Probabilistic broadcast (sound-based) ---
        # Hearing probability decreases with distance and smoke/noise.
        # Sound penetrates walls (no LOS check) but attenuates.
        # Based on acoustic propagation in fire conditions.
        max_range = self.broadcast_radius  # 100 units (10 cells / ~30m)

        from evacsim.civilian import (MOVING_TO_EXIT, ESCAPED, DEAD,
                                       IDLE, REACTING, FOLLOWING_GUIDE)

        exit_pos = (self.assigned_exit[0], self.assigned_exit[1])

        for civ in civilians:
            if civ.state in (MOVING_TO_EXIT, ESCAPED, DEAD, FOLLOWING_GUIDE):
                continue

            dist = np.linalg.norm(civ.pos - self.pos)
            if dist >= max_range:
                continue

            # Distance factor: linear falloff (closer = easier to hear)
            distance_factor = 1.0 - (dist / max_range)

            # Smoke factor: fire noise + smoke reduces hearing
            # Smoke at CIVILIAN's position (they're the listener)
            civ_smoke = smoke.get_density_at(civ.pos[0], civ.pos[1])
            smoke_factor = max(0.3, 1.0 - civ_smoke * 0.5)

            # Final hearing probability this tick
            hear_chance = distance_factor * smoke_factor

            # Roll the dice
            if self._rng.random() > hear_chance:
                continue  # Didn't hear this tick — try again next tick

            # Civilian heard the guide!
            if civ.state == IDLE:
                civ.state = REACTING
                civ._reaction_counter = 0
                civ._reaction_delay = min(civ._reaction_delay, 10)
                civ._guide_told_exit = exit_pos
            elif civ.state == REACTING:
                civ._reaction_delay = min(civ._reaction_delay, 5)
                civ._guide_told_exit = exit_pos
            else:
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

        # DANGER CHECK: proactive detection — check guide surroundings + group members
        if self._group:
            danger = False
            sg = getattr(self, '_smoke_grid', None)
            fg = getattr(self, '_fire_grid', None)
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
                if self._group:
                    if recruitable == 0:
                        self._start_escort(env)
                        return
                self._advance_patrol(env)
            return

        if not self._waypoints:
            self._waiting = True
            self._wait_counter = 0
            self.vel = np.zeros(2)
        else:
            self._follow_path(env)

    def _do_escorting(self, env, smoke, civilians):
        """Lead group to exit. Pick up anyone on the way (no limit during escort)."""
        if not self._group:
            self._phase = self.RETURNING
            self._advance_patrol(env)
            return
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
        """Stand at exit — civilians self-release when they spot the exit sign.
        Guide just waits here as a reference point. After timeout, drop stragglers."""
        from evacsim.civilian import FOLLOWING_GUIDE, WANDERING

        self.vel = np.zeros(2)
        self._release_wait += 1

        # Civilians spot the exit themselves (Fix 4 in civilian.py)
        # and leave the group automatically. We just wait.
        # _clean_group() in update() removes escaped/dead civilians.

        # Done waiting? (all self-released OR timeout)
        if not self._group or self._release_wait >= self._release_max_wait:
            for civ in self._group:
                if civ.state == FOLLOWING_GUIDE:
                    civ.assigned_guide = None
                    civ.state = WANDERING
            self._group.clear()
            self._phase = self.RETURNING
            self._advance_patrol(env)

    def _do_returning(self, env, smoke, civilians):
        """Go back for more civilians."""
        if len(self._group) >= self._max_group:
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
                sg = getattr(self, '_smoke_grid', None)
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
        if len(self._group) >= self._max_group:
            return
        radius = self.get_visual_recruit_radius(smoke)

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
        from evacsim.civilian import (MOVING_TO_EXIT, ESCAPED, DEAD,
                                       IDLE, REACTING, FOLLOWING_GUIDE)
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
        from evacsim.civilian import (MOVING_TO_EXIT, ESCAPED, DEAD,
                                       IDLE, FOLLOWING_GUIDE)
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
            # Escorting group — smoke is full danger (civilians vulnerable)
            # Must match _follow_path's reroute weight (sg * 1.0 when has_group)
            sg = getattr(self, '_smoke_grid', None)
            if fg is not None and sg is not None:
                danger = np.maximum(fg, sg * 1.0)
            elif fg is not None:
                danger = fg
            elif sg is not None:
                danger = sg * 1.0
            else:
                danger = None
            self._waypoints = bfs_path(env, self.pos, np.array(best_exit), fire_grid=danger)

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
        """Move to next patrol point. Skip cleared and dangerous points."""
        if not self._patrol_route:
            return

        fg = getattr(self, '_fire_grid', None)
        sg = getattr(self, '_smoke_grid', None)

        # Try up to len(route) points to find a valid one
        for _ in range(len(self._patrol_route)):
            self._patrol_idx = (self._patrol_idx + 1) % len(self._patrol_route)

            # Full cycle completed? Reset cleared points (re-check everything)
            if self._patrol_idx == self._cycle_start_idx:
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

        # Compute path — avoid smoke if carrying group
        if self._group:
            if fg is not None and sg is not None:
                danger = np.maximum(fg, sg * 0.5)
            elif fg is not None:
                danger = fg
            elif sg is not None:
                danger = sg * 0.5
            else:
                danger = None
            self._waypoints = bfs_path(env, self.pos, self._patrol_route[self._patrol_idx], fire_grid=danger)
        else:
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
    Congestion-Aware Router (inspired by Zhang et al. 2020, IEEE ICRA).

    Uses congestion-aware Dijkstra pathfinding where path cost includes
    civilian density, smoke, and fire. Routes civilians on the LEAST
    CONGESTED PATH to the LEAST CONGESTED EXIT.

    Fundamentally different from other policies:
      - Beacon: STATIC at exits, broadcasts one exit
      - Sweeper: PATROLS rooms, collects groups, escorts to NEAREST exit
      - Router: REACTS to congestion, collects stuck civilians,
                escorts to OPTIMAL exit via LEAST CONGESTED path

    Behaviour:
      1. Go to territory center, scan for congestion hotspots
      2. Hotspot found → go there, collect stuck civilians into group
      3. Escort group to least congested exit using Dijkstra density path
      4. Release at exit → return to territory center → scan again

    Intelligence (Zhang et al. 2020):
      - Population density map for congestion detection
      - Dijkstra pathfinding with density-weighted costs
      - Per-civilian optimal exit selection
      - Can REDIRECT civilians already heading to congested exit
    """

    SCANNING = "scanning"
    RESPONDING = "responding"
    COLLECTING = "collecting"
    ESCORTING = "escorting"
    RELEASING = "releasing"
    RETURNING = "returning"

    def __init__(self, x, y, speed, env, rng, territory_bounds=None):
        super().__init__(x, y, speed, env, rng)

        # Territory
        if territory_bounds:
            self._territory = territory_bounds
        else:
            self._territory = (0, env.grid.shape[0], 0, env.grid.shape[1])

        self._home_pos = self._find_territory_center(env)
        self._waypoints = []
        self._phase = self.SCANNING

        # Group management (like Sweeper but for congestion)
        self._group = []
        self._max_group = 6

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

        # Walk to territory center
        self._waypoints = bfs_path(env, self.pos, self._home_pos)

    def _find_territory_center(self, env):
        min_r, max_r, min_c, max_c = self._territory
        cr = max(0, min(env.grid.shape[0]-1, (min_r + max_r) // 2))
        cc = max(0, min(env.grid.shape[1]-1, (min_c + max_c) // 2))
        if env.grid[cr, cc] != CellType.WALL:
            return np.array(env.grid_to_world(cr, cc), dtype=float)
        visited = {(cr, cc)}
        q = deque([(cr, cc)])
        rows, cols = env.grid.shape
        while q:
            r, c = q.popleft()
            if env.grid[r, c] != CellType.WALL:
                return np.array(env.grid_to_world(r, c), dtype=float)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<rows and 0<=nc<cols and (nr,nc) not in visited:
                    visited.add((nr,nc))
                    q.append((nr,nc))
        return np.array(env.grid_to_world(cr, cc), dtype=float)

    def update(self, env, smoke, fire, civilians):
        super().update(env, smoke, fire, civilians)
        if not self.alive:
            self._release_group_wandering()
            return
        if self._do_self_evacuation(env):
            self._release_group_wandering()
            return

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
            self._do_releasing(env)
        elif self._phase == self.RETURNING:
            self._do_returning(env, smoke, civilians)

    # ── Phase behaviors ──

    def _do_scanning(self, env, smoke, civilians):
        if self._waypoints:
            self._follow_path(env)
            # Recruit while walking to home
            self._recruit_nearby(env, smoke, civilians)
            return
        self.vel = np.zeros(2)
        # Scan every eval_interval ticks (not just on counter==0)
        hotspot = self._find_hotspot(env, smoke, civilians)
        if hotspot is not None:
            fg = getattr(self, '_fire_grid', None)
            self._waypoints = bfs_path(env, self.pos, hotspot, fire_grid=fg)
            self._phase = self.RESPONDING
        elif self._group:
            # Picked up civilians while walking — escort them
            self._start_escort(env, smoke, civilians)

    def _do_responding(self, env, smoke=None, civilians=None):
        if not self._waypoints:
            self._phase = self.COLLECTING
            self._collect_wait = 0
            self.vel = np.zeros(2)
        else:
            self._follow_path(env)
            # Recruit while walking to hotspot
            if smoke and civilians:
                self._recruit_nearby(env, smoke, civilians)

    def _do_collecting(self, env, smoke, civilians):
        """At hotspot: recruit civilians into group. Soft limit — collect
        as many as possible, but start escorting once we have enough or timeout."""
        self.vel = np.zeros(2)
        self._collect_wait += 1
        self._recruit_nearby(env, smoke, civilians)

        # Start escorting when: enough people OR timeout OR no more nearby
        has_enough = len(self._group) >= self._max_group
        timed_out = self._collect_wait >= self._max_collect_wait
        if has_enough or timed_out:
            if self._group:
                self._start_escort(env, smoke, civilians)
            else:
                self._phase = self.RETURNING
                fg = getattr(self, '_fire_grid', None)
                self._waypoints = bfs_path(env, self.pos, self._home_pos, fire_grid=fg)

    def _do_escorting(self, env, smoke, civilians):
        """Lead group to optimal exit. KEY: always pick up civilians on the
        way — no group limit while escorting. Maximize survivors per trip."""
        if not self._group:
            # Everyone died — abort escort, go back to scanning
            self._phase = self.RETURNING
            fg = getattr(self, '_fire_grid', None)
            self._waypoints = bfs_path(self.env, self.pos, self._home_pos, fire_grid=fg)
            return
        self._recruit_nearby_unlimited(env, smoke, civilians)
        if not self._waypoints:
            if self._escort_exit is not None:
                dist = np.linalg.norm(self.pos - np.array(self._escort_exit))
                if dist < env.cell_size * 2:
                    self._phase = self.RELEASING
                    self._release_wait = 0
                    self.vel = np.zeros(2)
                    return
            self.vel = np.zeros(2)
        else:
            self._follow_path(env)

    def _do_releasing(self, env):
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
            if dist_exit < self.env.cell_size * 5 or dist_guide < self.env.cell_size * 3:
                civ._set_exit_target_force(self.env, self._escort_exit)
            else:
                still.append(civ)
        self._group = still

        if not self._group or self._release_wait >= self._release_max_wait:
            for civ in self._group:
                if civ.state == FOLLOWING_GUIDE:
                    civ.assigned_guide = None
                    civ.state = WANDERING
            self._group.clear()
            self._phase = self.RETURNING
            fg = getattr(self, '_fire_grid', None)
            self._waypoints = bfs_path(self.env, self.pos, self._home_pos, fire_grid=fg)

    def _do_returning(self, env, smoke=None, civilians=None):
        if not self._waypoints:
            self._phase = self.SCANNING
            self.vel = np.zeros(2)
        else:
            self._follow_path(env)
            # Recruit while walking home
            if smoke and civilians:
                self._recruit_nearby(env, smoke, civilians)
                if self._group:
                    # Found civilians on the way — escort them
                    self._start_escort(env, smoke, civilians)

    # ── Intelligence methods ──

    def _find_hotspot(self, env, smoke, civilians):
        """Find the area with MOST un-helped civilians in territory.
        Router's intelligence: go where you can help the MOST people.
        Sweeper follows a fixed route. Router goes where the need is greatest."""
        from evacsim.civilian import ESCAPED, DEAD, IDLE, REACTING
        rows, cols = env.grid.shape
        min_r, max_r, min_c, max_c = self._territory
        fg = getattr(self, '_fire_grid', None)
        sg = getattr(self, '_smoke_grid', None)
        region = 5
        best_pos, best_n = None, 1  # Need 2+ civilians

        for rr in range(min_r, min(max_r, rows), region):
            for rc in range(min_c, min(max_c, cols), region):
                cr = min(rr + region//2, rows-1)
                cc = min(rc + region//2, cols-1)
                if fg is not None and fg[cr, cc] > 0.2:
                    continue
                if sg is not None and sg[cr, cc] > 0.5:
                    continue
                n, sx, sy = 0, 0.0, 0.0
                for civ in civilians:
                    if civ.state in (ESCAPED, DEAD, IDLE, REACTING):
                        continue
                    if civ.assigned_guide is not None:
                        continue
                    r, c = env.world_to_grid(civ.pos[0], civ.pos[1])
                    if rr <= r < rr+region and rc <= c < rc+region:
                        n += 1; sx += civ.pos[0]; sy += civ.pos[1]
                if n > best_n:
                    best_n = n
                    best_pos = np.array([sx/n, sy/n])

        if best_pos is not None:
            gr, gc = env.world_to_grid(best_pos[0], best_pos[1])
            if env.grid[gr, gc] == CellType.WALL:
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        nr, nc = gr+dr, gc+dc
                        if 0<=nr<rows and 0<=nc<cols and env.grid[nr,nc]!=CellType.WALL:
                            return np.array(env.grid_to_world(nr, nc), dtype=float)
                return None
        return best_pos

    def _recruit_nearby(self, env, smoke, civilians):
        """Recruit nearby civilians using visual range (Jin, 1978) + LOS."""
        if len(self._group) >= self._max_group:
            return
        radius = self.get_visual_recruit_radius(smoke)
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
                    from evacsim.civilian import FOLLOWING_GUIDE
                    civ.state = FOLLOWING_GUIDE

    def _recruit_nearby_unlimited(self, env, smoke, civilians):
        """Recruit ANY nearby civilian while escorting — no group limit.
        Uses Jin (1978) visual range + LOS. Picks up everyone on the way."""
        radius = self.get_visual_recruit_radius(smoke)
        from evacsim.civilian import (MOVING_TO_EXIT, ESCAPED, DEAD,
                                       IDLE, REACTING, FOLLOWING_GUIDE)
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

    def _start_escort(self, env, smoke, civilians):
        """Find LEAST CONGESTED exit using density-aware scoring, then
        use congestion-aware Dijkstra to plan the path."""
        self._phase = self.ESCORTING
        fg = getattr(self, '_fire_grid', None)
        sg = getattr(self, '_smoke_grid', None)

        # Score each exit: distance + congestion + smoke
        best_exit, best_score = None, float("inf")
        for ex, ey in self.known_exits:
            dist = np.hypot(self.pos[0] - ex, self.pos[1] - ey)
            congestion = self._exit_counts.get((ex, ey), 0) * 30.0
            smoke_pen = smoke.get_density_at(ex, ey) * 200.0
            fire_pen = 0
            if fg is not None:
                er, ec = env.world_to_grid(ex, ey)
                if 0<=er<fg.shape[0] and 0<=ec<fg.shape[1]:
                    fire_pen = fg[er, ec] * 500.0
            score = dist + congestion + smoke_pen + fire_pen
            if score < best_score:
                best_score = score
                best_exit = (ex, ey)

        self._escort_exit = best_exit
        if best_exit:
            # Use congestion-aware Dijkstra (Zhang et al. 2020)
            self._waypoints = dijkstra_congestion_path(
                env, self.pos, np.array(best_exit),
                fire_grid=fg, smoke_grid=sg,
                civilian_density=self._density_grid)

    def _update_exit_counts(self, civilians):
        from evacsim.civilian import MOVING_TO_EXIT, QUEUED
        self._exit_counts = {(ex, ey): 0 for ex, ey in self.known_exits}
        for civ in civilians:
            if civ.state not in (MOVING_TO_EXIT, QUEUED) or civ._target_exit is None:
                continue
            tx, ty = civ._target_exit
            best, best_d = None, float("inf")
            for ex, ey in self.known_exits:
                d = np.hypot(tx-ex, ty-ey)
                if d < best_d:
                    best_d = d; best = (ex, ey)
            if best and best_d < 20:
                self._exit_counts[best] += 1

    def _broadcast_redirect(self, env, smoke, civilians):
        """Broadcast optimal exit. Redirect civilians heading to congested exit."""
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
                if civ._target_exit is not None:
                    cur = civ._target_exit
                    cur_s = np.hypot(civ.pos[0]-cur[0], civ.pos[1]-cur[1]) + self._exit_counts.get((cur[0],cur[1]),0)*30
                    opt_s = np.hypot(civ.pos[0]-optimal[0], civ.pos[1]-optimal[1]) + self._exit_counts.get(optimal,0)*30
                    if opt_s < cur_s * 0.75:
                        civ._set_exit_target_force(env, optimal)
            else:
                civ._set_exit_target(env, optimal)

    def _best_exit_for(self, civ_pos, smoke):
        best, best_s = None, float("inf")
        for ex, ey in self.known_exits:
            s = np.hypot(civ_pos[0]-ex, civ_pos[1]-ey) + self._exit_counts.get((ex,ey),0)*30 + smoke.get_density_at(ex,ey)*100
            if s < best_s:
                best_s = s; best = (ex, ey)
        return best

    def _release_group_wandering(self):
        from evacsim.civilian import WANDERING, FOLLOWING_GUIDE
        for civ in self._group:
            if civ.state == FOLLOWING_GUIDE:
                civ.assigned_guide = None
                civ.state = WANDERING
        self._group.clear()

    def _clean_group(self):
        from evacsim.civilian import ESCAPED, DEAD
        self._group = [c for c in self._group if c.state not in (ESCAPED, DEAD)]


def spawn_router_guides(env, n_guides, speed=2.0, seed=77):
    """Spawn router guides at exits, each assigned to a territory."""
    rng = np.random.default_rng(seed)
    exits = list(env.exits)
    rows, cols = env.grid.shape
    mid_r, mid_c = rows // 2, cols // 2

    if n_guides == 1:
        territories = [(0, rows, 0, cols)]
    elif n_guides == 2:
        territories = [(0, mid_r, 0, cols), (mid_r, rows, 0, cols)]
    elif n_guides <= 4:
        territories = [
            (0, mid_r, 0, mid_c), (0, mid_r, mid_c, cols),
            (mid_r, rows, 0, mid_c), (mid_r, rows, mid_c, cols)]
    else:
        base = [(0,mid_r,0,mid_c),(0,mid_r,mid_c,cols),
                (mid_r,rows,0,mid_c),(mid_r,rows,mid_c,cols)]
        territories = [base[i%4] for i in range(n_guides)]

    guides = []
    for i in range(n_guides):
        ex, ey = exits[i % len(exits)]
        t = territories[i] if i < len(territories) else territories[0]
        guides.append(RouterGuide(ex, ey, speed, env, rng, territory_bounds=t))
    return guides
