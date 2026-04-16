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

import math
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


def dijkstra_congestion_path(
    env, start_world, goal_world, fire_grid=None, smoke_grid=None, civilian_density=None
):
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
    cost = np.full((rows, cols), float("inf"))
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


def predict_spread(density, spreadable_mask, source_cells, spread_rate, steps):
    """Simulate cellular automaton fire/smoke spread forward N steps.

    Used by Router for deliberative planning: predicts where fire/smoke
    WILL be when the escort group arrives, not just where it IS now.
    This enables proactive route planning that avoids corridors predicted
    to become dangerous during transit.

    Uses mean noise (1.0) instead of stochastic noise for deterministic
    prediction. This provides the expected-value approximation of the
    CA's stochastic evolution.

    Parameters:
        density: current fire/smoke density grid (2D numpy array)
        spreadable_mask: boolean grid (True = can spread here)
        source_cells: list of (row, col) fire source positions
        spread_rate: CA spread rate (0.005 for fire, 0.02 for smoke)
        steps: number of ticks to predict forward

    Returns:
        Predicted density grid (does NOT modify the original).
    """
    d = density.copy()
    for _ in range(steps):
        padded = np.pad(d, 1, mode="constant", constant_values=0.0)
        max_neighbour = np.maximum(
            np.maximum(padded[0:-2, 1:-1], padded[2:, 1:-1]),
            np.maximum(padded[1:-1, 0:-2], padded[1:-1, 2:]),
        )
        # Deterministic: no random noise (use expected value 1.0)
        delta = spread_rate * np.maximum(0.0, max_neighbour - d)
        d = np.where(spreadable_mask, np.minimum(1.0, d + delta), 0.0)
        for r, c in source_cells:
            d[r, c] = 1.0
    return d


class RouterBlackboard:
    """Shared coordination blackboard for Router guides.

    Multi-agent coordination mechanism (blackboard architecture):
    Routers write claimed hotspot positions when committing to a rescue.
    Other Routers read the blackboard and skip claimed hotspots to avoid
    overlap. Claims auto-expire after EXPIRY ticks to handle guide death
    or rerouting.

    This enables global scanning without territory constraints — Routers
    dynamically distribute across hotspots rather than being confined to
    fixed quadrants.
    """

    EXPIRY = 60  # ticks before a claim expires (3 × eval_interval)

    def __init__(self):
        self._claims = {}  # guide_id → (position, tick_claimed)

    def claim(self, guide_id, position, tick):
        """Claim a hotspot position."""
        self._claims[guide_id] = (np.array(position), tick)

    def release(self, guide_id):
        """Release a claim (guide finished escorting or died)."""
        self._claims.pop(guide_id, None)

    def is_claimed(self, position, current_tick, exclude_guide=None,
                   radius=30.0):
        """Check if a position is already claimed by another guide.
        Uses radius matching — positions within `radius` units count as same hotspot."""
        for gid, (pos, tick) in list(self._claims.items()):
            if gid == exclude_guide:
                continue
            if current_tick - tick > self.EXPIRY:
                del self._claims[gid]  # expired
                continue
            if np.linalg.norm(pos - np.array(position)) < radius:
                return True
        return False

    def reset(self):
        """Clear all claims (new simulation)."""
        self._claims.clear()


def dbscan_cluster(points, epsilon, min_points):
    """DBSCAN spatial clustering (Ester et al., 1996).

    Finds clusters of arbitrary shape without needing to know the number
    of clusters in advance. Used by Router to identify civilian groups
    that need assistance.

    Parameters:
        points: list of np.array positions (world coordinates)
        epsilon: neighbourhood radius — points within this distance are neighbours.
                 Set to guide visual range (justified by Jin 1978).
        min_points: minimum neighbours to form a core point.
                    Clusters smaller than this are noise.

    Returns:
        List of clusters, each cluster = list of indices into `points`.
        Empty list if no clusters found. Noise points are excluded.
    """
    n = len(points)
    if n == 0:
        return []

    UNVISITED = -1
    NOISE = -2
    labels = [UNVISITED] * n
    cluster_id = 0

    # Precompute squared distances (avoid sqrt for performance)
    eps_sq = epsilon * epsilon

    def range_query(idx):
        """Find all points within epsilon of point idx."""
        px, py = points[idx][0], points[idx][1]
        neighbors = []
        for j in range(n):
            dx = px - points[j][0]
            dy = py - points[j][1]
            if dx * dx + dy * dy <= eps_sq:
                neighbors.append(j)
        return neighbors

    for i in range(n):
        if labels[i] != UNVISITED:
            continue

        neighbors = range_query(i)

        if len(neighbors) < min_points:
            labels[i] = NOISE
            continue

        # Start new cluster
        labels[i] = cluster_id
        seed_set = set(neighbors) - {i}
        processed = {i}

        while seed_set:
            q = seed_set.pop()
            if labels[q] == NOISE:
                labels[q] = cluster_id  # Border point joins cluster
            if labels[q] != UNVISITED:
                processed.add(q)
                continue
            labels[q] = cluster_id
            processed.add(q)

            q_neighbors = range_query(q)
            if len(q_neighbors) >= min_points:
                # q is a core point — expand cluster with its neighbours
                seed_set.update(set(q_neighbors) - processed)

        cluster_id += 1

    # Group point indices by cluster ID
    clusters = {}
    for i, label in enumerate(labels):
        if label >= 0:
            clusters.setdefault(label, []).append(i)

    return list(clusters.values())


def build_density_grid(env, civilians):
    """Build a 2D grid of civilian density per cell with 3x3 blur.
    Used by Router for congestion-aware Dijkstra pathfinding (Zhang et al. 2020).

    Each civilian adds density to a 3x3 neighbourhood: 1.0 at their cell,
    0.5 at adjacent cells. This makes corridors NEAR crowds more expensive,
    not just the exact cells civilians stand on. Without blurring, the
    density is too sparse (1 civilian per cell) for Dijkstra to find
    meaningfully different paths from BFS."""
    from evacsim.civilian import ESCAPED, DEAD

    rows, cols = env.grid.shape
    grid = np.zeros((rows, cols), dtype=float)
    for civ in civilians:
        if civ.state in (ESCAPED, DEAD):
            continue
        r, c = env.world_to_grid(civ.pos[0], civ.pos[1])
        if 0 <= r < rows and 0 <= c < cols:
            grid[r, c] += 1.0
            # Spread density to 3x3 neighbourhood (adjacent cells get 0.5)
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        grid[nr, nc] += 0.5
    return grid


class Guide:
    """Base class for all guide types."""

    def __init__(
        self,
        x: float,
        y: float,
        speed: float,
        env: Environment,
        rng: np.random.Generator,
    ):
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
        self.escaped = False  # True if guide successfully reached an exit

        # Which exit this guide is assigned to
        self.assigned_exit = None

    @property
    def health(self) -> float:
        """Current health as 0.0-1.0 ratio. 1.0 = full health, 0.0 = dead."""
        fire_ratio = self._fire_damage / self._max_fire_damage
        smoke_ratio = self._smoke_damage / self._max_smoke_damage
        # Worst of the two damage types determines health
        return max(0.0, 1.0 - max(fire_ratio, smoke_ratio))

    def update(
        self, env: Environment, smoke: SmokeSystem, fire: FireSystem, civilians: list
    ) -> None:
        """Base update — handles damage and self-evacuation.
        Override in subclasses but call super().update() first."""
        if not self.alive:
            return

        # Store grids for danger-aware BFS
        self._fire_grid = fire.density
        self._smoke_grid = smoke.density

        # Store grid position for dynamic max_group danger sampling
        self._last_grid_pos = env.world_to_grid(self.pos[0], self.pos[1])

        # Check fire damage (guides resist smoke but not fire)
        local_fire = fire.get_density_at(self.pos[0], self.pos[1])
        local_smoke = smoke.get_density_at(self.pos[0], self.pos[1])

        if local_fire > 0.3:
            self._fire_damage += 2.0
        if local_smoke > 0.3:
            self._smoke_damage += local_smoke * 0.5  # Half damage vs civilians

        # Check for death
        if (
            self._fire_damage >= self._max_fire_damage
            or self._smoke_damage >= self._max_smoke_damage
        ):
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
            fg = getattr(self, "_fire_grid", None)
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
                self.escaped = True  # Distinguish escaped from killed
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
        fg = getattr(self, "_fire_grid", None)
        sg = getattr(self, "_smoke_grid", None)
        # Lower smoke threshold when carrying civilians (they're vulnerable)
        has_group = hasattr(self, "_group") and len(self._group) > 0
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

        if assigned_exit is not None:
            self.assigned_exit = np.array(assigned_exit, dtype=float)
        else:
            self.assigned_exit = self._find_nearest_exit()

        self._beacon_pos = self._find_beacon_position()
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
        """Stand 2 cells in front of the exit (into the building).
        'In front' = the direction where FLOOR cells are (not walls)."""
        ex, ey = self.assigned_exit
        gr, gc = self.env.world_to_grid(ex, ey)
        rows, cols = self.env.grid.shape

        # Find which direction is "inward" — check neighbors for FLOOR
        inward_dr, inward_dc = 0, 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = gr + dr, gc + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if self.env.grid[nr, nc] == CellType.FLOOR:
                    inward_dr, inward_dc = dr, dc
                    break

        # Step 2 cells inward
        target_r = gr + inward_dr * 2
        target_c = gc + inward_dc * 2
        if (0 <= target_r < rows and 0 <= target_c < cols
                and self.env.grid[target_r, target_c] != CellType.WALL):
            return np.array(self.env.grid_to_world(target_r, target_c), dtype=float)

        # Try 1 cell inward
        target_r = gr + inward_dr
        target_c = gc + inward_dc
        if (0 <= target_r < rows and 0 <= target_c < cols
                and self.env.grid[target_r, target_c] != CellType.WALL):
            return np.array(self.env.grid_to_world(target_r, target_c), dtype=float)

        # Fallback: stand at exit
        return self.assigned_exit.copy()

    def update(
        self, env: Environment, smoke: SmokeSystem, fire: FireSystem, civilians: list
    ) -> None:
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

        from evacsim.civilian import (
            MOVING_TO_EXIT,
            ESCAPED,
            DEAD,
            IDLE,
            REACTING,
            FOLLOWING_GUIDE,
        )

        exit_pos = (self.assigned_exit[0], self.assigned_exit[1])

        for civ in civilians:
            if civ.state in (MOVING_TO_EXIT, ESCAPED, DEAD, FOLLOWING_GUIDE):
                continue

            dist = np.linalg.norm(civ.pos - self.pos)
            if dist >= max_range:
                continue

            # Distance factor: linear falloff (closer = easier to hear)
            distance_factor = 1.0 - (dist / max_range)

            # Listener-side: smoke at civilian's position (irritation, disorientation)
            civ_smoke = smoke.get_density_at(civ.pos[0], civ.pos[1])
            listener_factor = max(0.3, 1.0 - civ_smoke * 0.5)

            # Source-side: smoke at guide's position (ambient noise from fire)
            # Guide has mask so less affected, but noise still reduces effectiveness
            guide_smoke = smoke.get_density_at(self.pos[0], self.pos[1])
            source_factor = max(0.5, 1.0 - guide_smoke * 0.3)

            # Final hearing probability this tick
            hear_chance = distance_factor * listener_factor * source_factor

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


def spawn_beacon_guides(
    env: Environment, n_guides: int, speed: float = 2.0, seed: int = 77
) -> list[BeaconGuide]:
    """Spawn beacon guides at exits. Simple round-robin assignment.
    Each guide stands 2-3 cells from its assigned exit and broadcasts.
    Overlap on clustered-exit maps is a realistic Beacon limitation."""
    rng = np.random.default_rng(seed)
    exits = list(env.exits)

    guides = []
    for i in range(n_guides):
        assigned_exit = exits[i % len(exits)]
        ex, ey = assigned_exit
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


# ════════════════════════════════════════════════════════════════════
# CONGESTION-AWARE ROUTER (Zhang et al. 2020, IEEE ICRA)
# ════════════════════════════════════════════════════════════════════


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
                    cong = self._exit_counts.get((ex, ey), 0) * 30.0
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
                self._waypoints = bfs_path(
                    env, self.pos, np.array(best_exit), fire_grid=danger
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
            cur_cong = self._exit_counts.get((cx, cy), 0) * 30.0
            cur_smoke = smoke.get_density_at(cx, cy) * 200.0
            cur_score = cur_dist + cur_cong + cur_smoke

            # Check all exits for a better option
            for ex, ey in self.known_exits:
                if (ex, ey) == self._escort_exit:
                    continue
                dist = np.hypot(self.pos[0] - ex, self.pos[1] - ey)
                cong = self._exit_counts.get((ex, ey), 0) * 30.0
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
                    self._waypoints = dijkstra_congestion_path(
                        env, self.pos, np.array((ex, ey)),
                        fire_grid=fg, smoke_grid=sg,
                        civilian_density=self._density_grid,
                    )
                    # BFS fallback if Dijkstra returns empty
                    if not self._waypoints:
                        if fg is not None and sg is not None:
                            danger = np.maximum(fg, sg * 1.0)
                        elif fg is not None:
                            danger = fg
                        elif sg is not None:
                            danger = sg * 1.0
                        else:
                            danger = None
                        self._waypoints = bfs_path(
                            env, self.pos, np.array((ex, ey)), fire_grid=danger
                        )
                    break  # Switch once per re-evaluation

        if not self._waypoints:
            if self._escort_exit is not None:
                dist = np.linalg.norm(self.pos - np.array(self._escort_exit))
                if dist < env.cell_size * 2:
                    self._phase = self.RELEASING
                    self._release_wait = 0
                    self.vel = np.zeros(2)
                    return
                # STUCK FIX: waypoints empty but not near exit.
                # Recompute path using BFS with danger grid (Sweeper pattern).
                # This prevents guide standing still mid-escort.
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
                self._waypoints = bfs_path(
                    env, self.pos, np.array(self._escort_exit), fire_grid=danger
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
                score = size * (1.0 - centroid_smoke * 0.5)

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
            congestion = self._exit_counts.get((ex, ey), 0) * 30.0
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
