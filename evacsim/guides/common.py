"""Common helper functions for guide agents.

Pure algorithmic primitives, extracted from the monolithic guide.py:
  - bfs_path: BFS pathfinding with optional fire avoidance
  - dijkstra_congestion_path: density-weighted pathfinding (Zhang 2020)
  - predict_spread: deterministic CA forward simulation
  - dbscan_cluster: spatial clustering (Ester 1996)
  - build_density_grid: per-cell civilian density for Router
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
