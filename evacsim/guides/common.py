"""Shared algorithmic helpers used by the guide policies and civilians.

Functions:
    bfs_path                  BFS path-finding with optional fire avoidance
    dijkstra_congestion_path  Cost-weighted Dijkstra path-finding
    predict_spread            Deterministic forward simulation of a CA grid
    dbscan_cluster            Density-based spatial clustering of points
    build_density_grid        Per-cell civilian density with a 3x3 blur

These primitives are stateless and pure: each call takes its inputs and
returns a result without mutating any global state. They are imported by
several guide policies and by the civilian module.
"""
import numpy as np
from collections import deque
from evacsim.environment import Environment, CellType


def bfs_path(env: Environment, start_world, goal_world, fire_grid=None):
    """Find a path from start to goal on the grid using breadth-first search.

    If fire_grid is provided, the search first attempts a path that
    avoids cells with fire density above 0.3. If no such path exists,
    it retries with the fire constraint relaxed so the agent can still
    escape through fire as a last resort.

    Returns a list of world-coordinate waypoints, or an empty list if no
    path exists at all.
    """

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
                # Reconstruct the path by walking the parent chain.
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
                    if avoid_fire and fire_grid is not None:
                        if fire_grid[nr, nc] > 0.3:
                            continue
                    visited[(nr, nc)] = (r, c)
                    queue.append((nr, nc))

        # None signals 'no path at all', distinct from [] which means
        # the start cell is the goal.
        return None

    if fire_grid is not None:
        # First attempt: avoid fire entirely.
        result = _bfs(avoid_fire=True)
        if result is not None:
            return result
        # No safe path. Retry while allowing fire as a last resort.
        result = _bfs(avoid_fire=False)
        return result if result is not None else []
    else:
        result = _bfs(avoid_fire=False)
        return result if result is not None else []


def dijkstra_congestion_path(
    env, start_world, goal_world, fire_grid=None, smoke_grid=None, civilian_density=None
):
    """Cost-weighted Dijkstra path that prefers safer and less-congested cells.

    Per-cell traversal cost is the sum of:
        base_cost      1.0 per cell
        fire penalty   fire_density x 20 (impassable above 0.5)
        smoke penalty  smoke_density x 5
        density penalty civilian_density x 8

    A longer path through clear corridors is therefore preferred over a
    shorter path through a jammed doorway or a smoky one. Any of the
    optional grids may be None, in which case its term contributes zero.

    civilian_density is a 2D numpy array indexed by (row, col) holding
    blurred civilian counts (see build_density_grid).
    """
    import heapq

    sr, sc = env.world_to_grid(start_world[0], start_world[1])
    gr, gc = env.world_to_grid(goal_world[0], goal_world[1])
    rows, cols = env.grid.shape

    if sr == gr and sc == gc:
        return []

    cost = np.full((rows, cols), float("inf"))
    cost[sr, sc] = 0.0
    parent = {}
    parent[(sr, sc)] = None

    pq = [(0.0, sr, sc)]

    while pq:
        c, r, cc = heapq.heappop(pq)

        if r == gr and cc == gc:
            # Reconstruct the path.
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
            continue  # Stale priority-queue entry.

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, cc + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if env.grid[nr, nc] == CellType.WALL:
                continue

            move_cost = 1.0

            if fire_grid is not None:
                fire_val = fire_grid[nr, nc]
                if fire_val > 0.5:
                    continue  # Impassable.
                move_cost += fire_val * 20.0

            if smoke_grid is not None:
                smoke_val = smoke_grid[nr, nc]
                move_cost += smoke_val * 5.0

            if civilian_density is not None:
                density = civilian_density[nr, nc]
                move_cost += density * 8.0

            new_cost = cost[r, cc] + move_cost
            if new_cost < cost[nr, nc]:
                cost[nr, nc] = new_cost
                parent[(nr, nc)] = (r, cc)
                heapq.heappush(pq, (new_cost, nr, nc))

    return []  # No path found.


def predict_spread(density, spreadable_mask, source_cells, spread_rate, steps):
    """Forward-simulate a fire or smoke CA grid for a given number of ticks.

    Used by RouterGuide for deliberative planning: the predicted grid
    represents where the hazard will be when the escort group arrives,
    allowing the planner to avoid corridors that are clear now but will
    become dangerous during transit.

    Random noise is replaced by its expected value (1.0), so this
    function is deterministic: identical inputs always produce identical
    outputs. The result is the expected-value approximation of the
    stochastic CA evolution.

    Parameters:
        density          Current fire or smoke density grid (2D numpy array).
        spreadable_mask  Boolean grid where True marks cells that can hold
                         the hazard (FLOOR cells but not walls or exits).
        source_cells     List of (row, col) source positions held at 1.0.
        spread_rate      Per-tick spread coefficient (0.005 for fire, 0.02
                         for smoke).
        steps            Number of ticks to advance the simulation.

    Returns the predicted density grid as a new array; the input is not
    modified.
    """
    d = density.copy()
    for _ in range(steps):
        padded = np.pad(d, 1, mode="constant", constant_values=0.0)
        max_neighbour = np.maximum(
            np.maximum(padded[0:-2, 1:-1], padded[2:, 1:-1]),
            np.maximum(padded[1:-1, 0:-2], padded[1:-1, 2:]),
        )
        # Deterministic step: no random noise term, expected value 1.0.
        delta = spread_rate * np.maximum(0.0, max_neighbour - d)
        d = np.where(spreadable_mask, np.minimum(1.0, d + delta), 0.0)
        for r, c in source_cells:
            d[r, c] = 1.0
    return d


def dbscan_cluster(points, epsilon, min_points):
    """Density-based spatial clustering of an unordered point set.

    Finds clusters of arbitrary shape without needing the cluster count
    in advance. Used by RouterGuide to identify civilian groups that
    need assistance.

    Parameters:
        points       List of position vectors (world coordinates). Each
                     entry must be indexable as point[0], point[1].
        epsilon      Neighbourhood radius. Two points are neighbours if
                     their Euclidean distance is at most epsilon.
        min_points   Minimum number of neighbours (including the point
                     itself) required to make a point a core point.
                     Points with fewer neighbours are noise unless they
                     are reachable from a core point.

    Returns a list of clusters; each cluster is a list of indices into
    'points'. Noise points are excluded. The list is empty if no
    clusters are found.
    """
    n = len(points)
    if n == 0:
        return []

    UNVISITED = -1
    NOISE = -2
    labels = [UNVISITED] * n
    cluster_id = 0

    # Compare squared distances to avoid the per-pair sqrt.
    eps_sq = epsilon * epsilon

    def range_query(idx):
        """Return the indices of all points within epsilon of point idx."""
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

        # Start a new cluster from this core point.
        labels[i] = cluster_id
        seed_set = set(neighbors) - {i}
        processed = {i}

        while seed_set:
            q = seed_set.pop()
            if labels[q] == NOISE:
                labels[q] = cluster_id  # Border point joins the cluster.
            if labels[q] != UNVISITED:
                processed.add(q)
                continue
            labels[q] = cluster_id
            processed.add(q)

            q_neighbors = range_query(q)
            if len(q_neighbors) >= min_points:
                # q is itself a core point, so expand the cluster outward.
                seed_set.update(set(q_neighbors) - processed)

        cluster_id += 1

    # Group point indices by their final cluster label, dropping noise.
    clusters = {}
    for i, label in enumerate(labels):
        if label >= 0:
            clusters.setdefault(label, []).append(i)

    return list(clusters.values())


def build_density_grid(env, civilians):
    """Build a 2D civilian-density grid with a 3x3 blur for routing.

    Each living civilian contributes 1.0 at the cell they occupy and
    0.5 at each of the eight adjacent cells. The blur ensures that
    corridors near a crowd are penalised in the Dijkstra cost, not just
    the exact cells civilians stand on. Without it the density is too
    sparse for Dijkstra to find paths that differ meaningfully from BFS.

    Civilians in the ESCAPED or DEAD state contribute nothing.
    """
    from evacsim.civilian import ESCAPED, DEAD

    rows, cols = env.grid.shape
    grid = np.zeros((rows, cols), dtype=float)
    for civ in civilians:
        if civ.state in (ESCAPED, DEAD):
            continue
        r, c = env.world_to_grid(civ.pos[0], civ.pos[1])
        if 0 <= r < rows and 0 <= c < cols:
            grid[r, c] += 1.0
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        grid[nr, nc] += 0.5
    return grid
