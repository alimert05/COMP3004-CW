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


def bfs_path(env: Environment, start_world, goal_world):
    """Find a path from start to goal using BFS on the grid.
    Returns a list of world-coordinate waypoints, or empty list if no path."""
    sr, sc = env.world_to_grid(start_world[0], start_world[1])
    gr, gc = env.world_to_grid(goal_world[0], goal_world[1])
    rows, cols = env.grid.shape

    if sr == gr and sc == gc:
        return []

    visited = {(sr, sc): None}  # cell -> parent cell
    queue = deque([(sr, sc)])

    while queue:
        r, c = queue.popleft()

        if r == gr and c == gc:
            # Reconstruct path
            path = []
            current = (r, c)
            while current is not None:
                path.append(current)
                current = visited[current]
            path.reverse()

            # Convert grid cells to world coordinates (skip the start cell)
            waypoints = []
            for pr, pc in path[1:]:
                wx, wy = env.grid_to_world(pr, pc)
                waypoints.append(np.array([wx, wy], dtype=float))
            return waypoints

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                if env.grid[nr, nc] != CellType.WALL:
                    visited[(nr, nc)] = (r, c)
                    queue.append((nr, nc))

    return []  # No path found


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
            self._escape_waypoints = bfs_path(env, self.pos, best_exit)

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
        """Follow the pre-computed waypoint path."""
        if not self._waypoints:
            self.vel = np.array([0.0, 0.0])
            return

        target = self._waypoints[0]
        diff = target - self.pos
        dist = np.linalg.norm(diff)

        if dist < self.env.cell_size * 0.5:
            # Reached this waypoint, move to next
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
