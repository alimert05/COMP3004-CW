"""
Civilian agent — research-based evacuation behavior.

Based on evacuation research findings:
  - In clear air: wander randomly, spot exit only within 1 cell
  - In smoke: speed drops to 30%, vision drops to ZERO
  - In smoke with no vision: switch to WALL-FOLLOWING (feel along walls)
  - Herding: if a civilian sees another civilian heading to an exit, follow them
  - Smoke damage accumulates; too much = death
  - Stay IDLE until they sense danger (realistic: people don't evacuate until they know)
"""

import numpy as np
from collections import deque
from evacsim.environment import Environment, CellType
from evacsim.smoke import SmokeSystem
from evacsim.fire import FireSystem
from evacsim.guide import bfs_path


# Civilian states
IDLE = 0           # Standing still, unaware of danger
FLEEING_ROOM = 1   # Detected danger, pathfinding to nearest doorway
WANDERING = 2      # In corridor, moving randomly
WALL_FOLLOWING = 3 # In smoke, can't see, following a wall
FOLLOWING = 4      # Following another civilian who knows the way
MOVING_TO_EXIT = 5 # Spotted an exit, heading straight for it
ESCAPED = 6
DEAD = 7


class Civilian:
    def __init__(self, x: float, y: float, speed: float, rng: np.random.Generator,
                 vision_radius: float = 10.0):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.array([0.0, 0.0])  # Start stationary (IDLE)
        self.speed = speed
        self._rng = rng
        self.vision_radius = vision_radius  # 10 = 1 cell in clear air

        self.state = IDLE

        # Wall-following: which hand on the wall? (1 = right-hand rule, -1 = left)
        self._wall_hand = 1 if rng.random() < 0.5 else -1

        # Target exit position
        self._target_exit = None

        # BFS waypoints to exit (computed when exit is found)
        self._waypoints = []

        # Who we're following (another Civilian reference)
        self._follow_target = None

        # Last safe position (no smoke/fire) — retreat here when danger spotted
        self._last_safe_pos = self.pos.copy()

        # Stuck detection
        self._prev_pos = self.pos.copy()
        self._stuck_ticks = 0

        # Direction cooldown — stick with chosen direction for a few ticks
        self._direction_cooldown = 0

        # Smoke damage
        self._smoke_damage = 0.0
        self._max_smoke_damage = 150.0

    def update(self, env: Environment, smoke: SmokeSystem, fire: FireSystem,
               all_civilians: list) -> None:
        """Move one tick."""
        if self.state in (ESCAPED, DEAD):
            return

        x, y = self.pos

        # --- What are we standing in? ---
        local_smoke = smoke.get_density_at(x, y)
        local_fire = fire.get_density_at(x, y)

        in_smoke = local_smoke > 0.3
        in_fire = local_fire > 0.1

        # --- IDLE: stay still until we sense danger ---
        if self.state == IDLE:
            if in_fire or in_smoke or self._sense_nearby_danger(env, smoke, fire):
                # Danger detected — find doorway and flee the room
                doorway = self._find_nearest_doorway(env)
                if doorway is not None:
                    self._waypoints = bfs_path(env, self.pos, doorway)
                    self.state = FLEEING_ROOM
                else:
                    # Already in corridor or can't find doorway — just wander
                    self.state = WANDERING
                    self._flee_from_danger(env, smoke, fire, self.speed)
            else:
                self._last_safe_pos = self.pos.copy()
                return  # Stay still

        # --- Smoke damage ---
        if in_fire:
            self._smoke_damage += 3.0
        elif in_smoke:
            self._smoke_damage += local_smoke
        else:
            self._smoke_damage = max(0.0, self._smoke_damage - 0.2)

        if self._smoke_damage >= self._max_smoke_damage:
            self.state = DEAD
            self.vel = np.array([0.0, 0.0])
            return

        # --- Effective vision ---
        # Clear air: full vision (10 units = 1 cell)
        # In smoke: vision drops to ZERO (can't see anything)
        if local_smoke > 0.5:
            effective_vision = 0.0
        elif local_smoke > 0.2:
            effective_vision = self.vision_radius * (1.0 - local_smoke * 2.0)
            effective_vision = max(0.0, effective_vision)
        else:
            effective_vision = self.vision_radius

        # --- Effective speed ---
        # In fire: sprint (panic)
        # In smoke: 30% speed (research-based)
        # Clear: normal
        if in_fire:
            current_speed = self.speed * 1.8
        elif in_smoke:
            current_speed = self.speed * 0.3
        else:
            current_speed = self.speed

        # --- Check for exit (only if we can see) ---
        if effective_vision > 0 and self.state not in (MOVING_TO_EXIT, ESCAPED, DEAD):
            nearest_exit, nearest_dist = self._find_nearest_exit(env, effective_vision)
            if nearest_exit is not None:
                self._set_exit_target(env, nearest_exit)

        # --- Check for herding (see another civilian heading to exit?) ---
        if effective_vision > 0 and self.state in (WANDERING, WALL_FOLLOWING):
            leader = self._find_leader(all_civilians, effective_vision * 2)
            if leader is not None:
                self._follow_target = leader
                self.state = FOLLOWING

        # --- Decide what to do based on state ---

        # FLEEING_ROOM: follow BFS waypoints to doorway, then switch to WANDERING
        if self.state == FLEEING_ROOM:
            if self._waypoints:
                target = self._waypoints[0]
                diff = target - self.pos
                dist = np.linalg.norm(diff)

                if dist < env.cell_size * 0.5:
                    self._waypoints.pop(0)
                    if self._waypoints:
                        target = self._waypoints[0]
                        diff = target - self.pos
                        dist = np.linalg.norm(diff)

                if dist > 0:
                    direction = diff / dist
                    self.vel = direction * current_speed
            else:
                # Reached doorway — now wander in the corridor
                self.state = WANDERING
                self._flee_from_danger(env, smoke, fire, current_speed)

        elif self.state == MOVING_TO_EXIT and self._target_exit is not None:
            # Check if we reached the exit
            diff_exit = np.array(self._target_exit) - self.pos
            dist_exit = np.linalg.norm(diff_exit)
            if dist_exit < env.cell_size:
                self.state = ESCAPED
                return

            # Follow BFS waypoints
            if self._waypoints:
                target = self._waypoints[0]
                diff = target - self.pos
                dist = np.linalg.norm(diff)

                if dist < env.cell_size * 0.5:
                    # Reached waypoint, move to next
                    self._waypoints.pop(0)
                    if self._waypoints:
                        target = self._waypoints[0]
                        diff = target - self.pos
                        dist = np.linalg.norm(diff)

                if dist > 0:
                    direction = diff / dist
                    self.vel = direction * current_speed
            else:
                # No waypoints left, walk directly to exit
                if dist_exit > 0:
                    direction = diff_exit / dist_exit
                    self.vel = direction * current_speed

        elif self.state == FOLLOWING and self._follow_target is not None:
            # Follow the leader civilian
            target = self._follow_target
            if target.state in (ESCAPED, DEAD):
                # Leader gone — go back to wandering
                self._follow_target = None
                self.state = WANDERING
                self._pick_new_direction()
            else:
                diff = target.pos - self.pos
                dist = np.linalg.norm(diff)
                if dist > 1.0:
                    direction = diff / dist
                    self.vel = direction * current_speed
                # If leader reached exit, we should be close to it too
                # Check if we can now see the exit ourselves
                if effective_vision > 0:
                    nearest_exit, _ = self._find_nearest_exit(env, effective_vision)
                    if nearest_exit is not None:
                        self._set_exit_target(env, nearest_exit)

        elif self.state == WALL_FOLLOWING:
            self._do_wall_follow(env, current_speed)

        elif self.state == WANDERING:
            # Tick down cooldown
            if self._direction_cooldown > 0:
                self._direction_cooldown -= 1
                # Keep current direction, just scale speed
                speed_now = np.linalg.norm(self.vel)
                if speed_now > 0:
                    self.vel = self.vel / speed_now * current_speed
            # In smoke and can't see? Switch to wall-following
            elif in_smoke and effective_vision == 0:
                self.state = WALL_FOLLOWING
                self._do_wall_follow(env, current_speed)
            elif in_smoke or in_fire:
                # In danger — flee, then keep that direction for 10 ticks
                self._flee_from_danger(env, smoke, fire, current_speed)
                self._direction_cooldown = 10
            else:
                # In clear air — remember safe position
                self._last_safe_pos = self.pos.copy()
                if self._sense_nearby_danger(env, smoke, fire):
                    self._flee_from_danger(env, smoke, fire, current_speed)
                    self._direction_cooldown = 10
                else:
                    # Random wandering with occasional direction change
                    if self._rng.random() < 0.03:
                        self._pick_new_direction()
                    speed_now = np.linalg.norm(self.vel)
                    if speed_now > 0:
                        self.vel = self.vel / speed_now * current_speed

        # --- Move ---
        vx, vy = self.vel
        moved = False

        if env.is_walkable(x + vx, y + vy):
            self.pos = np.array([x + vx, y + vy])
            moved = True

        if not moved:
            # Hit a wall
            if self.state == WALL_FOLLOWING:
                self._wall_turn(env, current_speed)
            else:
                # Pick direction biased toward open space (finds doorways)
                self._pick_open_direction(env, current_speed)

        # --- Stuck detection ---
        dist_moved = np.linalg.norm(self.pos - self._prev_pos)
        if dist_moved < 0.5:
            self._stuck_ticks += 1
        else:
            self._stuck_ticks = 0
            self._prev_pos = self.pos.copy()

        if self._stuck_ticks >= 8:
            # Really stuck — try hard to find an opening
            self._pick_open_direction(env, current_speed)
            self._stuck_ticks = 0

    def _find_nearest_doorway(self, env: Environment):
        """Find the nearest corridor cell using BFS.

        A corridor cell = you can walk 10+ cells in a straight line
        (horizontally or vertically) without hitting a wall.
        Room cells are enclosed, so max straight-line walk is short (5-7).

        BFS from our position, return the first corridor cell found.
        The BFS path naturally goes through the doorway."""
        gr, gc = env.world_to_grid(self.pos[0], self.pos[1])
        rows, cols = env.grid.shape

        def max_straight_line(r, c):
            """Longest straight-line walk from (r,c) in any cardinal direction."""
            best = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                count = 0
                nr, nc = r + dr, c + dc
                while 0 <= nr < rows and 0 <= nc < cols and env.grid[nr, nc] != CellType.WALL:
                    count += 1
                    nr += dr
                    nc += dc
                best = max(best, count)
            return best

        # Check if we're already in a corridor
        if max_straight_line(gr, gc) >= 10:
            return None  # Already in corridor, no need to flee room

        # BFS outward — find the nearest cell with a long straight-line walk
        visited = {(gr, gc)}
        queue = deque([(gr, gc, 0)])

        while queue:
            r, c, d = queue.popleft()
            if d > 20:
                continue

            # Check if this cell is in a corridor (long straight walk)
            if d >= 1 and max_straight_line(r, c) >= 10:
                wx, wy = env.grid_to_world(r, c)
                return np.array([wx, wy], dtype=float)

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    if env.grid[nr, nc] != CellType.WALL:
                        visited.add((nr, nc))
                        queue.append((nr, nc, d + 1))

        return None  # Can't find corridor — already in one or map is weird

    def _set_exit_target(self, env: Environment, exit_pos: tuple) -> None:
        """Set exit target and compute BFS path to it."""
        self._target_exit = exit_pos
        self._waypoints = bfs_path(env, self.pos, np.array(exit_pos))
        self.state = MOVING_TO_EXIT

    def _flee_from_danger(self, env: Environment, smoke: SmokeSystem,
                          fire: FireSystem, current_speed: float) -> None:
        """Move toward the direction with LEAST smoke/fire.
        When multiple directions have equal danger (e.g. no smoke yet),
        prefer the direction with the most open space ahead — this
        naturally pulls civilians toward doorways and corridors."""
        x, y = self.pos
        cs = env.cell_size

        candidates = []

        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            dx = np.cos(angle)
            dy = np.sin(angle)
            check_x, check_y = x + dx * cs, y + dy * cs

            if not env.is_walkable(check_x, check_y):
                continue

            # Danger at that cell
            s = smoke.get_density_at(check_x, check_y)
            f = fire.get_density_at(check_x, check_y)
            danger = s + f * 3.0

            # Count how many cells we can walk in this direction (openness)
            # More open = corridor/doorway, less open = wall ahead
            openness = 0
            for step in range(1, 6):
                cx = x + dx * cs * step
                cy = y + dy * cs * step
                if env.is_walkable(cx, cy):
                    openness += 1
                else:
                    break

            candidates.append((danger, -openness, angle))

        if candidates:
            # Sort: lowest danger first, then most open space (negative so it sorts first)
            candidates.sort()
            best_angle = candidates[0][2]
            best_dir = np.array([np.cos(best_angle), np.sin(best_angle)])

            # Small noise so they don't all move identically
            noise_angle = self._rng.uniform(-0.3, 0.3)
            cos_n, sin_n = np.cos(noise_angle), np.sin(noise_angle)
            best_dir = np.array([
                best_dir[0] * cos_n - best_dir[1] * sin_n,
                best_dir[0] * sin_n + best_dir[1] * cos_n,
            ])
            self.vel = best_dir * current_speed
        else:
            self._pick_new_direction()
            norm = np.linalg.norm(self.vel)
            if norm > 0:
                self.vel = self.vel / norm * current_speed

    def _do_wall_follow(self, env: Environment, current_speed: float) -> None:
        """
        Wall-following in smoke (research: search wall → move toward wall → walk along wall).
        Uses right-hand or left-hand rule to trace walls.
        """
        x, y = self.pos
        cs = env.cell_size * 0.5  # half cell for checking nearby

        # Find direction along the wall
        # Check 4 cardinal directions for walls
        directions = [
            np.array([1.0, 0.0]),   # right
            np.array([-1.0, 0.0]),  # left
            np.array([0.0, 1.0]),   # down
            np.array([0.0, -1.0]),  # up
        ]

        # First: find a wall nearby
        wall_dir = None
        for d in directions:
            check_x = x + d[0] * cs
            check_y = y + d[1] * cs
            if not env.is_walkable(check_x, check_y):
                wall_dir = d
                break

        if wall_dir is not None:
            # Walk perpendicular to the wall (using hand rule)
            if self._wall_hand == 1:
                # Right-hand rule: wall on right, walk perpendicular
                walk_dir = np.array([-wall_dir[1], wall_dir[0]])
            else:
                walk_dir = np.array([wall_dir[1], -wall_dir[0]])
            self.vel = walk_dir * current_speed
        else:
            # No wall nearby — move toward nearest wall
            # Pick a random direction and walk until we hit something
            if np.linalg.norm(self.vel) < 0.1:
                self._pick_new_direction()
            norm = np.linalg.norm(self.vel)
            if norm > 0:
                self.vel = self.vel / norm * current_speed

    def _wall_turn(self, env: Environment, current_speed: float) -> None:
        """When wall-following and hitting a wall ahead, turn according to hand rule."""
        # Rotate velocity 90 degrees in the hand direction
        vx, vy = self.vel
        if self._wall_hand == 1:
            self.vel = np.array([vy, -vx])  # turn right
        else:
            self.vel = np.array([-vy, vx])  # turn left
        norm = np.linalg.norm(self.vel)
        if norm > 0:
            self.vel = self.vel / norm * current_speed

    def _find_nearest_exit(self, env: Environment, effective_vision: float):
        """Find nearest exit within effective vision range."""
        best_exit = None
        best_dist = float("inf")

        for ex, ey in env.exits:
            dist = np.sqrt((self.pos[0] - ex) ** 2 + (self.pos[1] - ey) ** 2)
            if dist < effective_vision and dist < best_dist:
                best_exit = (ex, ey)
                best_dist = dist

        return best_exit, best_dist

    def _find_leader(self, all_civilians: list, search_radius: float):
        """Find a nearby civilian who is heading to an exit (MOVING_TO_EXIT state)."""
        best = None
        best_dist = float("inf")

        for other in all_civilians:
            if other is self:
                continue
            if other.state not in (MOVING_TO_EXIT, FOLLOWING):
                continue
            dist = np.linalg.norm(self.pos - other.pos)
            if dist < search_radius and dist < best_dist:
                best = other
                best_dist = dist

        return best

    def _sense_nearby_danger(self, env: Environment, smoke: SmokeSystem, fire: FireSystem) -> bool:
        """Check if there's smoke or fire within vision radius."""
        gr, gc = env.world_to_grid(self.pos[0], self.pos[1])
        vision_cells = max(1, int(self.vision_radius / env.cell_size))
        rows, cols = smoke.density.shape

        for dr in range(-vision_cells, vision_cells + 1):
            for dc in range(-vision_cells, vision_cells + 1):
                r, c = gr + dr, gc + dc
                if 0 <= r < rows and 0 <= c < cols:
                    if smoke.density[r, c] > 0.1 or fire.density[r, c] > 0.05:
                        return True
        return False

    def _pick_open_direction(self, env: Environment, current_speed: float) -> None:
        """Pick a new direction biased toward the most open space.
        This naturally pulls civilians toward doorways and corridors
        instead of bouncing randomly inside rooms."""
        x, y = self.pos
        cs = env.cell_size

        best_angle = None
        best_openness = -1

        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        # Shuffle so ties are broken randomly
        self._rng.shuffle(angles)

        for angle in angles:
            dx, dy = np.cos(angle), np.sin(angle)
            if not env.is_walkable(x + dx * cs, y + dy * cs):
                continue

            openness = 0
            for step in range(1, 8):
                if env.is_walkable(x + dx * cs * step, y + dy * cs * step):
                    openness += 1
                else:
                    break

            if openness > best_openness:
                best_openness = openness
                best_angle = angle

        if best_angle is not None:
            self.vel = np.array([np.cos(best_angle), np.sin(best_angle)]) * current_speed
        else:
            self._pick_new_direction()

    def _pick_new_direction(self) -> None:
        angle = self._rng.uniform(0, 2 * np.pi)
        self.vel = np.array([np.cos(angle), np.sin(angle)]) * self.speed


def spawn_civilians(env: Environment, n: int, speed: float = 1.5,
                    seed: int = 99) -> list[Civilian]:
    """Create n civilians on random floor cells."""
    positions = env.spawn_positions(n, seed=seed)
    rng = np.random.default_rng(seed)

    civilians = []
    for x, y in positions:
        civilians.append(Civilian(x, y, speed, rng))

    return civilians
