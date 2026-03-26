"""
Civilian agent — research-based evacuation behavior with Social Force Model.

Based on evacuation research:
  - Spawned in rooms clustered at desks (not random scatter)
  - IDLE until they sense danger, then reaction delay (0.5-3s)
  - Movement uses simplified Social Force Model:
      * Desired force: pulls toward goal (doorway, exit)
      * Wall repulsion: smooth push away from walls (no bouncing)
      * Agent repulsion: keeps distance from other civilians
  - In smoke: speed 30%, vision ZERO
  - Smoke damage accumulates → death
  - Can follow other civilians heading to exit (herding)
"""

import numpy as np
from collections import deque
from evacsim.environment import Environment, CellType
from evacsim.smoke import SmokeSystem
from evacsim.fire import FireSystem
from evacsim.guide import bfs_path


# Civilian states
IDLE = 0           # At desk, unaware of danger
REACTING = 1       # Noticed danger, reaction delay before moving
FLEEING_ROOM = 2   # Pathfinding to nearest doorway
WANDERING = 3      # In corridor, searching for exit
WALL_FOLLOWING = 4 # In smoke, can't see, feeling along wall
FOLLOWING = 5      # Following another civilian who knows the way
MOVING_TO_EXIT = 6 # Spotted an exit, heading there via BFS
ESCAPED = 7
DEAD = 8


class Civilian:
    def __init__(self, x: float, y: float, speed: float, rng: np.random.Generator,
                 vision_radius: float = 10.0):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.array([0.0, 0.0])  # Start stationary (IDLE)
        self.speed = speed
        self._rng = rng
        self.vision_radius = vision_radius  # 10 = 1 cell in clear air

        self.state = IDLE

        # Reaction delay: random 15-90 ticks (0.5-3 seconds at 30fps)
        self._reaction_delay = int(rng.uniform(15, 90))
        self._reaction_counter = 0

        # Wall-following hand rule
        self._wall_hand = 1 if rng.random() < 0.5 else -1

        # Target exit position
        self._target_exit = None

        # BFS waypoints
        self._waypoints = []

        # Following target
        self._follow_target = None

        # Last safe position
        self._last_safe_pos = self.pos.copy()

        # Stuck detection
        self._prev_pos = self.pos.copy()
        self._stuck_ticks = 0

        # Direction cooldown
        self._direction_cooldown = 0

        # Smoke damage
        self._smoke_damage = 0.0
        self._max_smoke_damage = 150.0

    # ================================================================
    # SOCIAL FORCE HELPERS
    # ================================================================

    def _wall_repulsion_force(self, env: Environment) -> np.ndarray:
        """Smooth repulsive force from nearby walls.
        Instead of bouncing off walls, civilians are gently pushed away.
        Checks 8 directions for walls within 1.5 cells."""
        x, y = self.pos
        cs = env.cell_size
        force = np.array([0.0, 0.0])

        # Check 8 directions around the agent
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            dx, dy = np.cos(angle), np.sin(angle)

            # Check at multiple distances (0.5, 1.0, 1.5 cells)
            for dist_mult in [0.5, 1.0, 1.5]:
                check_x = x + dx * cs * dist_mult
                check_y = y + dy * cs * dist_mult

                if not env.is_walkable(check_x, check_y):
                    # Wall found — push AWAY from it
                    # Closer walls push harder (inverse distance)
                    strength = 2.0 / (dist_mult + 0.5)
                    force -= np.array([dx, dy]) * strength
                    break  # Only count nearest wall in this direction

        return force

    def _agent_repulsion_force(self, all_civilians: list) -> np.ndarray:
        """Repulsive force from nearby civilians to prevent overlap."""
        force = np.array([0.0, 0.0])
        min_dist = 8.0  # World units — start repelling within this distance

        for other in all_civilians:
            if other is self:
                continue
            if other.state in (ESCAPED, DEAD):
                continue

            diff = self.pos - other.pos
            dist = np.linalg.norm(diff)

            if 0 < dist < min_dist:
                # Push away, stronger when closer
                strength = (1.0 - dist / min_dist) * 0.8
                force += (diff / dist) * strength

        return force

    def _compute_social_force(self, desired_dir: np.ndarray, current_speed: float,
                               env: Environment, all_civilians: list,
                               has_goal: bool = False) -> np.ndarray:
        """Combine desired direction with wall repulsion and agent repulsion.

        has_goal=True means civilian has a BFS path or exit target.
        When they have a goal, desired force is much stronger — they push
        through smoke and tight spaces instead of getting stuck.
        """
        if has_goal:
            # Strong desire, weak wall push — they know where to go
            desired = desired_dir * current_speed * 1.5
            wall_rep = self._wall_repulsion_force(env) * current_speed * 0.15
            agent_rep = self._agent_repulsion_force(all_civilians) * current_speed * 0.2
        else:
            # Wandering — wall repulsion matters more to avoid getting stuck
            desired = desired_dir * current_speed * 1.0
            wall_rep = self._wall_repulsion_force(env) * current_speed * 0.3
            agent_rep = self._agent_repulsion_force(all_civilians) * current_speed * 0.3

        total = desired + wall_rep + agent_rep

        speed = np.linalg.norm(total)
        if speed > current_speed * 1.3:
            total = total / speed * current_speed * 1.3

        return total

    # ================================================================
    # MAIN UPDATE
    # ================================================================

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

        # --- IDLE: at desk, unaware ---
        if self.state == IDLE:
            if in_fire or in_smoke or self._sense_nearby_danger(env, smoke, fire):
                self.state = REACTING
                self._reaction_counter = 0
            else:
                self._last_safe_pos = self.pos.copy()
                return

        # --- REACTING: noticed danger, processing (reaction delay) ---
        if self.state == REACTING:
            self._reaction_counter += 1
            if self._reaction_counter >= self._reaction_delay:
                # Done reacting — start fleeing
                doorway = self._find_nearest_doorway(env)
                if doorway is not None:
                    self._waypoints = bfs_path(env, self.pos, doorway)
                    self.state = FLEEING_ROOM
                else:
                    self.state = WANDERING
                    self._pick_new_direction()
            return  # Stand still during reaction

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
        if local_smoke > 0.5:
            effective_vision = 0.0
        elif local_smoke > 0.2:
            effective_vision = self.vision_radius * (1.0 - local_smoke * 2.0)
            effective_vision = max(0.0, effective_vision)
        else:
            effective_vision = self.vision_radius

        # --- Effective speed ---
        # Goal-directed states (have a BFS path): only mild slowdown in smoke
        # Wandering/no-goal states: heavy slowdown in smoke
        has_path = self.state in (FLEEING_ROOM, MOVING_TO_EXIT, FOLLOWING)

        if in_fire:
            current_speed = self.speed * 1.8
        elif in_smoke:
            if has_path:
                current_speed = self.speed * 0.7  # Know where to go, push through
            else:
                current_speed = self.speed * 0.3  # Lost, stumbling
        else:
            current_speed = self.speed

        # --- Check for exit ---
        if effective_vision > 0 and self.state not in (MOVING_TO_EXIT, ESCAPED, DEAD):
            nearest_exit, nearest_dist = self._find_nearest_exit(env, effective_vision)
            if nearest_exit is not None:
                self._set_exit_target(env, nearest_exit)

        # --- Check for herding ---
        if effective_vision > 0 and self.state in (WANDERING, WALL_FOLLOWING):
            leader = self._find_leader(all_civilians, effective_vision * 2)
            if leader is not None:
                self._follow_target = leader
                self.state = FOLLOWING

        # --- State behaviors ---

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
                    # Direct path following — NO social forces
                    # They know the room, they push straight to the door
                    direction = diff / dist
                    self.vel = direction * current_speed
            else:
                self.state = WANDERING
                self._pick_new_direction()

        elif self.state == MOVING_TO_EXIT and self._target_exit is not None:
            diff_exit = np.array(self._target_exit) - self.pos
            dist_exit = np.linalg.norm(diff_exit)
            if dist_exit < env.cell_size:
                self.state = ESCAPED
                return

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
                    # Direct path — push through smoke to reach exit
                    direction = diff / dist
                    self.vel = direction * current_speed
            else:
                if dist_exit > 0:
                    direction = diff_exit / dist_exit
                    self.vel = direction * current_speed

        elif self.state == FOLLOWING and self._follow_target is not None:
            target = self._follow_target
            if target.state in (ESCAPED, DEAD):
                self._follow_target = None
                self.state = WANDERING
                self._pick_new_direction()
            else:
                diff = target.pos - self.pos
                dist = np.linalg.norm(diff)
                if dist > 1.0:
                    direction = diff / dist
                    self.vel = direction * current_speed
                if effective_vision > 0:
                    nearest_exit, _ = self._find_nearest_exit(env, effective_vision)
                    if nearest_exit is not None:
                        self._set_exit_target(env, nearest_exit)

        elif self.state == WALL_FOLLOWING:
            self._do_wall_follow(env, current_speed)

        elif self.state == WANDERING:
            if self._direction_cooldown > 0:
                self._direction_cooldown -= 1
                speed_now = np.linalg.norm(self.vel)
                if speed_now > 0:
                    self.vel = self.vel / speed_now * current_speed
            elif in_smoke and effective_vision == 0:
                self.state = WALL_FOLLOWING
                self._do_wall_follow(env, current_speed)
            elif in_smoke or in_fire or self._sense_nearby_danger(env, smoke, fire):
                # KEY FIX: Don't flee FROM smoke — go TOWARD open space.
                # Smoke enters through the doorway, so "flee from smoke"
                # pushes civilians to the back of the room (death trap).
                # Instead, go toward the most open direction (= doorway).
                self._pick_open_direction(env, current_speed)
                self._direction_cooldown = 15
            else:
                self._last_safe_pos = self.pos.copy()
                if self._rng.random() < 0.03:
                    self._pick_new_direction()
                speed_now = np.linalg.norm(self.vel)
                if speed_now > 0:
                    self.vel = self.vel / speed_now * current_speed

        # --- Move (with wall collision check) ---
        vx, vy = self.vel
        new_x, new_y = x + vx, y + vy

        if env.is_walkable(new_x, new_y):
            self.pos = np.array([new_x, new_y])
        elif env.is_walkable(new_x, y):
            # Slide along X
            self.pos = np.array([new_x, y])
        elif env.is_walkable(x, new_y):
            # Slide along Y
            self.pos = np.array([x, new_y])
        else:
            # Completely blocked — pick open direction
            if self.state == WALL_FOLLOWING:
                self._wall_turn(env, current_speed)
            else:
                self._pick_open_direction(env, current_speed)

        # --- Stuck detection ---
        dist_moved = np.linalg.norm(self.pos - self._prev_pos)
        if dist_moved < 0.5:
            self._stuck_ticks += 1
        else:
            self._stuck_ticks = 0
            self._prev_pos = self.pos.copy()

        if self._stuck_ticks >= 8:
            self._pick_open_direction(env, current_speed)
            self._stuck_ticks = 0

    # ================================================================
    # HELPER METHODS
    # ================================================================

    def _find_nearest_doorway(self, env: Environment):
        """BFS to find nearest corridor cell.

        A corridor cell has a long unbroken walkable line in at least one
        cardinal direction (>= 6 cells). Room cells are enclosed by walls
        so their max straight line is shorter.

        If we're already in a corridor, returns None."""
        gr, gc = env.world_to_grid(self.pos[0], self.pos[1])
        rows, cols = env.grid.shape
        threshold = 6  # Corridor = can walk 6+ cells in a straight line

        def max_straight_line(r, c):
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

        # Already in corridor? No need to flee room
        if max_straight_line(gr, gc) >= threshold:
            return None

        # BFS outward — find nearest corridor cell
        visited = {(gr, gc)}
        queue = deque([(gr, gc, 0)])

        while queue:
            r, c, d = queue.popleft()
            if d > 30:
                continue
            if d >= 1 and max_straight_line(r, c) >= threshold:
                wx, wy = env.grid_to_world(r, c)
                return np.array([wx, wy], dtype=float)

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    if env.grid[nr, nc] != CellType.WALL:
                        visited.add((nr, nc))
                        queue.append((nr, nc, d + 1))

        return None

    def _set_exit_target(self, env: Environment, exit_pos: tuple) -> None:
        """Set exit target and compute BFS path."""
        self._target_exit = exit_pos
        self._waypoints = bfs_path(env, self.pos, np.array(exit_pos))
        self.state = MOVING_TO_EXIT

    def _flee_from_danger(self, env: Environment, smoke: SmokeSystem,
                          fire: FireSystem, current_speed: float) -> None:
        """Move toward direction with LEAST danger + most open space."""
        x, y = self.pos
        cs = env.cell_size
        candidates = []

        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            dx, dy = np.cos(angle), np.sin(angle)
            check_x, check_y = x + dx * cs, y + dy * cs

            if not env.is_walkable(check_x, check_y):
                continue

            s = smoke.get_density_at(check_x, check_y)
            f = fire.get_density_at(check_x, check_y)
            danger = s + f * 3.0

            openness = 0
            for step in range(1, 6):
                cx, cy = x + dx * cs * step, y + dy * cs * step
                if env.is_walkable(cx, cy):
                    openness += 1
                else:
                    break

            candidates.append((danger, -openness, angle))

        if candidates:
            candidates.sort()
            best_angle = candidates[0][2]
            best_dir = np.array([np.cos(best_angle), np.sin(best_angle)])
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
        """Wall-following in smoke using hand rule."""
        x, y = self.pos
        cs = env.cell_size * 0.5

        directions = [
            np.array([1.0, 0.0]),
            np.array([-1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, -1.0]),
        ]

        wall_dir = None
        for d in directions:
            if not env.is_walkable(x + d[0] * cs, y + d[1] * cs):
                wall_dir = d
                break

        if wall_dir is not None:
            if self._wall_hand == 1:
                walk_dir = np.array([-wall_dir[1], wall_dir[0]])
            else:
                walk_dir = np.array([wall_dir[1], -wall_dir[0]])
            self.vel = walk_dir * current_speed
        else:
            if np.linalg.norm(self.vel) < 0.1:
                self._pick_new_direction()
            norm = np.linalg.norm(self.vel)
            if norm > 0:
                self.vel = self.vel / norm * current_speed

    def _wall_turn(self, env: Environment, current_speed: float) -> None:
        """Turn when wall-following hits a wall ahead."""
        vx, vy = self.vel
        if self._wall_hand == 1:
            self.vel = np.array([vy, -vx])
        else:
            self.vel = np.array([-vy, vx])
        norm = np.linalg.norm(self.vel)
        if norm > 0:
            self.vel = self.vel / norm * current_speed

    def _find_nearest_exit(self, env: Environment, effective_vision: float):
        """Find nearest exit within vision range."""
        best_exit = None
        best_dist = float("inf")
        for ex, ey in env.exits:
            dist = np.sqrt((self.pos[0] - ex) ** 2 + (self.pos[1] - ey) ** 2)
            if dist < effective_vision and dist < best_dist:
                best_exit = (ex, ey)
                best_dist = dist
        return best_exit, best_dist

    def _find_leader(self, all_civilians: list, search_radius: float):
        """Find nearby civilian heading to exit."""
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

    def _sense_nearby_danger(self, env: Environment, smoke: SmokeSystem,
                              fire: FireSystem) -> bool:
        """Check for smoke/fire within vision radius."""
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
        """Pick direction biased toward most open space."""
        x, y = self.pos
        cs = env.cell_size

        best_angle = None
        best_openness = -1

        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
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
    """Create n civilians clustered in rooms."""
    positions = env.spawn_positions(n, seed=seed)
    rng = np.random.default_rng(seed)

    civilians = []
    for x, y in positions:
        civilians.append(Civilian(x, y, speed, rng))

    return civilians
