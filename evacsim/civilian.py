"""
Civilian agent — clean rewrite.

Simple state machine:
  IDLE → REACTING → NAVIGATING → ESCAPED / DEAD

Movement:
  - Has a BFS path? Follow it (doorway, exit, guide instruction).
  - No path? Walk along corridor using run-length detection.
  - Blind in smoke? Wall-follow with hand rule.

Key design: civilians are DUMB. They don't know where exits are (1-cell vision).
They need guides to survive. Without guides, they wander corridors randomly
and most die from smoke.
"""

import numpy as np
from collections import deque
from evacsim.environment import Environment, CellType
from evacsim.smoke import SmokeSystem
from evacsim.fire import FireSystem
from evacsim.guide import bfs_path


# ── States ──────────────────────────────────────────────────────────
IDLE = 0            # At desk, unaware
REACTING = 1        # Noticed danger, standing still (reaction delay)
FLEEING_ROOM = 2    # Following BFS path to corridor doorway
WANDERING = 3       # In corridor, no goal, corridor-following
WALL_FOLLOWING = 4  # Blind in smoke, hand-on-wall
FOLLOWING = 5       # Walking toward another civilian who has a goal
MOVING_TO_EXIT = 6  # Following BFS path to exit
ESCAPED = 7
DEAD = 8


class Civilian:

    def __init__(self, x: float, y: float, speed: float,
                 rng: np.random.Generator, vision_radius: float = 10.0):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.array([0.0, 0.0])
        self.speed = speed
        self._rng = rng
        self.vision_radius = vision_radius

        self.state = IDLE

        # Reaction delay (random 15-90 ticks ≈ 0.5-3s at 30fps)
        self._reaction_delay = int(rng.uniform(15, 90))
        self._reaction_counter = 0

        # BFS waypoints (shared by FLEEING_ROOM and MOVING_TO_EXIT)
        self._waypoints = []
        self._target_exit = None

        # Following target (another civilian)
        self._follow_target = None

        # Corridor-following state
        self._corridor_dir = None   # Current committed cardinal direction
        self._corridor_ticks = 0    # Ticks walking this direction

        # Wall-following hand rule (random per civilian)
        self._wall_hand = 1 if rng.random() < 0.5 else -1

        # Smoke/fire damage
        self._smoke_damage = 0.0
        self._max_smoke_damage = 150.0

        # Stuck detection
        self._prev_pos = self.pos.copy()
        self._stuck_ticks = 0

    # ================================================================
    # MAIN UPDATE — one tick
    # ================================================================

    def update(self, env: Environment, smoke: SmokeSystem,
               fire: FireSystem, all_civilians: list) -> None:

        if self.state in (ESCAPED, DEAD):
            return

        x, y = self.pos
        local_smoke = smoke.get_density_at(x, y)
        local_fire = fire.get_density_at(x, y)
        in_smoke = local_smoke > 0.3
        in_fire = local_fire > 0.1

        # ── 1. IDLE: wait until danger sensed ──
        if self.state == IDLE:
            if in_fire or in_smoke or self._sense_danger(env, smoke, fire):
                self.state = REACTING
                self._reaction_counter = 0
            return  # Do nothing while idle

        # ── 2. REACTING: freeze, then start moving ──
        if self.state == REACTING:
            self._reaction_counter += 1
            if self._reaction_counter >= self._reaction_delay:
                self._start_fleeing(env)
            return  # Stand still during reaction

        # ── 3. Smoke damage (all active states) ──
        if in_fire:
            self._smoke_damage += 3.0
        elif in_smoke:
            self._smoke_damage += local_smoke
        else:
            self._smoke_damage = max(0.0, self._smoke_damage - 0.2)

        if self._smoke_damage >= self._max_smoke_damage:
            self.state = DEAD
            self.vel = np.zeros(2)
            return

        # ── 4. Vision & speed ──
        effective_vision = self._calc_vision(local_smoke)
        current_speed = self._calc_speed(in_fire, in_smoke)

        # ── 5. Opportunistic checks (any active state) ──
        # Spot an exit?
        if effective_vision > 0 and self.state not in (MOVING_TO_EXIT,):
            exit_pos = self._spot_exit(env, effective_vision)
            if exit_pos is not None:
                self._set_exit_target(env, exit_pos)

        # Spot a civilian heading to exit? (only if wandering/wall-following)
        if effective_vision > 0 and self.state in (WANDERING, WALL_FOLLOWING):
            leader = self._spot_leader(all_civilians, effective_vision * 2)
            if leader is not None:
                self._follow_target = leader
                self.state = FOLLOWING

        # ── 6. State behavior ──
        if self.state == FLEEING_ROOM:
            self._do_follow_path(env, current_speed, fallback=WANDERING)

        elif self.state == MOVING_TO_EXIT:
            # Check if close enough to exit
            if self._target_exit is not None:
                dist = np.linalg.norm(self.pos - np.array(self._target_exit))
                if dist < env.cell_size:
                    self.state = ESCAPED
                    return
            self._do_follow_path(env, current_speed, fallback=WANDERING)

        elif self.state == FOLLOWING:
            self._do_following(env, current_speed, effective_vision)

        elif self.state == WALL_FOLLOWING:
            self._do_wall_follow(env, current_speed)
            # Regained vision? Switch back to corridor-follow
            if effective_vision > 0 and not in_smoke:
                self.state = WANDERING
                self._corridor_dir = None

        elif self.state == WANDERING:
            if in_smoke and effective_vision == 0:
                # Blind → wall-follow
                self.state = WALL_FOLLOWING
                self._corridor_dir = None
                self._do_wall_follow(env, current_speed)
            else:
                self._do_corridor_follow(env, current_speed)

        # ── 7. Apply movement ──
        self._move(env, current_speed)

    # ================================================================
    # STATE TRANSITIONS
    # ================================================================

    def _start_fleeing(self, env: Environment):
        """After reaction delay: find doorway → FLEEING_ROOM, or → WANDERING."""
        doorway = self._find_doorway(env)
        if doorway is not None:
            self._waypoints = bfs_path(env, self.pos, doorway)
            self.state = FLEEING_ROOM
        else:
            # Already in corridor or can't find one
            self.state = WANDERING
            self._corridor_dir = None

    def _set_exit_target(self, env: Environment, exit_pos):
        """Set exit as goal, compute BFS path."""
        self._target_exit = exit_pos
        self._waypoints = bfs_path(env, self.pos, np.array(exit_pos))
        self.state = MOVING_TO_EXIT

    # ================================================================
    # STATE BEHAVIORS
    # ================================================================

    def _do_follow_path(self, env, speed, fallback):
        """Follow BFS waypoints. When done, switch to fallback state."""
        if not self._waypoints:
            self.state = fallback
            self._corridor_dir = None
            return

        target = self._waypoints[0]
        diff = target - self.pos
        dist = np.linalg.norm(diff)

        if dist < env.cell_size * 0.5:
            self._waypoints.pop(0)
            if not self._waypoints:
                self.state = fallback
                self._corridor_dir = None
                return
            target = self._waypoints[0]
            diff = target - self.pos
            dist = np.linalg.norm(diff)

        if dist > 0:
            self.vel = (diff / dist) * speed

    def _do_following(self, env, speed, effective_vision):
        """Walk toward another civilian. If they're gone, wander."""
        target = self._follow_target
        if target is None or target.state in (ESCAPED, DEAD):
            self._follow_target = None
            self.state = WANDERING
            self._corridor_dir = None
            return

        diff = target.pos - self.pos
        dist = np.linalg.norm(diff)
        if dist > 1.0:
            self.vel = (diff / dist) * speed

        # Can we see an exit now?
        if effective_vision > 0:
            exit_pos = self._spot_exit(env, effective_vision)
            if exit_pos is not None:
                self._set_exit_target(env, exit_pos)

    def _do_corridor_follow(self, env, speed):
        """Walk along corridors using run-length detection.

        Measures how far you can walk in each cardinal direction.
        Only treats 3+ cell runs as real paths (ignores corridor width).
        Picks a direction and commits. Turns at real junctions.
        """
        x, y = self.pos
        cs = env.cell_size
        MIN_RUN = 3

        four = [
            np.array([1.0, 0.0]),
            np.array([-1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, -1.0]),
        ]

        # Measure run-length in each direction
        runs = {}
        for d in four:
            n = 0
            cx, cy = x + d[0] * cs, y + d[1] * cs
            while env.is_walkable(cx, cy) and n < 20:
                n += 1
                cx += d[0] * cs
                cy += d[1] * cs
            runs[(d[0], d[1])] = n

        # Real paths = directions with 3+ cells of depth
        real = [d for d in four if runs[(d[0], d[1])] >= MIN_RUN]

        if not real:
            # Tight space — try any walkable direction
            for d in four:
                if runs[(d[0], d[1])] >= 1:
                    self.vel = d * speed
                    return
            return

        # First time or reset? Pick a direction
        if self._corridor_dir is None:
            self._corridor_dir = real[self._rng.integers(len(real))]
            self._corridor_ticks = 0

        self._corridor_ticks += 1
        curr = self._corridor_dir
        curr_run = runs.get((curr[0], curr[1]), 0)

        # Wall ahead?
        if curr_run < 1:
            # Must turn. Prefer perpendicular, avoid U-turn.
            perp = [d for d in real
                    if not np.allclose(d, curr) and not np.allclose(d, -curr)]
            if not perp:
                perp = [d for d in real if not np.allclose(d, curr)]
            if perp:
                self._corridor_dir = perp[self._rng.integers(len(perp))]
            self._corridor_ticks = 0

        elif curr_run <= 2:
            # Approaching wall — turn now if there's a side path
            perp = [d for d in real
                    if not np.allclose(d, curr) and not np.allclose(d, -curr)]
            if perp:
                self._corridor_dir = perp[self._rng.integers(len(perp))]
                self._corridor_ticks = 0

        else:
            # Path is clear. Check for real junctions.
            perp = [d for d in real
                    if not np.allclose(d, curr) and not np.allclose(d, -curr)]

            if perp and self._corridor_ticks > 10:
                # Real junction — 20% chance to turn
                if self._rng.random() < 0.20:
                    self._corridor_dir = perp[self._rng.integers(len(perp))]
                    self._corridor_ticks = 0

        self.vel = self._corridor_dir * speed

    def _do_wall_follow(self, env, speed):
        """Blind in smoke: feel along wall with hand rule."""
        x, y = self.pos
        cs = env.cell_size * 0.5

        # Find nearest wall direction
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if not env.is_walkable(x + dx * cs, y + dy * cs):
                wall = np.array([dx, dy], dtype=float)
                # Walk perpendicular to wall
                if self._wall_hand == 1:
                    self.vel = np.array([-wall[1], wall[0]]) * speed
                else:
                    self.vel = np.array([wall[1], -wall[0]]) * speed
                return

        # No wall nearby — walk forward until we find one
        norm = np.linalg.norm(self.vel)
        if norm > 0:
            self.vel = (self.vel / norm) * speed
        else:
            angle = self._rng.uniform(0, 2 * np.pi)
            self.vel = np.array([np.cos(angle), np.sin(angle)]) * speed

    # ================================================================
    # MOVEMENT
    # ================================================================

    def _move(self, env, speed):
        """Apply velocity with wall collision. Detect stuck."""
        x, y = self.pos
        vx, vy = self.vel

        if env.is_walkable(x + vx, y + vy):
            self.pos = np.array([x + vx, y + vy])
        elif env.is_walkable(x + vx, y):
            self.pos = np.array([x + vx, y])
        elif env.is_walkable(x, y + vy):
            self.pos = np.array([x, y + vy])
        else:
            # Blocked — reset corridor direction
            self._corridor_dir = None

        # Stuck detection
        moved = np.linalg.norm(self.pos - self._prev_pos)
        if moved < 0.5:
            self._stuck_ticks += 1
        else:
            self._stuck_ticks = 0
            self._prev_pos = self.pos.copy()

        if self._stuck_ticks >= 10:
            self._corridor_dir = None
            # Try random walkable direction
            angle = self._rng.uniform(0, 2 * np.pi)
            self.vel = np.array([np.cos(angle), np.sin(angle)]) * speed
            self._stuck_ticks = 0

    # ================================================================
    # PERCEPTION (what can the civilian sense?)
    # ================================================================

    def _calc_vision(self, smoke_density):
        """Effective vision radius. Smoke reduces it to zero."""
        if smoke_density > 0.5:
            return 0.0
        elif smoke_density > 0.2:
            return self.vision_radius * max(0.0, 1.0 - smoke_density * 2.0)
        return self.vision_radius

    def _calc_speed(self, in_fire, in_smoke):
        """Effective speed. Fire = sprint, smoke = slow, clear = normal.
        If following a path, only mildly slowed in smoke."""
        has_path = self.state in (FLEEING_ROOM, MOVING_TO_EXIT, FOLLOWING)
        if in_fire:
            return self.speed * 1.8
        elif in_smoke:
            return self.speed * (0.7 if has_path else 0.3)
        return self.speed

    def _sense_danger(self, env, smoke, fire):
        """Can we see/smell smoke or fire within 1 cell?"""
        gr, gc = env.world_to_grid(self.pos[0], self.pos[1])
        rows, cols = smoke.density.shape
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r, c = gr + dr, gc + dc
                if 0 <= r < rows and 0 <= c < cols:
                    if smoke.density[r, c] > 0.1 or fire.density[r, c] > 0.05:
                        return True
        return False

    def _spot_exit(self, env, effective_vision):
        """Find nearest exit within vision range. Returns pos or None."""
        best, best_d = None, float("inf")
        for ex, ey in env.exits:
            d = np.hypot(self.pos[0] - ex, self.pos[1] - ey)
            if d < effective_vision and d < best_d:
                best, best_d = (ex, ey), d
        return best

    def _spot_leader(self, all_civilians, radius):
        """Find nearest civilian heading to exit."""
        best, best_d = None, float("inf")
        for other in all_civilians:
            if other is self or other.state not in (MOVING_TO_EXIT,):
                continue
            d = np.linalg.norm(self.pos - other.pos)
            if d < radius and d < best_d:
                best, best_d = other, d
        return best

    # ================================================================
    # PATHFINDING HELPERS
    # ================================================================

    def _find_doorway(self, env):
        """BFS to find nearest corridor cell (6+ straight-line walk).
        Returns world-coordinate position, or None if already in corridor."""
        gr, gc = env.world_to_grid(self.pos[0], self.pos[1])
        rows, cols = env.grid.shape

        def longest_run(r, c):
            best = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                n, nr, nc = 0, r + dr, c + dc
                while 0 <= nr < rows and 0 <= nc < cols and env.grid[nr, nc] != CellType.WALL:
                    n += 1
                    nr += dr
                    nc += dc
                best = max(best, n)
            return best

        if longest_run(gr, gc) >= 6:
            return None  # Already in corridor

        visited = {(gr, gc)}
        queue = deque([(gr, gc, 0)])
        while queue:
            r, c, d = queue.popleft()
            if d > 25:
                continue
            if d >= 1 and longest_run(r, c) >= 6:
                wx, wy = env.grid_to_world(r, c)
                return np.array([wx, wy], dtype=float)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    if env.grid[nr, nc] != CellType.WALL:
                        visited.add((nr, nc))
                        queue.append((nr, nc, d + 1))
        return None


# ════════════════════════════════════════════════════════════════════
# SPAWNING
# ════════════════════════════════════════════════════════════════════

def spawn_civilians(env: Environment, n: int, speed: float = 1.5,
                    seed: int = 99) -> list[Civilian]:
    """Create n civilians at random floor positions."""
    positions = env.spawn_positions(n, seed=seed)
    rng = np.random.default_rng(seed)
    return [Civilian(x, y, speed, rng) for x, y in positions]
