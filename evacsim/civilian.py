"""
Civilian agent — paper-grounded 5-rule model.

Academic basis:
  - Potency system: Kasereka et al. 2018 (energy 50-100, decreases in smoke/fire, no recovery)
  - Reaction delay: Proulx 2001 (pre-evacuation time 0.5-2s)
  - Random walk: Standard ABM approach (Senanayake et al. 2024 review)
  - Disorientation in smoke: Smart Cities MDPI 2024 (random direction change proportional to density)
  - Herding: Helbing et al. 2000 (follow average crowd direction, not individual)
  - Speed in smoke: Jin 1978 (visibility vs walking speed)
  - BFS pathfinding: Dijkstra variant (Kasereka 2018)

5 Rules:
  1. No target → random walk (pick cardinal direction, walk until wall)
  2. Has target → follow BFS path (push through smoke)
  3. In smoke → disorientation (random direction change, probability ∝ density)
  4. Nearby civilians moving → herding (bias toward average crowd direction)
  5. Potency ≤ 0 → dead
"""

import numpy as np
from evacsim.environment import Environment, CellType
from evacsim.smoke import SmokeSystem
from evacsim.fire import FireSystem
from evacsim.guide import bfs_path


# ── States (simplified) ─────────────────────────────────────────────
IDLE = 0            # At desk, unaware
REACTING = 1        # Noticed danger, frozen (reaction delay)
FLEEING_ROOM = 2    # Following BFS path to corridor doorway
WANDERING = 3       # Random walk, no goal
WALL_FOLLOWING = 4  # Blind in smoke, hand-on-wall
FOLLOWING = 5       # Herding — biased toward crowd direction
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

        # Reaction delay: 0.5-2s at 30fps (Proulx 2001)
        self._reaction_delay = int(rng.uniform(15, 60))
        self._reaction_counter = 0

        # Potency: energy/health system (Kasereka 2018)
        # Random 50-100. Decreases in smoke/fire. NO recovery. At 0 = dead.
        self.potency = rng.uniform(50.0, 100.0)

        # BFS waypoints + target
        self._waypoints = []
        self._target_exit = None

        # Random walk: current direction (cardinal)
        self._walk_dir = None

        # Exit told by guide (set before reaction finishes, used after)
        self._guide_told_exit = None

        # Wall-following hand rule
        self._wall_hand = 1 if rng.random() < 0.5 else -1

        # Stuck detection
        self._prev_pos = self.pos.copy()
        self._stuck_ticks = 0

        # For sweeper guide (Phase 2)
        self.assigned_guide = None

    # ================================================================
    # MAIN UPDATE
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

        # ── IDLE: wait for danger ──
        if self.state == IDLE:
            if (in_fire or in_smoke
                    or self._sense_danger(env, smoke, fire)
                    or self._see_others_fleeing(all_civilians)):
                self.state = REACTING
                self._reaction_counter = 0
            return

        # ── REACTING: freeze, then move ──
        if self.state == REACTING:
            self._reaction_counter += 1
            if self._reaction_counter >= self._reaction_delay:
                self._start_moving(env)
            return

        # ── Rule 5: Potency damage (no recovery) ──
        if in_fire:
            self.potency -= 3.0
        elif in_smoke:
            self.potency -= local_smoke

        if self.potency <= 0:
            self.state = DEAD
            self.vel = np.zeros(2)
            return

        # ── Vision & speed ──
        eff_vision = self._calc_vision(local_smoke)
        speed = self._calc_speed(in_fire, in_smoke)

        # ── Opportunistic: spot exit ──
        if eff_vision > 0 and self.state not in (MOVING_TO_EXIT,):
            exit_pos = self._spot_exit(env, eff_vision)
            if exit_pos is not None:
                self._set_exit_target(env, exit_pos)

        # ── Rule 4: Herding (crowd direction) ──
        if eff_vision > 0 and self.state == WANDERING:
            crowd_dir = self._sense_crowd_direction(all_civilians, eff_vision * 3)
            if crowd_dir is not None:
                self._walk_dir = crowd_dir
                self.state = FOLLOWING

        # ── State behaviors ──

        if self.state == FLEEING_ROOM:
            # Rule 2: follow BFS path to doorway
            self._do_follow_path(env, speed, fallback=WANDERING)

        elif self.state == MOVING_TO_EXIT:
            # Rule 2: follow BFS path to exit
            if self._target_exit is not None:
                dist = np.linalg.norm(self.pos - np.array(self._target_exit))
                if dist < env.cell_size:
                    self.state = ESCAPED
                    return
            self._do_follow_path(env, speed, fallback=WANDERING)

        elif self.state == FOLLOWING:
            # Rule 4: walk in crowd direction
            if self._walk_dir is not None:
                self.vel = self._walk_dir * speed
            # Re-check crowd each tick
            if eff_vision > 0:
                crowd_dir = self._sense_crowd_direction(all_civilians, eff_vision * 3)
                if crowd_dir is not None:
                    self._walk_dir = crowd_dir
                else:
                    self.state = WANDERING
                    self._walk_dir = None
                # Can we see an exit?
                exit_pos = self._spot_exit(env, eff_vision)
                if exit_pos is not None:
                    self._set_exit_target(env, exit_pos)

        elif self.state == WALL_FOLLOWING:
            # Blind in smoke: hand-on-wall
            self._do_wall_follow(env, speed)
            if eff_vision > 0 and not in_smoke:
                self.state = WANDERING
                self._walk_dir = None

        elif self.state == WANDERING:
            if in_smoke and eff_vision == 0:
                # Blind → wall-follow
                self.state = WALL_FOLLOWING
                self._do_wall_follow(env, speed)
            else:
                # Rule 1: random walk
                # Rule 3: disorientation in smoke
                self._do_random_walk(env, speed, local_smoke)

        # ── Apply movement ──
        self._move(env, speed)

    # ================================================================
    # STATE TRANSITIONS
    # ================================================================

    def _start_moving(self, env):
        """After reaction delay: if guide told us an exit, go there.
        Otherwise find doorway or start wandering."""
        # Did a guide tell us where the exit is?
        if self._guide_told_exit is not None:
            self._set_exit_target(env, self._guide_told_exit)
            self._guide_told_exit = None
            return

        doorway = self._find_doorway(env)
        if doorway is not None:
            self._waypoints = bfs_path(env, self.pos, doorway)
            self.state = FLEEING_ROOM
        else:
            self.state = WANDERING
            self._pick_random_cardinal()

    def _set_exit_target(self, env, exit_pos):
        """Set exit as goal with BFS path."""
        self._target_exit = exit_pos
        self._waypoints = bfs_path(env, self.pos, np.array(exit_pos))
        self.state = MOVING_TO_EXIT

    # ================================================================
    # MOVEMENT RULES
    # ================================================================

    def _do_random_walk(self, env, speed, smoke_density):
        """Rule 1 + Rule 3: Random walk with disorientation in smoke.
        Pick a cardinal direction, walk until wall. In smoke, random
        chance to change direction (disorientation)."""

        # Rule 3: Disorientation — smoke makes you lose your way
        if smoke_density > 0.2:
            # Higher smoke = more chance to randomly change direction
            if self._rng.random() < smoke_density * 0.3:
                self._pick_random_cardinal()

        # No direction yet? Pick one
        if self._walk_dir is None:
            self._pick_random_cardinal()

        self.vel = self._walk_dir * speed

    def _do_follow_path(self, env, speed, fallback):
        """Rule 2: Follow BFS waypoints."""
        if not self._waypoints:
            self.state = fallback
            self._walk_dir = None
            return

        target = self._waypoints[0]
        diff = target - self.pos
        dist = np.linalg.norm(diff)

        if dist < env.cell_size * 0.5:
            self._waypoints.pop(0)
            if not self._waypoints:
                self.state = fallback
                self._walk_dir = None
                return
            target = self._waypoints[0]
            diff = target - self.pos
            dist = np.linalg.norm(diff)

        if dist > 0:
            self.vel = (diff / dist) * speed

    def _do_wall_follow(self, env, speed):
        """Blind in smoke: hand-on-wall rule."""
        x, y = self.pos
        cs = env.cell_size * 0.5

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if not env.is_walkable(x + dx * cs, y + dy * cs):
                wall = np.array([dx, dy], dtype=float)
                if self._wall_hand == 1:
                    self.vel = np.array([-wall[1], wall[0]]) * speed
                else:
                    self.vel = np.array([wall[1], -wall[0]]) * speed
                return

        # No wall nearby — walk forward
        norm = np.linalg.norm(self.vel)
        if norm > 0:
            self.vel = (self.vel / norm) * speed
        else:
            self._pick_random_cardinal()
            self.vel = self._walk_dir * speed

    def _move(self, env, speed):
        """Apply velocity with wall collision."""
        x, y = self.pos
        vx, vy = self.vel

        if env.is_walkable(x + vx, y + vy):
            self.pos = np.array([x + vx, y + vy])
        elif env.is_walkable(x + vx, y):
            self.pos = np.array([x + vx, y])
        elif env.is_walkable(x, y + vy):
            self.pos = np.array([x, y + vy])
        else:
            # Blocked — pick new direction
            self._pick_random_cardinal()

        # Stuck detection
        moved = np.linalg.norm(self.pos - self._prev_pos)
        if moved < 0.5:
            self._stuck_ticks += 1
        else:
            self._stuck_ticks = 0
            self._prev_pos = self.pos.copy()

        if self._stuck_ticks >= 10:
            self._pick_random_cardinal()
            self._stuck_ticks = 0

    def _pick_random_cardinal(self):
        """Pick a random cardinal direction (up/down/left/right)."""
        dirs = [
            np.array([1.0, 0.0]),
            np.array([-1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, -1.0]),
        ]
        self._walk_dir = dirs[self._rng.integers(4)]

    # ================================================================
    # PERCEPTION
    # ================================================================

    def _calc_vision(self, smoke_density):
        if smoke_density > 0.5:
            return 0.0
        elif smoke_density > 0.2:
            return self.vision_radius * max(0.0, 1.0 - smoke_density * 2.0)
        return self.vision_radius

    def _calc_speed(self, in_fire, in_smoke):
        has_path = self.state in (FLEEING_ROOM, MOVING_TO_EXIT, FOLLOWING)
        if in_fire:
            return self.speed * 1.8
        elif in_smoke:
            return self.speed * (0.7 if has_path else 0.3)
        return self.speed

    def _see_others_fleeing(self, all_civilians):
        """Kasereka 2018: 'when they see a person who evacuates the place,
        they shall run towards the nearest exit door.'
        If a non-idle civilian is within vision range, we notice the danger."""
        for other in all_civilians:
            if other is self:
                continue
            if other.state in (IDLE, REACTING, ESCAPED, DEAD):
                continue
            # Other civilian is actively moving (fleeing/wandering/etc.)
            d = np.linalg.norm(self.pos - other.pos)
            if d < self.vision_radius * 2:  # 2 cells — can see someone running nearby
                return True
        return False

    def _sense_danger(self, env, smoke, fire):
        """Detect smoke/fire within 1 cell."""
        gr, gc = env.world_to_grid(self.pos[0], self.pos[1])
        rows, cols = smoke.density.shape
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r, c = gr + dr, gc + dc
                if 0 <= r < rows and 0 <= c < cols:
                    if smoke.density[r, c] > 0.1 or fire.density[r, c] > 0.05:
                        return True
        return False

    def _spot_exit(self, env, eff_vision):
        """Find nearest exit within vision."""
        best, best_d = None, float("inf")
        for ex, ey in env.exits:
            d = np.hypot(self.pos[0] - ex, self.pos[1] - ey)
            if d < eff_vision and d < best_d:
                best, best_d = (ex, ey), d
        return best

    def _sense_crowd_direction(self, all_civilians, radius):
        """Rule 4: Herding — average direction of nearby moving civilians.
        If 2+ nearby civilians are moving, bias toward their average direction.
        This is crowd-level herding, not 1-to-1 following (Helbing 2000)."""
        avg_dir = np.zeros(2)
        count = 0

        for other in all_civilians:
            if other is self or other.state in (IDLE, REACTING, ESCAPED, DEAD):
                continue
            d = np.linalg.norm(self.pos - other.pos)
            if d < radius and d > 0:
                other_speed = np.linalg.norm(other.vel)
                if other_speed > 0.5:  # Only count civilians actually moving
                    avg_dir += other.vel / other_speed
                    count += 1

        if count >= 2:
            norm = np.linalg.norm(avg_dir)
            if norm > 0:
                return avg_dir / norm
        return None

    # ================================================================
    # PATHFINDING
    # ================================================================

    def _find_doorway(self, env):
        """BFS to nearest corridor cell (6+ straight-line walk)."""
        from collections import deque
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
            return None

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
