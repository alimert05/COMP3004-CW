"""
Civilian agent — dumb wandering with panic reaction.

Civilians are NOT smart. They:
  - Wander randomly, bouncing off walls
  - If standing IN smoke or fire, they panic (move faster)
  - On first contact with danger, they pick a retreat direction and KEEP going
  - They do NOT sense danger ahead or steer around it
  - Only guides (later) will give them intelligent direction
"""

import numpy as np
from evacsim.environment import Environment
from evacsim.smoke import SmokeSystem
from evacsim.fire import FireSystem


class Civilian:
    def __init__(self, x: float, y: float, speed: float, rng: np.random.Generator):
        # Position in world coordinates (floats)
        self.pos = np.array([x, y], dtype=float)

        # Pick a random initial direction
        angle = rng.uniform(0, 2 * np.pi)
        self.vel = np.array([np.cos(angle), np.sin(angle)]) * speed

        self.speed = speed
        self._rng = rng

        # Stuck detection
        self._prev_pos = self.pos.copy()
        self._stuck_ticks = 0

        # Remember last safe position (where there was no smoke/fire)
        self._last_safe_pos = self.pos.copy()

        # Are we currently panicking? (keeps us moving in same direction)
        self._panicking = False

    def update(self, env: Environment, smoke: SmokeSystem, fire: FireSystem) -> None:
        """Move one tick: wander randomly, panic if in smoke/fire."""
        x, y = self.pos

        # --- Check what we're standing in ---
        local_smoke = smoke.get_density_at(x, y)
        local_fire = fire.get_density_at(x, y)

        in_danger = local_fire > 0.1 or local_smoke > 0.3

        # --- Decide speed ---
        if local_fire > 0.1:
            current_speed = self.speed * 2.0
        elif local_smoke > 0.3:
            current_speed = self.speed * 1.5
        else:
            current_speed = self.speed

        # --- Panic logic ---
        if in_danger and not self._panicking:
            # Just entered danger! Pick retreat direction ONCE
            self._panicking = True
            self._set_retreat_direction(current_speed)

        elif not in_danger and self._panicking:
            # We escaped! Stop panicking, go back to wandering
            self._panicking = False

        elif not in_danger:
            # Safe — remember this position, wander normally
            self._last_safe_pos = self.pos.copy()
            if self._rng.random() < 0.02:
                self._pick_new_direction()

        # If panicking, keep current direction but scale to panic speed
        if self._panicking:
            speed_now = np.linalg.norm(self.vel)
            if speed_now > 0:
                self.vel = self.vel / speed_now * current_speed

        # --- Move (try multiple options so we don't get stuck on walls) ---
        vx, vy = self.vel
        moved = False

        # Try 1: full move
        if env.is_walkable(x + vx, y + vy):
            self.pos = np.array([x + vx, y + vy])
            moved = True
        # Try 2: horizontal only (slide along wall)
        elif env.is_walkable(x + vx, y):
            self.pos = np.array([x + vx, y])
            moved = True
        # Try 3: vertical only (slide along wall)
        elif env.is_walkable(x, y + vy):
            self.pos = np.array([x, y + vy])
            moved = True

        # If all failed — wall ahead, pick a new escape direction
        if not moved:
            if self._panicking:
                # Try perpendicular to current direction (slide along wall)
                perp = np.array([-self.vel[1], self.vel[0]])
                if self._rng.random() < 0.5:
                    perp = -perp
                norm = np.linalg.norm(perp)
                if norm > 0:
                    self.vel = perp / norm * current_speed
            else:
                self._pick_new_direction()

        # --- Stuck detection ---
        dist_moved = np.linalg.norm(self.pos - self._prev_pos)
        if dist_moved < 1.0:
            self._stuck_ticks += 1
        else:
            self._stuck_ticks = 0
            self._prev_pos = self.pos.copy()

        # If stuck for 5 ticks, force a new direction
        if self._stuck_ticks >= 5:
            self._pick_new_direction()
            # Scale to current speed
            self.vel = self.vel / np.linalg.norm(self.vel) * current_speed
            self._stuck_ticks = 0

    def _set_retreat_direction(self, current_speed: float) -> None:
        """Pick a retreat direction — toward last safe position."""
        diff = self._last_safe_pos - self.pos
        dist = np.linalg.norm(diff)

        if dist > 1.0:
            # Head back toward where we were safe
            direction = diff / dist
            # Add ±30° randomness
            noise_angle = self._rng.uniform(-0.5, 0.5)
            cos_n, sin_n = np.cos(noise_angle), np.sin(noise_angle)
            direction = np.array([
                direction[0] * cos_n - direction[1] * sin_n,
                direction[0] * sin_n + direction[1] * cos_n,
            ])
            self.vel = direction * current_speed
        else:
            # Safe pos already smoky — just reverse current direction
            self.vel = -self.vel
            speed_now = np.linalg.norm(self.vel)
            if speed_now > 0:
                self.vel = self.vel / speed_now * current_speed

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
