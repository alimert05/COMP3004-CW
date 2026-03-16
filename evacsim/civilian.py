"""
Civilian agent — simple wandering for now.

Each civilian has a position (world coordinates) and a velocity.
They walk around randomly, bouncing off walls.
Later we'll add: states, steering forces, smoke damage, exit seeking.
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

    def update(self, env: Environment, smoke: SmokeSystem, fire: FireSystem) -> None:
        """Move one tick: avoid smoke/fire, bounce off walls."""
        x, y = self.pos
        d = env.cell_size  # sample 1 cell away in each direction

        # --- Danger gradient ---
        # Sample smoke in 4 directions, compute which way has LESS smoke
        s_right = smoke.get_density_at(x + d, y)
        s_left  = smoke.get_density_at(x - d, y)
        s_down  = smoke.get_density_at(x, y + d)
        s_up    = smoke.get_density_at(x, y - d)

        # Same for fire (weighted 3x heavier — fire is scarier)
        f_right = fire.get_density_at(x + d, y) * 3.0
        f_left  = fire.get_density_at(x - d, y) * 3.0
        f_down  = fire.get_density_at(x, y + d) * 3.0
        f_up    = fire.get_density_at(x, y - d) * 3.0

        # Gradient points AWAY from danger (high→low)
        gx = (s_left + f_left) - (s_right + f_right)
        gy = (s_up + f_up) - (s_down + f_down)

        # --- Speed boost when in danger ---
        local_smoke = smoke.get_density_at(x, y)
        local_fire = fire.get_density_at(x, y)

        if local_fire > 0.1:
            current_speed = self.speed * 2.0
        elif local_smoke > 0.3:
            current_speed = self.speed * 1.5
        else:
            current_speed = self.speed

        # --- Apply avoidance force ---
        danger = abs(gx) + abs(gy)
        if danger > 0.01:
            # Steer away from danger
            avoid = np.array([gx, gy])
            avoid = avoid / np.linalg.norm(avoid)  # normalize to unit vector
            # Blend: 70% avoidance + 30% current direction
            self.vel = 0.7 * avoid * current_speed + 0.3 * self.vel
            # Re-normalize to current speed
            speed_now = np.linalg.norm(self.vel)
            if speed_now > 0:
                self.vel = self.vel / speed_now * current_speed
        else:
            # No danger nearby — scale velocity to current speed
            speed_now = np.linalg.norm(self.vel)
            if speed_now > 0:
                self.vel = self.vel / speed_now * current_speed

        # --- Move ---
        new_pos = self.pos + self.vel
        nx, ny = new_pos

        if env.is_walkable(nx, ny):
            self.pos = new_pos
        else:
            # Hit a wall — pick a new random direction
            self._pick_new_direction()

        # 2% chance to randomly change direction (looks more natural)
        if danger < 0.01 and self._rng.random() < 0.02:
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
