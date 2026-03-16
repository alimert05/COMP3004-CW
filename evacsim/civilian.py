"""
Civilian agent — simple wandering for now.

Each civilian has a position (world coordinates) and a velocity.
They walk around randomly, bouncing off walls.
Later we'll add: states, steering forces, smoke damage, exit seeking.
"""

import numpy as np
from evacsim.environment import Environment


class Civilian:
    def __init__(self, x: float, y: float, speed: float, rng: np.random.Generator):
        # Position in world coordinates (floats)
        self.pos = np.array([x, y], dtype=float)

        # Pick a random initial direction
        angle = rng.uniform(0, 2 * np.pi)
        self.vel = np.array([np.cos(angle), np.sin(angle)]) * speed

        self.speed = speed
        self._rng = rng

    def update(self, env: Environment) -> None:
        """Move one tick: walk forward, bounce off walls."""
        new_pos = self.pos + self.vel
        nx, ny = new_pos

        if env.is_walkable(nx, ny):
            self.pos = new_pos
        else:
            # Hit a wall — pick a new random direction
            self._pick_new_direction()

        # 2% chance to randomly change direction (looks more natural)
        if self._rng.random() < 0.02:
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
