"""
Smoke spread using cellular automaton diffusion.

Each cell holds a density [0.0 to 1.0].
Every tick, smoke flows from high-density cells to lower-density neighbours.
Walls block smoke. Fire sources stay at 1.0.
Simple, predictable, easy to tune.
"""

import numpy as np
from evacsim.environment import Environment, CellType


class SmokeSystem:
    def __init__(self, env: Environment, spread_rate: float = 0.02, seed: int = 42):
        self.env = env
        self.spread_rate = spread_rate
        self._rng = np.random.default_rng(seed)

        self.density = np.zeros((env.rows, env.cols), dtype=float)
        # Smoke can spread on floor only — not walls, not exits
        self._smokeable = (env.grid != CellType.WALL) & (env.grid != CellType.EXIT)

        self._fire_cells = list(env.fire_sources)
        for r, c in self._fire_cells:
            self.density[r, c] = 1.0

    def step(self) -> None:
        """Diffuse smoke one tick: each cell moves toward its highest neighbour."""
        d = self.density

        padded = np.pad(d, 1, mode="constant", constant_values=0.0)

        max_neighbour = np.maximum(
            np.maximum(padded[0:-2, 1:-1], padded[2:, 1:-1]),
            np.maximum(padded[1:-1, 0:-2], padded[1:-1, 2:]),
        )

        # Add slight random noise so edges aren't perfectly symmetric
        noise = self._rng.uniform(0.8, 1.2, size=d.shape)
        delta = (self.spread_rate * noise) * np.maximum(0.0, max_neighbour - d)

        self.density = np.where(self._smokeable, np.minimum(1.0, d + delta), 0.0)

        for r, c in self._fire_cells:
            self.density[r, c] = 1.0

    def get_density_at(self, x: float, y: float) -> float:
        row, col = self.env.world_to_grid(x, y)
        if 0 <= row < self.env.rows and 0 <= col < self.env.cols:
            return float(self.density[row, col])
        return 0.0

    def get_density_grid(self) -> np.ndarray:
        return self.density.copy()
