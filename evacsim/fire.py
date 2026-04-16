"""
Fire spread using cellular automaton diffusion with Moore neighborhood.

Same CA approach as smoke but slower spread rate. Fire sources stay at 1.0
and propagate to neighbouring cells via the Moore (8-connected) neighborhood.
Diagonal neighbours attenuate by 1/sqrt(2) to account for distance.
Walls do not burn.

References:
  - Zheng et al. (2014) — Cellular Automata Based Simulation for Smoke and
    Fire Spreading in Large Buildings.
  - Kasereka et al. (2018) — Agent-Based Modelling and Simulation for
    evacuation of people from a building in case of fire.
"""

import numpy as np
from evacsim.environment import Environment, CellType


# 1 / sqrt(2) — diagonal neighbour attenuation (distance is sqrt(2) cells)
_INV_SQRT2 = 1.0 / np.sqrt(2.0)


class FireSystem:
    def __init__(self, env: Environment, spread_rate: float = 0.005, seed: int = 123):
        self.env = env
        self.spread_rate = spread_rate
        self._rng = np.random.default_rng(seed)

        self.density = np.zeros((env.rows, env.cols), dtype=float)
        # Fire can spread on floor only — not walls, not exits
        self._burnable = (env.grid != CellType.WALL) & (env.grid != CellType.EXIT)

        self._fire_cells = list(env.fire_sources)
        for r, c in self._fire_cells:
            self.density[r, c] = 1.0

    def step(self) -> None:
        """Diffuse fire one tick using Moore (8-connected) neighborhood."""
        d = self.density

        padded = np.pad(d, 1, mode="constant", constant_values=0.0)

        # Cardinal neighbours (up, down, left, right) — full weight
        cardinals = np.maximum(
            np.maximum(padded[0:-2, 1:-1], padded[2:, 1:-1]),
            np.maximum(padded[1:-1, 0:-2], padded[1:-1, 2:]),
        )

        # Diagonal neighbours — attenuated by 1/sqrt(2) for longer distance
        diagonals = np.maximum(
            np.maximum(padded[0:-2, 0:-2], padded[0:-2, 2:]),
            np.maximum(padded[2:, 0:-2], padded[2:, 2:]),
        ) * _INV_SQRT2

        max_neighbour = np.maximum(cardinals, diagonals)

        noise = self._rng.uniform(0.8, 1.2, size=d.shape)
        delta = (self.spread_rate * noise) * np.maximum(0.0, max_neighbour - d)

        self.density = np.where(self._burnable, np.minimum(1.0, d + delta), 0.0)

        for r, c in self._fire_cells:
            self.density[r, c] = 1.0

    def get_density_at(self, x: float, y: float) -> float:
        """Get fire density at a world position."""
        row, col = self.env.world_to_grid(x, y)
        if 0 <= row < self.env.rows and 0 <= col < self.env.cols:
            return float(self.density[row, col])
        return 0.0

    def get_density_grid(self) -> np.ndarray:
        return self.density.copy()
