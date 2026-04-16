"""
Smoke spread using cellular automaton diffusion with Moore neighborhood.

Each cell holds a density [0.0 to 1.0]. Every tick, smoke flows from
high-density cells to lower-density neighbours using the Moore neighborhood
(8-connected: 4 cardinal + 4 diagonal), with diagonal neighbours attenuated
by 1/sqrt(2) to account for the longer diffusion distance. Walls block smoke.

Fire–smoke coupling: when a fire reference is provided, smoke density is
continuously sourced from fire cells (smoke at each cell is at least the
local fire density). This models real combustion — fire generates smoke
continuously — and ensures smoke follows fire as it spreads. Smoke can
still exceed fire density via its own (faster) diffusion, so smoke naturally
extends beyond the fire boundary.

References:
  - Makmul (2020) — Cellular Automaton Model for Pedestrians' Movements
    Influenced by Gaseous Hazardous Material Spreading (advection-diffusion
    with source term).
  - Zheng et al. (2014) — Cellular Automata Based Simulation for Smoke and
    Fire Spreading in Large Buildings (Moore neighborhood).
"""

import numpy as np
from evacsim.environment import Environment, CellType


# 1 / sqrt(2) — diagonal neighbour attenuation (distance is sqrt(2) cells)
_INV_SQRT2 = 1.0 / np.sqrt(2.0)


class SmokeSystem:
    def __init__(
        self,
        env: Environment,
        spread_rate: float = 0.02,
        seed: int = 42,
        fire=None,
    ):
        self.env = env
        self.spread_rate = spread_rate
        self._rng = np.random.default_rng(seed)

        self.density = np.zeros((env.rows, env.cols), dtype=float)
        # Smoke can spread on floor only — not walls, not exits
        self._smokeable = (env.grid != CellType.WALL) & (env.grid != CellType.EXIT)

        self._fire_cells = list(env.fire_sources)
        for r, c in self._fire_cells:
            self.density[r, c] = 1.0

        # Optional fire coupling. Set at construction or via attribute.
        self._fire = fire

    def step(self) -> None:
        """Diffuse smoke one tick using Moore (8-connected) neighborhood.
        If a fire reference is attached, smoke is also sourced from fire cells."""
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

        # Slight random noise so edges aren't perfectly symmetric
        noise = self._rng.uniform(0.8, 1.2, size=d.shape)
        delta = (self.spread_rate * noise) * np.maximum(0.0, max_neighbour - d)

        self.density = np.where(self._smokeable, np.minimum(1.0, d + delta), 0.0)

        # Keep initial fire source cells pinned (static sources)
        for r, c in self._fire_cells:
            self.density[r, c] = 1.0

        # Fire–smoke coupling: smoke density ≥ fire density at every cell
        # (Makmul 2020 source term — fire continuously emits smoke)
        if self._fire is not None:
            self.density = np.where(
                self._smokeable,
                np.maximum(self.density, self._fire.density),
                0.0,
            )

    def get_density_at(self, x: float, y: float) -> float:
        row, col = self.env.world_to_grid(x, y)
        if 0 <= row < self.env.rows and 0 <= col < self.env.cols:
            return float(self.density[row, col])
        return 0.0

    def get_density_grid(self) -> np.ndarray:
        return self.density.copy()
