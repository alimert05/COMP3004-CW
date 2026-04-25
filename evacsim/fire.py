"""Fire spread via cellular-automaton diffusion on a Moore neighbourhood.

Same CA structure as the smoke system but with a slower default spread
rate. Fire sources are held at 1.0 and propagate to neighbouring cells
using the Moore (8-connected) neighbourhood. Cardinal neighbours
contribute at full weight; diagonal neighbours are attenuated by
1/sqrt(2). Walls do not burn.
"""

import numpy as np
from evacsim.environment import Environment, CellType


# 1 / sqrt(2): diagonal neighbour attenuation, since diagonal distance
# is sqrt(2) cells.
_INV_SQRT2 = 1.0 / np.sqrt(2.0)


class FireSystem:
    def __init__(self, env: Environment, spread_rate: float = 0.005, seed: int = 123):
        self.env = env
        self.spread_rate = spread_rate
        self._rng = np.random.default_rng(seed)

        self.density = np.zeros((env.rows, env.cols), dtype=float)
        # Fire can spread on floor cells only, not on walls or exits.
        self._burnable = (env.grid != CellType.WALL) & (env.grid != CellType.EXIT)

        self._fire_cells = list(env.fire_sources)
        for r, c in self._fire_cells:
            self.density[r, c] = 1.0

    def step(self) -> None:
        """Advance one diffusion tick on the Moore neighbourhood."""
        d = self.density

        padded = np.pad(d, 1, mode="constant", constant_values=0.0)

        # Cardinal neighbours (up, down, left, right) at full weight.
        cardinals = np.maximum(
            np.maximum(padded[0:-2, 1:-1], padded[2:, 1:-1]),
            np.maximum(padded[1:-1, 0:-2], padded[1:-1, 2:]),
        )

        # Diagonal neighbours, attenuated by 1/sqrt(2).
        diagonals = np.maximum(
            np.maximum(padded[0:-2, 0:-2], padded[0:-2, 2:]),
            np.maximum(padded[2:, 0:-2], padded[2:, 2:]),
        ) * _INV_SQRT2

        max_neighbour = np.maximum(cardinals, diagonals)

        # Per-cell multiplicative noise so the front edge is not
        # perfectly symmetric.
        noise = self._rng.uniform(0.8, 1.2, size=d.shape)
        delta = (self.spread_rate * noise) * np.maximum(0.0, max_neighbour - d)

        self.density = np.where(self._burnable, np.minimum(1.0, d + delta), 0.0)

        # Pin source cells to 1.0 (static sources).
        for r, c in self._fire_cells:
            self.density[r, c] = 1.0

    def get_density_at(self, x: float, y: float) -> float:
        """Return fire density at a world position; 0.0 if out of bounds."""
        row, col = self.env.world_to_grid(x, y)
        if 0 <= row < self.env.rows and 0 <= col < self.env.cols:
            return float(self.density[row, col])
        return 0.0

    def get_density_grid(self) -> np.ndarray:
        """Return a copy of the fire density grid."""
        return self.density.copy()
