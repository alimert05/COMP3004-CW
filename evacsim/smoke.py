"""
Probabilistic smoke spread with wind bias and seeded randomness.

- Smoke only spreads to cells adjacent to existing smoke
- Each candidate cell has a random chance of catching per tick
- Wind bias: smoke is 2x more likely to spread in the wind direction
- Density grows slowly over time (thin wispy → thick opaque)
- Smoke flows through corridors and gaps in walls naturally
- Fixed seed = identical pattern every run
"""

import numpy as np
from evacsim.environment import Environment, CellType


class SmokeSystem:
    def __init__(self, env: Environment, spread_rate: float, seed: int = 42,
                 wind_direction: str = "right", wind_strength: float = 1.5,
                 start_delay: int = 60):
        """
        spread_rate:    base chance per tick for a cell to catch smoke (e.g. 0.03)
        wind_direction: "up", "down", "left", "right" — dominant smoke flow
        wind_strength:  how much more likely smoke spreads in the wind direction
                        1.0 = no wind, 2.0 = twice as likely downwind
        start_delay:    ticks before smoke starts spreading (gives evacuation time)
        """
        self.env = env
        self.spread_rate = spread_rate
        self.wind_strength = wind_strength
        self.start_delay = start_delay
        self._tick = 0

        # Fixed-seed RNG
        self._rng = np.random.default_rng(seed)

        # Density grid
        self.density = np.zeros((env.rows, env.cols), dtype=float)

        # Walkable mask
        self._walkable = env.grid != CellType.WALL

        # Fire sources start at full smoke
        self._fire_cells = list(env.fire_sources)
        for r, c in self._fire_cells:
            self.density[r, c] = 1.0

        # Dynamic fire cells — updated each tick from outside
        # Every burning cell becomes a smoke source
        self._dynamic_fire_cells: list[tuple[int, int]] = []

        # Wind bias grids — each direction gets a multiplier
        # Wind "right" means smoke is more likely to spread rightward
        rows, cols = env.rows, env.cols
        self._wind_bias = {
            "up":    np.ones((rows, cols)),
            "down":  np.ones((rows, cols)),
            "left":  np.ones((rows, cols)),
            "right": np.ones((rows, cols)),
        }
        # Boost the wind direction
        if wind_direction in self._wind_bias:
            self._wind_bias[wind_direction] *= wind_strength

    def step(self) -> None:
        """Advance one tick."""
        self._tick += 1

        # During delay period, only grow existing smoke cells (no spreading)
        if self._tick <= self.start_delay:
            # Just make fire source cells stay at 1.0
            for r, c in self._fire_cells:
                self.density[r, c] = 1.0
            return

        d = self.density
        rows, cols = d.shape

        # --- Spread to new cells with directional bias ---
        has_smoke = d > 0.0

        # Check each direction separately so we can apply wind bias
        # "from above" = smoke at [r-1, c] can spread down to [r, c]
        from_above = np.zeros((rows, cols), dtype=bool)
        from_below = np.zeros((rows, cols), dtype=bool)
        from_left  = np.zeros((rows, cols), dtype=bool)
        from_right = np.zeros((rows, cols), dtype=bool)

        from_above[1:, :]  = has_smoke[:-1, :]   # cell above has smoke → spreads down
        from_below[:-1, :] = has_smoke[1:, :]     # cell below has smoke → spreads up
        from_left[:, 1:]   = has_smoke[:, :-1]    # cell left has smoke → spreads right
        from_right[:, :-1] = has_smoke[:, 1:]     # cell right has smoke → spreads left

        # Each direction rolls separately with its own wind-adjusted chance
        rolls = self._rng.random(size=(rows, cols, 4))

        catches = np.zeros((rows, cols), dtype=bool)
        catches |= from_above & (rolls[:,:,0] < self.spread_rate * self._wind_bias["down"])
        catches |= from_below & (rolls[:,:,1] < self.spread_rate * self._wind_bias["up"])
        catches |= from_left  & (rolls[:,:,2] < self.spread_rate * self._wind_bias["right"])
        catches |= from_right & (rolls[:,:,3] < self.spread_rate * self._wind_bias["left"])

        # Only walkable cells that don't already have smoke
        new_smoke = self._walkable & (~has_smoke) & catches

        # New smoke starts thin (low density)
        d[new_smoke] = self._rng.uniform(0.05, 0.15, size=int(new_smoke.sum()))

        # --- Existing smoke grows denser slowly ---
        growing = (d > 0) & (d < 1.0) & self._walkable
        growth = self._rng.uniform(0.005, 0.02, size=(rows, cols))
        d[growing] = np.minimum(1.0, d[growing] + growth[growing])

        # Fire sources always at 1.0
        for r, c in self._fire_cells:
            d[r, c] = 1.0

        # Dynamic fire cells also produce smoke
        for r, c in self._dynamic_fire_cells:
            d[r, c] = max(d[r, c], 0.8)

        self.density = d

    def update_fire_cells(self, fire_grid: np.ndarray, threshold: float = 0.3) -> None:
        """Tell smoke system which cells are currently on fire.

        Any cell in fire_grid above the threshold becomes a smoke source.
        Called each tick from the main loop so new fires produce smoke.
        """
        burning = np.argwhere(fire_grid > threshold)
        self._dynamic_fire_cells = [(int(r), int(c)) for r, c in burning]

    def get_density_at(self, x: float, y: float) -> float:
        """Return smoke density at a world position."""
        row, col = self.env.world_to_grid(x, y)
        if 0 <= row < self.env.rows and 0 <= col < self.env.cols:
            return float(self.density[row, col])
        return 0.0

    def get_density_grid(self) -> np.ndarray:
        """Return a copy of the density grid."""
        return self.density.copy()
