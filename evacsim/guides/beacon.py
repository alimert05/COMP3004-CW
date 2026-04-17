"""BeaconGuide — stationary broadcast policy (L1 passive)."""
import math
import numpy as np
from collections import deque
from evacsim.environment import Environment, CellType
from evacsim.smoke import SmokeSystem
from evacsim.fire import FireSystem
from evacsim.guides.base import Guide
from evacsim.guides.common import bfs_path


class BeaconGuide(Guide):
    """
    Beacon policy: stand near an assigned exit and broadcast.

    Behaviour:
      1. On creation, assigned to an exit
      2. Uses BFS pathfinding to navigate through corridors to the exit
      3. Stands 2-3 cells in front of exit and broadcasts
      4. Any civilian within hearing range learns the exit location

    Why it helps:
      - Civilians with 1-cell vision would never find exits on their own
      - Guide's broadcast (8 cells) reaches civilians in corridors
      - Even in smoke (3-cell broadcast), much better than blind civilians
    """

    def __init__(self, x: float, y: float, speed: float,
                 env: Environment, rng: np.random.Generator,
                 assigned_exit: tuple = None):
        super().__init__(x, y, speed, env, rng)

        if assigned_exit is not None:
            self.assigned_exit = np.array(assigned_exit, dtype=float)
        else:
            self.assigned_exit = self._find_nearest_exit()

        self._beacon_pos = self._find_beacon_position()
        self._waypoints = bfs_path(env, self.pos, self._beacon_pos)
        self._in_position = False

    def _find_nearest_exit(self) -> np.ndarray:
        """Find the exit closest to spawn position."""
        best = None
        best_dist = float("inf")
        for ex, ey in self.known_exits:
            dist = np.sqrt((self.pos[0] - ex) ** 2 + (self.pos[1] - ey) ** 2)
            if dist < best_dist:
                best_dist = dist
                best = np.array([ex, ey], dtype=float)
        return best

    def _find_beacon_position(self) -> np.ndarray:
        """Stand 2 cells in front of the exit (into the building).
        'In front' = the direction where FLOOR cells are (not walls)."""
        ex, ey = self.assigned_exit
        gr, gc = self.env.world_to_grid(ex, ey)
        rows, cols = self.env.grid.shape

        # Find which direction is "inward" — check neighbors for FLOOR
        inward_dr, inward_dc = 0, 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = gr + dr, gc + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if self.env.grid[nr, nc] == CellType.FLOOR:
                    inward_dr, inward_dc = dr, dc
                    break

        # Step 2 cells inward
        target_r = gr + inward_dr * 2
        target_c = gc + inward_dc * 2
        if (0 <= target_r < rows and 0 <= target_c < cols
                and self.env.grid[target_r, target_c] != CellType.WALL):
            return np.array(self.env.grid_to_world(target_r, target_c), dtype=float)

        # Try 1 cell inward
        target_r = gr + inward_dr
        target_c = gc + inward_dc
        if (0 <= target_r < rows and 0 <= target_c < cols
                and self.env.grid[target_r, target_c] != CellType.WALL):
            return np.array(self.env.grid_to_world(target_r, target_c), dtype=float)

        # Fallback: stand at exit
        return self.assigned_exit.copy()

    def update(
        self, env: Environment, smoke: SmokeSystem, fire: FireSystem, civilians: list
    ) -> None:
        """Move to beacon position via BFS path, then stand and broadcast."""
        super().update(env, smoke, fire, civilians)
        if not self.alive:
            return

        # Self-evacuation overrides everything
        if self._do_self_evacuation(env):
            return

        # --- Phase 1: Follow path to beacon position ---
        if not self._in_position:
            if not self._waypoints:
                # Path done — lock in position
                self._in_position = True
                self.vel = np.array([0.0, 0.0])
            else:
                self._follow_path(env)
            return

        # --- Phase 2: Probabilistic broadcast (sound-based) ---
        # Hearing probability decreases with distance and smoke/noise.
        # Sound penetrates walls (no LOS check) but attenuates.
        # Based on acoustic propagation in fire conditions.
        max_range = self.broadcast_radius  # 100 units (10 cells / ~30m)

        from evacsim.civilian import (
            MOVING_TO_EXIT,
            ESCAPED,
            DEAD,
            IDLE,
            REACTING,
            FOLLOWING_GUIDE,
        )

        exit_pos = (self.assigned_exit[0], self.assigned_exit[1])

        for civ in civilians:
            if civ.state in (MOVING_TO_EXIT, ESCAPED, DEAD, FOLLOWING_GUIDE):
                continue

            dist = np.linalg.norm(civ.pos - self.pos)
            if dist >= max_range:
                continue

            # Distance factor: linear falloff (closer = easier to hear)
            distance_factor = 1.0 - (dist / max_range)

            # Listener-side: smoke at civilian's position (irritation, disorientation)
            civ_smoke = smoke.get_density_at(civ.pos[0], civ.pos[1])
            listener_factor = max(0.3, 1.0 - civ_smoke * 0.5)

            # Source-side: smoke at guide's position (ambient noise from fire)
            # Guide has mask so less affected, but noise still reduces effectiveness
            guide_smoke = smoke.get_density_at(self.pos[0], self.pos[1])
            source_factor = max(0.5, 1.0 - guide_smoke * 0.3)

            # Final hearing probability this tick
            hear_chance = distance_factor * listener_factor * source_factor

            # Roll the dice
            if self._rng.random() > hear_chance:
                continue  # Didn't hear this tick — try again next tick

            # Civilian heard the guide!
            if civ.state == IDLE:
                civ.state = REACTING
                civ._reaction_counter = 0
                civ._reaction_delay = min(civ._reaction_delay, 10)
                civ._guide_told_exit = exit_pos
            elif civ.state == REACTING:
                civ._reaction_delay = min(civ._reaction_delay, 5)
                civ._guide_told_exit = exit_pos
            else:
                civ._set_exit_target(env, exit_pos)


def spawn_beacon_guides(
    env: Environment, n_guides: int, speed: float = 2.0, seed: int = 77
) -> list[BeaconGuide]:
    """Spawn beacon guides at exits. Simple round-robin assignment.
    Each guide stands 2-3 cells from its assigned exit and broadcasts.
    Overlap on clustered-exit maps is a realistic Beacon limitation."""
    rng = np.random.default_rng(seed)
    exits = list(env.exits)

    guides = []
    for i in range(n_guides):
        assigned_exit = exits[i % len(exits)]
        ex, ey = assigned_exit
        guide = BeaconGuide(ex, ey, speed, env, rng, assigned_exit=assigned_exit)
        guides.append(guide)

    return guides
