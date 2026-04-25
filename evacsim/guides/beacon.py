"""BeaconGuide: stationary broadcast policy.

The guide walks to a position two to three cells in front of an
assigned exit and broadcasts from there. Each tick, every civilian
within hearing range rolls against a probabilistic hearing model
that combines distance with listener-side and source-side smoke. On
success the civilian either reacts sooner (if IDLE or REACTING) or
has its exit target set directly (if already moving).
"""
import math
import numpy as np
from collections import deque
from evacsim.environment import Environment, CellType
from evacsim.smoke import SmokeSystem
from evacsim.fire import FireSystem
from evacsim.guides.base import Guide
from evacsim.guides.common import bfs_path


class BeaconGuide(Guide):
    """Stationary guide that broadcasts the location of an assigned exit.

    Behaviour:
        1. On creation, assigned to a specific exit (round-robin in
           spawn_beacon_guides).
        2. Plans a BFS path from spawn to the beacon position.
        3. Stands two to three cells in front of the exit and broadcasts.
        4. Every tick, civilians within hearing range roll independently
           against a probabilistic hearing model. On success they receive
           the exit location.

    Self-evacuation (inherited from Guide) overrides broadcasting when
    the guide's health drops below the threshold.
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
        """Return the exit closest to the current spawn position."""
        best = None
        best_dist = float("inf")
        for ex, ey in self.known_exits:
            dist = np.sqrt((self.pos[0] - ex) ** 2 + (self.pos[1] - ey) ** 2)
            if dist < best_dist:
                best_dist = dist
                best = np.array([ex, ey], dtype=float)
        return best

    def _find_beacon_position(self) -> np.ndarray:
        """Return a world position two cells inward of the assigned exit.

        The inward direction is the cardinal neighbour that is a FLOOR
        cell (not a wall). If the two-cell offset lands on a wall, the
        method retries at one cell inward, then falls back to the exit
        cell itself.
        """
        ex, ey = self.assigned_exit
        gr, gc = self.env.world_to_grid(ex, ey)
        rows, cols = self.env.grid.shape

        # Identify the inward direction by looking for a FLOOR neighbour.
        inward_dr, inward_dc = 0, 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = gr + dr, gc + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if self.env.grid[nr, nc] == CellType.FLOOR:
                    inward_dr, inward_dc = dr, dc
                    break

        # Step two cells inward.
        target_r = gr + inward_dr * 2
        target_c = gc + inward_dc * 2
        if (0 <= target_r < rows and 0 <= target_c < cols
                and self.env.grid[target_r, target_c] != CellType.WALL):
            return np.array(self.env.grid_to_world(target_r, target_c), dtype=float)

        # Try one cell inward.
        target_r = gr + inward_dr
        target_c = gc + inward_dc
        if (0 <= target_r < rows and 0 <= target_c < cols
                and self.env.grid[target_r, target_c] != CellType.WALL):
            return np.array(self.env.grid_to_world(target_r, target_c), dtype=float)

        # Fallback: stand at the exit cell itself.
        return self.assigned_exit.copy()

    def update(
        self, env: Environment, smoke: SmokeSystem, fire: FireSystem, civilians: list
    ) -> None:
        """Walk to the beacon position via BFS, then broadcast each tick."""
        super().update(env, smoke, fire, civilians)
        if not self.alive:
            return

        # Self-evacuation overrides broadcasting.
        if self._do_self_evacuation(env):
            return

        # Phase 1: follow the BFS path to the beacon position.
        if not self._in_position:
            if not self._waypoints:
                # Path complete, lock in position.
                self._in_position = True
                self.vel = np.array([0.0, 0.0])
            else:
                self._follow_path(env)
            return

        # Phase 2: probabilistic broadcast.
        # Sound penetrates walls (no line-of-sight check) but attenuates
        # with distance and with smoke at both endpoints.
        max_range = self.broadcast_radius  # 100 units, ten cells.

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

            # Distance factor: linear falloff with a floor of zero at max_range.
            distance_factor = 1.0 - (dist / max_range)

            # Listener-side smoke factor (irritation, disorientation at the civilian).
            civ_smoke = smoke.get_density_at(civ.pos[0], civ.pos[1])
            listener_factor = max(0.3, 1.0 - civ_smoke * 0.5)

            # Source-side smoke factor. The guide's mask reduces the effect
            # but ambient noise from fire still degrades broadcast quality.
            guide_smoke = smoke.get_density_at(self.pos[0], self.pos[1])
            source_factor = max(0.5, 1.0 - guide_smoke * 0.3)

            hear_chance = distance_factor * listener_factor * source_factor

            if self._rng.random() > hear_chance:
                continue  # Roll failed this tick.

            # The civilian heard the broadcast.
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
    """Spawn beacon guides at exits using round-robin assignment.

    Each guide is placed at one exit; if there are more guides than
    exits, the assignment wraps around. Overlapping beacons on
    clustered-exit maps is left as a realistic limitation of the policy.
    """
    rng = np.random.default_rng(seed)
    exits = list(env.exits)

    guides = []
    for i in range(n_guides):
        assigned_exit = exits[i % len(exits)]
        ex, ey = assigned_exit
        guide = BeaconGuide(ex, ey, speed, env, rng, assigned_exit=assigned_exit)
        guides.append(guide)

    return guides
