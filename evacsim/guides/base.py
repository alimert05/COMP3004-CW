"""Base Guide class — shared state, health, self-evacuation."""
import math
import numpy as np
from collections import deque
from evacsim.environment import Environment, CellType
from evacsim.smoke import SmokeSystem
from evacsim.fire import FireSystem
from evacsim.guides.common import bfs_path


class Guide:
    """Base class for all guide types."""

    def __init__(
        self,
        x: float,
        y: float,
        speed: float,
        env: Environment,
        rng: np.random.Generator,
    ):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.array([0.0, 0.0])
        self.speed = speed
        self.env = env
        self._rng = rng

        # Guides know ALL exit locations (trained, have map)
        self.known_exits = list(env.exits)

        # Beacon broadcast radius (sound-based: megaphone/whistle)
        # 100 units = 10 cells = ~30m at 3m/cell scale
        self.broadcast_radius = 100.0  # 10 cells

        # Guides are alive unless fire kills them (they have gas masks for smoke)
        self.alive = True
        self._fire_damage = 0.0
        self._max_fire_damage = 300.0  # Much tougher than civilians

        # Also take smoke damage (less than civilians, but still accumulates)
        self._smoke_damage = 0.0
        self._max_smoke_damage = 250.0

        # Total health = fire + smoke combined
        # When health drops below threshold, guide abandons mission and escapes
        self._health_threshold = 0.6  # 60% — start escaping at 40% health remaining

        # Self-evacuation state
        self._self_evacuating = False
        self._escape_waypoints = []
        self.escaped = False  # True if guide successfully reached an exit

        # Which exit this guide is assigned to
        self.assigned_exit = None

    @property
    def health(self) -> float:
        """Current health as 0.0-1.0 ratio. 1.0 = full health, 0.0 = dead."""
        fire_ratio = self._fire_damage / self._max_fire_damage
        smoke_ratio = self._smoke_damage / self._max_smoke_damage
        # Worst of the two damage types determines health
        return max(0.0, 1.0 - max(fire_ratio, smoke_ratio))

    def update(
        self, env: Environment, smoke: SmokeSystem, fire: FireSystem, civilians: list
    ) -> None:
        """Base update — handles damage and self-evacuation.
        Override in subclasses but call super().update() first."""
        if not self.alive:
            return

        # Store grids for danger-aware BFS
        self._fire_grid = fire.density
        self._smoke_grid = smoke.density

        # Store grid position for dynamic max_group danger sampling
        self._last_grid_pos = env.world_to_grid(self.pos[0], self.pos[1])

        # Check fire damage (guides resist smoke but not fire)
        local_fire = fire.get_density_at(self.pos[0], self.pos[1])
        local_smoke = smoke.get_density_at(self.pos[0], self.pos[1])

        if local_fire > 0.3:
            self._fire_damage += 2.0
        if local_smoke > 0.3:
            self._smoke_damage += local_smoke * 0.5  # Half damage vs civilians

        # Check for death
        if (
            self._fire_damage >= self._max_fire_damage
            or self._smoke_damage >= self._max_smoke_damage
        ):
            self.alive = False
            return

        # Check health threshold — start self-evacuating
        if not self._self_evacuating and self.health < self._health_threshold:
            self._start_self_evacuation(env)

    def _start_self_evacuation(self, env: Environment) -> None:
        """Guide's health is critical — abandon mission, head to nearest exit."""
        self._self_evacuating = True

        # Find nearest exit
        best_exit = None
        best_dist = float("inf")
        for ex, ey in self.known_exits:
            d = np.sqrt((self.pos[0] - ex) ** 2 + (self.pos[1] - ey) ** 2)
            if d < best_dist:
                best_dist = d
                best_exit = np.array([ex, ey], dtype=float)

        if best_exit is not None:
            fg = getattr(self, "_fire_grid", None)
            self._escape_waypoints = bfs_path(env, self.pos, best_exit, fire_grid=fg)

    def _do_self_evacuation(self, env: Environment) -> bool:
        """Follow escape path to exit. Returns True if currently self-evacuating."""
        if not self._self_evacuating:
            return False

        # Check if reached exit
        for ex, ey in self.known_exits:
            dist = np.sqrt((self.pos[0] - ex) ** 2 + (self.pos[1] - ey) ** 2)
            if dist < env.cell_size:
                self.alive = False  # Guide escaped (remove from simulation)
                self.escaped = True  # Distinguish escaped from killed
                return True

        # Follow escape waypoints
        if self._escape_waypoints:
            target = self._escape_waypoints[0]
            diff = target - self.pos
            dist = np.linalg.norm(diff)

            if dist < env.cell_size * 0.5:
                self._escape_waypoints.pop(0)
                if self._escape_waypoints:
                    target = self._escape_waypoints[0]
                    diff = target - self.pos
                    dist = np.linalg.norm(diff)

            if dist > 0:
                direction = diff / dist
                self.vel = direction * self.speed * 1.5  # Sprint to exit

            # Move
            vx, vy = self.vel
            x, y = self.pos
            if env.is_walkable(x + vx, y + vy):
                self.pos = np.array([x + vx, y + vy])
            elif env.is_walkable(x + vx, y):
                self.pos = np.array([x + vx, y])
            elif env.is_walkable(x, y + vy):
                self.pos = np.array([x, y + vy])

        return True

    def get_broadcast_radius(self, smoke: SmokeSystem) -> float:
        """Broadcast radius reduced by smoke but still much better than civilian vision.
        Even in heavy smoke, guide can be heard from 3 cells away."""
        local_smoke = smoke.get_density_at(self.pos[0], self.pos[1])
        # Minimum 30 units (3 cells) even in full smoke
        return max(30.0, self.broadcast_radius * (1.0 - local_smoke * 0.5))

    def get_visual_recruit_radius(self, smoke: SmokeSystem) -> float:
        """Visual recruit range based on Jin (1978) visibility model.
        Guide is a light-emitting object (hi-vis gear, flashlight).
        Guide wears breathing apparatus — smoke reduces vision by 60%
        of what it does to unprotected civilians (mask prevents eye
        irritation, Jin 1978 showed irritation is the main vision reducer).
        Used by Sweeper and Router (physical contact, not broadcast)."""
        local_smoke = smoke.get_density_at(self.pos[0], self.pos[1])
        base_range = 70.0  # 7 cells in clear air (~20m at 3m/cell)
        # Mask reduces smoke's effect by 40% (only particle obscuration remains)
        smoke_effect = local_smoke * 0.6
        return base_range * max(0.15, 1.0 - smoke_effect)

    def _has_line_of_sight(self, target_pos, env) -> bool:
        """Ray-cast: can we see the target without a wall blocking?
        Used for visual recruit — Sweeper/Router can't recruit through walls."""
        diff = target_pos - self.pos
        dist = np.linalg.norm(diff)
        if dist < env.cell_size:
            return True
        direction = diff / dist
        steps = int(dist / env.cell_size) + 1
        for i in range(1, min(steps + 1, 25)):
            cx = self.pos[0] + direction[0] * env.cell_size * i
            cy = self.pos[1] + direction[1] * env.cell_size * i
            if not env.is_walkable(cx, cy):
                return False
        return True

    def _follow_path(self, env: Environment) -> None:
        """Follow the pre-computed waypoint path.
        Checks ahead for fire AND heavy smoke — reroutes if dangerous.
        Guides resist smoke but their civilian group does not."""
        if not self._waypoints:
            self.vel = np.array([0.0, 0.0])
            return

        # Check next 10 waypoints for fire or smoke
        fg = getattr(self, "_fire_grid", None)
        sg = getattr(self, "_smoke_grid", None)
        # Lower smoke threshold when carrying civilians (they're vulnerable)
        has_group = hasattr(self, "_group") and len(self._group) > 0
        smoke_threshold = 0.3 if has_group else 0.6

        if fg is not None or sg is not None:
            needs_reroute = False
            for wp in self._waypoints[:10]:
                wr, wc = env.world_to_grid(wp[0], wp[1])
                if fg is not None and 0 <= wr < fg.shape[0] and 0 <= wc < fg.shape[1]:
                    if fg[wr, wc] > 0.2:
                        needs_reroute = True
                        break
                if sg is not None and 0 <= wr < sg.shape[0] and 0 <= wc < sg.shape[1]:
                    if sg[wr, wc] > smoke_threshold:
                        needs_reroute = True
                        break
            if needs_reroute:
                # Combine fire + smoke into danger grid for BFS avoidance
                # When carrying a group: smoke is MORE dangerous (civilians vulnerable)
                danger = fg if fg is not None else np.zeros_like(env.grid, dtype=float)
                if sg is not None:
                    smoke_weight = 1.0 if has_group else 0.5
                    danger = np.maximum(danger, sg * smoke_weight)
                final = self._waypoints[-1]
                self._waypoints = bfs_path(env, self.pos, final, fire_grid=danger)
                if not self._waypoints:
                    # No safe path — try without smoke avoidance as fallback
                    self._waypoints = bfs_path(env, self.pos, final, fire_grid=fg)
                if not self._waypoints:
                    self.vel = np.array([0.0, 0.0])
                    return

        target = self._waypoints[0]

        diff = target - self.pos
        dist = np.linalg.norm(diff)

        if dist < self.env.cell_size * 0.5:
            self._waypoints.pop(0)
            if not self._waypoints:
                self.vel = np.array([0.0, 0.0])
                return
            target = self._waypoints[0]
            diff = target - self.pos
            dist = np.linalg.norm(diff)

        if dist > 0:
            direction = diff / dist
            # Slow down when carrying a group (pace with civilians)
            if has_group:
                move_speed = self.speed * 0.75  # ~1.5 — close to civilian speed
            else:
                move_speed = self.speed
            self.vel = direction * move_speed

        # Move
        vx, vy = self.vel
        x, y = self.pos
        if env.is_walkable(x + vx, y + vy):
            self.pos = np.array([x + vx, y + vy])
        elif env.is_walkable(x + vx, y):
            self.pos = np.array([x + vx, y])
        elif env.is_walkable(x, y + vy):
            self.pos = np.array([x, y + vy])
