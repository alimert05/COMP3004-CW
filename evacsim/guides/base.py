"""Base Guide class.

Provides shared state, damage tracking, line-of-sight checks, path
following, and self-evacuation behaviour used by every guide policy.
Subclasses (BeaconGuide, SweeperGuide, RouterGuide) override update()
to implement policy-specific behaviour but call super().update() first
so damage and health checks remain centralised.
"""
import numpy as np
from evacsim.environment import Environment
from evacsim.smoke import SmokeSystem
from evacsim.fire import FireSystem
from evacsim.civilian import (
    MOVING_TO_EXIT,
    ESCAPED,
    DEAD,
    IDLE,
    REACTING,
    FOLLOWING_GUIDE,
    QUEUED,
    WANDERING,
)
from evacsim.guides.common import bfs_path


class Guide:
    """Common base for all guide policies.

    Maintains position, velocity, known exit list, fire and smoke damage
    pools, and the self-evacuation flag. Provides perception helpers
    (broadcast radius, visual recruit radius, line-of-sight raycast) and
    a generic waypoint-following movement routine that reroutes around
    fire or heavy smoke when needed.
    """

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

        # Guides know all exit locations (trained, have a map of the building).
        self.known_exits = list(env.exits)

        # Beacon broadcast radius (sound based, e.g. megaphone or whistle).
        # 100 units == 10 cells == approximately 30 m at the 3 m / cell scale.
        self.broadcast_radius = 100.0

        # Damage is tracked in two independent pools because the per-tick
        # rates differ. Fire deals a flat +2.0 per tick when local fire
        # exceeds 0.3, so the 300 cap corresponds to roughly 150 ticks of
        # sustained heavy fire before death. Smoke deals at most +0.5 per
        # tick (local_smoke x 0.5), so the lower 250 cap still corresponds
        # to roughly 500 ticks of sustained heavy smoke. Fire is therefore
        # lethal much faster despite the higher cap; the gas masks slow
        # smoke accumulation but do not prevent it.
        self.alive = True
        self._fire_damage = 0.0
        self._max_fire_damage = 300.0

        self._smoke_damage = 0.0
        self._max_smoke_damage = 250.0

        # Health is the worst of the two normalised damage ratios. When it
        # drops below the threshold the guide abandons its mission and
        # heads for the nearest exit (60 percent health remaining).
        self._health_threshold = 0.6

        # Self-evacuation state.
        self._self_evacuating = False
        self._escape_waypoints = []
        self.escaped = False  # True if the guide successfully reached an exit.

        # Exit assigned to this guide (used by BeaconGuide).
        self.assigned_exit = None

        # Group of civilians being escorted (used by SweeperGuide and
        # RouterGuide). Beacon never appends to it.
        self._group = []

    @property
    def health(self) -> float:
        """Current health as a 0.0 to 1.0 ratio. 1.0 is full, 0.0 is dead.

        Both damage pools accumulate at the same time when both fire and
        smoke are present, but death is triggered by either pool hitting
        its cap independently (see update()). Health therefore reports
        the ratio closest to a kill threshold, i.e. the worse of the two.
        """
        fire_ratio = self._fire_damage / self._max_fire_damage
        smoke_ratio = self._smoke_damage / self._max_smoke_damage
        return max(0.0, 1.0 - max(fire_ratio, smoke_ratio))

    def update(
        self, env: Environment, smoke: SmokeSystem, fire: FireSystem, civilians: list
    ) -> None:
        """Apply per-tick damage, store grid references, trigger self-evacuation.

        Subclasses must call super().update() before their own logic so
        damage accumulation, death checks, and self-evacuation start at
        the right moment.
        """
        if not self.alive:
            return

        # Cache the current grids so danger-aware path-finding helpers can
        # read them without recomputation.
        self._fire_grid = fire.density
        self._smoke_grid = smoke.density

        # Cache the current grid position used by dynamic-max-group sampling.
        self._last_grid_pos = env.world_to_grid(self.pos[0], self.pos[1])

        local_fire = fire.get_density_at(self.pos[0], self.pos[1])
        local_smoke = smoke.get_density_at(self.pos[0], self.pos[1])

        if local_fire > 0.3:
            self._fire_damage += 2.0
        if local_smoke > 0.3:
            self._smoke_damage += local_smoke * 0.5  # Half of civilian rate.

        if (
            self._fire_damage >= self._max_fire_damage
            or self._smoke_damage >= self._max_smoke_damage
        ):
            self.alive = False
            return

        if not self._self_evacuating and self.health < self._health_threshold:
            self._start_self_evacuation(env)

    def _start_self_evacuation(self, env: Environment) -> None:
        """Mark the guide as self-evacuating and plan a path to the nearest exit."""
        self._self_evacuating = True

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
        """Advance along the escape path. Return True while still self-evacuating."""
        if not self._self_evacuating:
            return False

        # Reaching any exit completes the self-evacuation.
        for ex, ey in self.known_exits:
            dist = np.sqrt((self.pos[0] - ex) ** 2 + (self.pos[1] - ey) ** 2)
            if dist < env.cell_size:
                self.alive = False  # Remove from active simulation.
                self.escaped = True  # Distinguish a successful escape from death.
                return True

        # Otherwise step toward the next waypoint.
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
                self.vel = direction * self.speed * 1.5  # Sprint when self-evacuating.

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
        """Smoke-attenuated broadcast radius with a 30-unit floor.

        The floor reflects that even in full smoke a guide can still be
        heard from roughly three cells away.
        """
        local_smoke = smoke.get_density_at(self.pos[0], self.pos[1])
        return max(30.0, self.broadcast_radius * (1.0 - local_smoke * 0.5))

    def get_visual_recruit_radius(self, smoke: SmokeSystem) -> float:
        """Smoke-attenuated visual recruit range for physical recruitment.

        The guide is treated as a light-emitting object (hi-vis gear,
        flashlight) and wears breathing apparatus. The mask reduces the
        smoke obscuration effect to 60 percent of what an unprotected
        civilian experiences, since particle obscuration remains while eye
        irritation is removed.

        Used by SweeperGuide and RouterGuide for physical recruitment, in
        contrast to BeaconGuide's broadcast-based recruitment.
        """
        local_smoke = smoke.get_density_at(self.pos[0], self.pos[1])
        base_range = 70.0  # 7 cells in clear air, approximately 20 m.
        smoke_effect = local_smoke * 0.6
        return base_range * max(0.15, 1.0 - smoke_effect)

    def _has_line_of_sight(self, target_pos, env) -> bool:
        """Ray-cast along the line to target_pos, returning False on wall hit.

        Used by visual recruitment so SweeperGuide and RouterGuide cannot
        recruit civilians on the other side of a wall.
        """
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
        """Move along the precomputed waypoint list, rerouting if it becomes unsafe.

        The next ten waypoints are inspected for fire and heavy smoke. If
        any waypoint exceeds the danger threshold, a new BFS path is
        planned over a danger grid that combines fire (weighted x3) and
        smoke (weight depending on whether civilians are being escorted).
        """
        if not self._waypoints:
            self.vel = np.array([0.0, 0.0])
            return

        fg = getattr(self, "_fire_grid", None)
        sg = getattr(self, "_smoke_grid", None)
        # Use a tighter smoke threshold while escorting civilians, since they
        # have no breathing protection.
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
                # Combine fire and smoke into a single danger grid for BFS.
                # Fire is scaled by three so that the 0.3 threshold rejects
                # fire above 0.1, which matches civilian fire-damage onset.
                # Smoke is given full weight when carrying a group and half
                # weight otherwise.
                danger = (fg * 3.0) if fg is not None else np.zeros_like(env.grid, dtype=float)
                if sg is not None:
                    smoke_weight = 1.0 if has_group else 0.5
                    danger = np.maximum(danger, sg * smoke_weight)
                final = self._waypoints[-1]
                self._waypoints = bfs_path(env, self.pos, final, fire_grid=danger)
                if not self._waypoints:
                    # No path that avoids danger. Retry without smoke avoidance.
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
            # Slow to roughly civilian pace when escorting a group.
            if has_group:
                move_speed = self.speed * 0.75
            else:
                move_speed = self.speed
            self.vel = direction * move_speed

        vx, vy = self.vel
        x, y = self.pos
        if env.is_walkable(x + vx, y + vy):
            self.pos = np.array([x + vx, y + vy])
        elif env.is_walkable(x + vx, y):
            self.pos = np.array([x + vx, y])
        elif env.is_walkable(x, y + vy):
            self.pos = np.array([x, y + vy])

    def _count_recruitable(self, env, smoke, civilians):
        """Count civilians actually recruitable by _recruit_nearby.

        Uses identical visual-range and line-of-sight checks so the
        count never reports false positives.
        """
        radius = self.get_visual_recruit_radius(smoke)
        count = 0
        for civ in civilians:
            if civ.state in (MOVING_TO_EXIT, ESCAPED, DEAD, FOLLOWING_GUIDE):
                continue
            if civ.assigned_guide is not None:
                continue
            dist = np.linalg.norm(civ.pos - self.pos)
            if dist < radius and self._has_line_of_sight(civ.pos, env):
                count += 1
        return count

    def _recruit_nearby(self, env, smoke, civilians):
        """Recruit civilians within visual range, requiring line of sight.

        Capped at the dynamic max-group ceiling defined by the subclass.
        Walls block recruitment. Civilians who are already on the way to
        an exit, queueing, or attached to another guide are skipped.
        IDLE and REACTING civilians have their reaction shortened so
        they will be ready for recruitment on a later tick.
        """
        if len(self._group) >= self._effective_max_group():
            return
        radius = self.get_visual_recruit_radius(smoke)
        for civ in civilians:
            if len(self._group) >= self._effective_max_group():
                break
            if civ.state in (MOVING_TO_EXIT, ESCAPED, DEAD, FOLLOWING_GUIDE, QUEUED):
                continue
            if civ.assigned_guide is not None:
                continue
            dist = np.linalg.norm(civ.pos - self.pos)
            if dist < radius and self._has_line_of_sight(civ.pos, env):
                if civ.state == IDLE:
                    civ.state = REACTING
                    civ._reaction_counter = 0
                    civ._reaction_delay = min(civ._reaction_delay, 5)
                elif civ.state == REACTING:
                    civ._reaction_delay = min(civ._reaction_delay, 5)
                    continue
                else:
                    self._group.append(civ)
                    civ.assigned_guide = self
                    civ.state = FOLLOWING_GUIDE

    def _recruit_nearby_unlimited(self, env, smoke, civilians):
        """Recruit any civilian within visual range without a group cap.

        Used during escort: every civilian passed on the way to the
        exit is picked up, provided line of sight allows. QUEUED
        civilians are skipped to avoid pulling them out of exit queues.
        """
        radius = self.get_visual_recruit_radius(smoke)
        for civ in civilians:
            if civ.state in (MOVING_TO_EXIT, ESCAPED, DEAD, FOLLOWING_GUIDE, QUEUED):
                continue
            if civ.assigned_guide is not None:
                continue
            dist = np.linalg.norm(civ.pos - self.pos)
            if dist < radius and self._has_line_of_sight(civ.pos, env):
                if civ.state == IDLE:
                    civ.state = REACTING
                    civ._reaction_counter = 0
                    civ._reaction_delay = min(civ._reaction_delay, 5)
                elif civ.state == REACTING:
                    civ._reaction_delay = min(civ._reaction_delay, 5)
                    continue
                else:
                    self._group.append(civ)
                    civ.assigned_guide = self
                    civ.state = FOLLOWING_GUIDE

    def _clean_group(self):
        """Remove escaped or dead civilians from the group list."""
        self._group = [c for c in self._group if c.state not in (ESCAPED, DEAD)]

    def _release_group_wandering(self):
        """Release the group when the guide dies or self-evacuates.

        Civilians attached to the group are returned to WANDERING so
        they can resume autonomous evacuation. The guide's live-state
        entries are also released from the shared blackboard so peers
        see immediate cleanup rather than waiting for TTL expiry.
        """
        for civ in self._group:
            if civ.state == FOLLOWING_GUIDE:
                civ.assigned_guide = None
                civ.state = WANDERING
        self._group.clear()
        self._blackboard.release(self._guide_id)

    def _do_releasing(self, env):
        """Stand at the exit and direct group members through it.

        Civilians within five cells of the exit or three cells of the
        guide receive a forced exit target. After release_max_wait
        ticks any remaining group members are forced through and the
        policy-specific transition is delegated to _on_release_complete.
        """
        self.vel = np.zeros(2)
        self._release_wait += 1

        still = []
        for civ in self._group:
            if civ.state != FOLLOWING_GUIDE:
                continue
            dist_exit = np.linalg.norm(civ.pos - np.array(self._escort_exit))
            dist_guide = np.linalg.norm(civ.pos - self.pos)
            if dist_exit < env.cell_size * 5 or dist_guide < env.cell_size * 3:
                civ._set_exit_target_force(env, self._escort_exit)
            else:
                still.append(civ)
        self._group = still

        if not self._group or self._release_wait >= self._release_max_wait:
            # On timeout, give stragglers an exit target rather than
            # leaving them lost.
            for civ in self._group:
                if civ.state == FOLLOWING_GUIDE:
                    civ._set_exit_target_force(env, self._escort_exit)
            self._group.clear()
            self._on_release_complete()

    def _on_release_complete(self):
        """Hook called when the release phase finishes.

        Subclasses override this to define the policy-specific
        transition (e.g. SweeperGuide returns to its territory,
        RouterGuide rescans from the current position).
        """
        pass
