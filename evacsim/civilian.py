"""Civilian agent: a 5-rule reactive model with a finite-state machine.

The agent's behaviour follows five rules that drive its state transitions
and movement choices:

    1. No target: random walk along a cardinal direction until a wall
       is reached, then turn.
    2. Has target: follow a pre-computed BFS path to a doorway or exit.
    3. In smoke: random direction change with probability proportional
       to local smoke density (disorientation).
    4. Other civilians moving nearby: bias toward their average
       direction (herding).
    5. Potency reaches zero: the agent dies.

The finite-state machine has nine reachable states (IDLE, REACTING,
FLEEING_ROOM, WANDERING, WALL_FOLLOWING, FOLLOWING, FOLLOWING_GUIDE,
MOVING_TO_EXIT, QUEUED) plus two terminal states (ESCAPED, DEAD).
Damage from fire and smoke is applied every tick and the agent dies
when its potency reaches zero.
"""

import numpy as np
from collections import deque

from evacsim.environment import Environment, CellType
from evacsim.smoke import SmokeSystem
from evacsim.fire import FireSystem

# -- States --
IDLE = 0  # Pre-evacuation: at desk, unaware of the emergency.
REACTING = 1  # Emergency noticed; frozen during the reaction delay.
FLEEING_ROOM = 2  # Following a BFS path to a corridor doorway.
WANDERING = 3  # Random walk with no goal.
WALL_FOLLOWING = 4  # Blind in smoke; hand-on-wall navigation.
FOLLOWING = 5  # Herding: biased toward a leader or a crowd direction.
FOLLOWING_GUIDE = 6  # In a guide's group; uses hybrid BFS plus boids.
MOVING_TO_EXIT = 7  # Following a BFS path to a known exit.
QUEUED = 8  # At the exit, waiting in line to escape.
ESCAPED = 9
DEAD = 10

# Imported after the state constants so that guides.common and the
# guide policies (which import these states) see them defined when
# their own module bodies run as part of this import chain.
from evacsim.guides.common import bfs_path


class Civilian:

    def __init__(self, x: float, y: float, speed: float, rng: np.random.Generator):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.array([0.0, 0.0])
        self.speed = speed
        self._rng = rng

        # Two vision ranges. People are large moving objects (easier to
        # spot); exits are static signs (smaller, easier to miss).
        self.people_vision_radius = 70.0  # 7 cells, about 20 m in clear air.
        self.exit_vision_radius = 25.0  # 2.5 cells, about 7.5 m in clear air.

        self.state = IDLE

        # Tick at which this civilian transitions to ESCAPED. Stays
        # None if the civilian never escapes (died or still in
        # building). Used by the evacuation-curve analysis to compute
        # the cumulative escape rate over time.
        self.escape_tick = None

        # Reaction delay: 0.5 to 2 seconds at 30 fps.
        self._reaction_delay = int(rng.uniform(15, 60))
        self._reaction_counter = 0

        # Potency: energy and health. Initial value drawn uniformly
        # in [50, 100]. Decreases in fire and smoke; no recovery.
        # When it reaches zero the agent dies.
        self.potency = rng.uniform(50.0, 100.0)

        # Herding state: once following a crowd, commit for N ticks.
        self._herding_commitment = 0
        self._herding_leader = None  # Specific civilian being followed.

        # BFS waypoints and target.
        self._waypoints = []
        self._target_exit = None

        # Random-walk direction (cardinal).
        self._walk_dir = None

        # Exit told by a guide before the reaction finishes; consumed
        # at the moment the agent starts moving.
        self._guide_told_exit = None

        # Wall-following hand rule (random left or right at spawn).
        self._wall_hand = 1 if rng.random() < 0.5 else -1

        # Stuck detection.
        self._prev_pos = self.pos.copy()
        self._stuck_ticks = 0

        # State for following a guide via the hybrid BFS-plus-boids loop.
        self.assigned_guide = None
        self._boids_bfs_cache = []  # Cached BFS path to the guide.
        self._boids_bfs_tick = 0  # Tick when the cache was last refreshed.
        self._cached_guide_pos = None  # Guide position at cache build, used
        # for distance-based invalidation.

    # ----------------------------------------------------------------
    # Main update
    # ----------------------------------------------------------------

    def update(
        self,
        env: Environment,
        smoke: SmokeSystem,
        fire: FireSystem,
        all_civilians: list,
    ) -> None:

        if self.state in (ESCAPED, DEAD):
            return

        # Cache references used by helper methods this tick.
        self._env_ref = env
        # Cache the fire grid for BFS calls within this tick.
        self._fire_grid = fire.density

        x, y = self.pos
        local_smoke = smoke.get_density_at(x, y)
        local_fire = fire.get_density_at(x, y)
        in_smoke = local_smoke > 0.3
        in_fire = local_fire > 0.1

        # IDLE: slow ambient movement plus passive triggers.
        # The agent is not yet aware of the emergency, moves at a
        # casual pace without any destination, and is excluded from
        # exit detection (the early return below bypasses the spotting
        # logic). Fire, smoke, sensed danger, and sight of others
        # fleeing are the triggers that move the agent into REACTING.
        # IDLE agents are also excluded from being seen as 'fleeing'
        # by others, so their ambient motion does not cause cascading
        # false alarms.
        if self.state == IDLE:
            if (
                in_fire
                or in_smoke
                or self._sense_danger(env, smoke, fire)
                or self._see_others_fleeing(all_civilians)
            ):
                self.state = REACTING
                self._reaction_counter = 0
                return

            # Ambient movement at 30 percent of normal speed. Slower
            # than evacuation pace so the visual distinction is clear
            # and so an IDLE agent does not accidentally reach an exit
            # cell before noticing the emergency.
            idle_speed = self.speed * 0.3

            # About a 2 percent chance to change direction per tick,
            # producing a natural-looking ambient motion of a new
            # heading every roughly 50 ticks.
            if self._walk_dir is None or self._rng.random() < 0.02:
                self._pick_random_cardinal()

            # _do_random_walk sets self.vel; _move applies wall
            # collision and inter-civilian separation.
            self._do_random_walk(idle_speed, local_smoke)
            self._move(env, idle_speed, all_civilians)
            return

        # Rule 5: potency damage with no recovery.
        # Applied before state-specific logic so that REACTING agents
        # also take damage. Fire and smoke damage stack: a cell with
        # mild fire and heavy smoke would otherwise deal less damage
        # than heavy smoke alone (fire would short-circuit the smoke
        # branch). Additive damage matches the physical reality that
        # heat exposure and asphyxiation are separate injury
        # mechanisms.
        if in_fire:
            self.potency -= 3.0 * local_fire
        if in_smoke:
            self.potency -= local_smoke

        if self.potency <= 0:
            self.state = DEAD
            self.vel = np.zeros(2)
            return

        # REACTING: freeze for the reaction delay, then start moving.
        if self.state == REACTING:
            self._reaction_counter += 1
            if self._reaction_counter >= self._reaction_delay:
                self._start_moving(env)
            return

        # Vision and speed for this tick.
        people_vision, exit_vision = self._calc_vision(local_smoke)
        speed = self._calc_speed(in_fire, in_smoke)

        # Opportunistic exit spotting. The exit_vision range is
        # tighter than people_vision because exit signs are small.
        # FOLLOWING_GUIDE agents can also spot exits: if the agent
        # sees an exit door it goes through it rather than waiting
        # for the guide to announce it.
        if exit_vision > 0 and self.state not in (MOVING_TO_EXIT, QUEUED):
            exit_pos = self._spot_exit(env, exit_vision)
            if exit_pos is not None:
                # If currently in a guide's group, leave the group first.
                if self.state == FOLLOWING_GUIDE and self.assigned_guide is not None:
                    try:
                        self.assigned_guide._group.remove(self)
                    except ValueError:
                        pass
                    self.assigned_guide = None
                self._set_exit_target_force(env, exit_pos)

        # Rule 4: herding. Follow someone who looks like they know the way.
        if people_vision > 0 and self.state == WANDERING:
            leader = self._find_goal_leader(all_civilians, people_vision)
            if leader is not None:
                # Found a goal-directed leader; follow them.
                self._herding_leader = leader
                self.state = FOLLOWING
                self._herding_commitment = 20
            else:
                # No leader. Fall back to the average crowd direction.
                crowd_dir = self._sense_crowd_direction(all_civilians, people_vision)
                if crowd_dir is not None:
                    self._walk_dir = crowd_dir
                    self._herding_leader = None
                    self.state = FOLLOWING
                    self._herding_commitment = 20

        # State behaviours.

        if self.state == FLEEING_ROOM:
            # Rule 2: follow the BFS path to a doorway.
            self._do_follow_path(env, speed, fallback=WANDERING)

        elif self.state == MOVING_TO_EXIT:
            # Rule 2: follow the BFS path to an exit.
            if self._target_exit is not None:
                dist = np.linalg.norm(self.pos - np.array(self._target_exit))
                if dist < env.cell_size:
                    # Do not escape immediately; enter the exit queue.
                    self.state = QUEUED
                    self.vel = np.zeros(2)
                    return
            self._do_follow_path(env, speed, fallback=WANDERING)

        elif self.state == QUEUED:
            # Waiting at the exit. The simulation engine processes the
            # queue. The agent still takes damage while waiting.
            self.vel = np.zeros(2)

        elif self.state == FOLLOWING:
            leader = self._herding_leader

            # Following a specific leader vs. a crowd direction.
            if leader is not None:
                # Leader died, escaped, or is no longer visible.
                if leader.state in (ESCAPED, DEAD):
                    leader = None
                    self._herding_leader = None
                elif people_vision > 0:
                    d = np.linalg.norm(self.pos - leader.pos)
                    if d > people_vision or not self._has_los(leader.pos, env):
                        leader = None
                        self._herding_leader = None

            if leader is not None:
                # Direct vector toward the leader (not boids).
                diff = leader.pos - self.pos
                dist = np.linalg.norm(diff)
                if dist > 1.0:
                    self.vel = (diff / dist) * speed
            elif self._walk_dir is not None:
                # Fallback: follow the crowd direction.
                self.vel = self._walk_dir * speed

            # Stuck for too long: give up herding.
            if self._stuck_ticks >= 8:
                self.state = WANDERING
                self._walk_dir = None
                self._herding_leader = None
                self._herding_commitment = 0
            else:
                self._herding_commitment = max(0, self._herding_commitment - 1)

                if self._herding_commitment > 0:
                    # Still committed; keep following but check for exits.
                    if exit_vision > 0:
                        exit_pos = self._spot_exit(env, exit_vision)
                        if exit_pos is not None:
                            self._set_exit_target(env, exit_pos)
                elif people_vision > 0:
                    # Commitment expired; recheck for a leader or crowd.
                    new_leader = self._find_goal_leader(all_civilians, people_vision)
                    if new_leader is not None:
                        self._herding_leader = new_leader
                        self._herding_commitment = 20
                    else:
                        crowd_dir = self._sense_crowd_direction(
                            all_civilians, people_vision
                        )
                        if crowd_dir is not None:
                            self._walk_dir = crowd_dir
                            self._herding_leader = None
                            self._herding_commitment = 20
                        else:
                            self.state = WANDERING
                            self._walk_dir = None
                            self._herding_leader = None
                    if exit_vision > 0:
                        exit_pos = self._spot_exit(env, exit_vision)
                        if exit_pos is not None:
                            self._set_exit_target(env, exit_pos)

        elif self.state == FOLLOWING_GUIDE:
            guide = self.assigned_guide
            if guide is None or not guide.alive or self not in guide._group:
                self.assigned_guide = None
                self.state = WANDERING
                self._walk_dir = None
            else:
                dist_to_guide = np.linalg.norm(self.pos - guide.pos)

                # Give up following if either:
                # (a) far from the guide for too long (lost in
                #     corridors), or
                # (b) stuck for a long time regardless of distance,
                #     for example pinned against a wall next to the
                #     guide where neither can move. Without case (b)
                #     a pinned agent could stay in FOLLOWING_GUIDE
                #     until it dies of smoke.
                far_and_stuck = (
                    dist_to_guide > env.cell_size * 20 and self._stuck_ticks >= 8
                )
                long_stuck = self._stuck_ticks >= 30
                if far_and_stuck or long_stuck:
                    try:
                        guide._group.remove(self)
                    except ValueError:
                        pass
                    self.assigned_guide = None
                    self.state = WANDERING
                    self._walk_dir = None
                else:
                    # Hybrid BFS plus boids:
                    # BFS gives direction (handles walls, doorways).
                    # Boids separation prevents piling up.
                    self.vel = self._hybrid_boids_follow(
                        guide, all_civilians, speed, env
                    )

        elif self.state == WALL_FOLLOWING:
            # Blind in smoke: hand-on-wall.
            self._do_wall_follow(env, speed)
            if people_vision > 10 and not in_smoke:
                self.state = WANDERING
                self._walk_dir = None

        elif self.state == WANDERING:
            if in_smoke and people_vision < 5:
                # Effectively blind: switch to wall-following.
                self.state = WALL_FOLLOWING
                self._do_wall_follow(env, speed)
            else:
                # Rule 1: random walk.
                # Rule 3: disorientation in smoke.
                self._do_random_walk(speed, local_smoke)

        # Apply movement.
        self._move(env, speed, all_civilians)

    # ----------------------------------------------------------------
    # State transitions
    # ----------------------------------------------------------------

    def _start_moving(self, env):
        """Choose a destination at the end of the reaction delay.

        If a guide pre-assigned an exit, head there. Otherwise look
        for a doorway out of the current room; if none is reachable
        within the search radius, fall back to wandering.
        """
        if self._guide_told_exit is not None:
            self._set_exit_target(env, self._guide_told_exit)
            self._guide_told_exit = None
            return

        doorway = self._find_doorway(env)
        if doorway is not None:
            fg = getattr(self, "_fire_grid", None)
            self._waypoints = bfs_path(env, self.pos, doorway, fire_grid=fg)
            self.state = FLEEING_ROOM
        else:
            self.state = WANDERING
            self._pick_random_cardinal()

    def _set_exit_target(self, env, exit_pos):
        """Set an exit as the goal with a fresh BFS path.

        Does not override FOLLOWING_GUIDE: a guided agent stays with
        its guide unless the guide releases it.
        """
        if self.state == FOLLOWING_GUIDE:
            return
        self._target_exit = exit_pos
        fg = getattr(self, "_fire_grid", None)
        self._waypoints = bfs_path(env, self.pos, np.array(exit_pos), fire_grid=fg)
        self.state = MOVING_TO_EXIT

    def _set_exit_target_force(self, env, exit_pos):
        """Force-set the exit target. Used when a guide releases the group."""
        self._target_exit = exit_pos
        fg = getattr(self, "_fire_grid", None)
        self._waypoints = bfs_path(env, self.pos, np.array(exit_pos), fire_grid=fg)
        self.state = MOVING_TO_EXIT
        self.assigned_guide = None

    # ----------------------------------------------------------------
    # Movement rules
    # ----------------------------------------------------------------

    def _do_random_walk(self, speed, smoke_density):
        """Rules 1 and 3: cardinal random walk with smoke-induced disorientation.

        Pick a cardinal direction and walk until a wall. In smoke,
        a random chance to change direction simulates disorientation.
        """
        # Rule 3: disorientation.
        if smoke_density > 0.2:
            # Higher smoke increases the chance of an arbitrary turn.
            if self._rng.random() < smoke_density * 0.3:
                self._pick_random_cardinal()

        # Pick an initial direction if there is none.
        if self._walk_dir is None:
            self._pick_random_cardinal()

        self.vel = self._walk_dir * speed

    def _do_follow_path(self, env, speed, fallback):
        """Rule 2: follow the BFS waypoints. Switch to fallback when exhausted."""
        if not self._waypoints:
            self.state = fallback
            self._walk_dir = None
            return

        target = self._waypoints[0]
        diff = target - self.pos
        dist = np.linalg.norm(diff)

        if dist < env.cell_size * 0.5:
            self._waypoints.pop(0)
            if not self._waypoints:
                self.state = fallback
                self._walk_dir = None
                return
            target = self._waypoints[0]
            diff = target - self.pos
            dist = np.linalg.norm(diff)

        if dist > 0:
            self.vel = (diff / dist) * speed

    def _do_wall_follow(self, env, speed):
        """Hand-on-wall navigation when the agent is effectively blind in smoke."""
        x, y = self.pos
        cs = env.cell_size * 0.5

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if not env.is_walkable(x + dx * cs, y + dy * cs):
                wall = np.array([dx, dy], dtype=float)
                if self._wall_hand == 1:
                    self.vel = np.array([-wall[1], wall[0]]) * speed
                else:
                    self.vel = np.array([wall[1], -wall[0]]) * speed
                return

        # No wall nearby: walk forward in the current direction.
        norm = np.linalg.norm(self.vel)
        if norm > 0:
            self.vel = (self.vel / norm) * speed
        else:
            self._pick_random_cardinal()
            self.vel = self._walk_dir * speed

    def _hybrid_boids_follow(self, guide, all_civilians, speed, env):
        """Hybrid BFS plus boids steering for following a guide.

        Three steering forces are combined:
            COHESION via BFS: a recomputed BFS path toward the guide
                handles walls, doorways and corridors automatically.
            ALIGNMENT: match the guide's current direction.
            SEPARATION: push away from nearby group members to
                prevent piling up at doorways and corridors.

        BFS handles navigation; boids handles group spacing.
        """
        # Recompute the BFS path on either trigger:
        # (a) eight ticks elapsed since the last recompute (time-based
        #     fallback);
        # (b) the guide has moved more than two cells since the cache
        #     was built (catches sharp turns and sprints that would
        #     otherwise leave the agent walking toward the guide's
        #     stale position for up to eight ticks).
        # Intentionally not recomputing on an empty cache: an empty
        # cache (BFS returned no path) would cause BFS to run every
        # single tick, which is O(rows * cols) per civilian per tick
        # when the guide is unreachable.
        self._boids_bfs_tick += 1
        guide_moved = 0.0
        if self._cached_guide_pos is not None:
            guide_moved = np.linalg.norm(guide.pos - self._cached_guide_pos)
        if self._boids_bfs_tick >= 8 or guide_moved > env.cell_size * 2:
            self._boids_bfs_tick = 0
            self._cached_guide_pos = guide.pos.copy()
            fg = getattr(self, "_fire_grid", None)
            self._boids_bfs_cache = bfs_path(env, self.pos, guide.pos, fire_grid=fg)

        # Follow the cached BFS waypoints.
        cohesion_dir = np.zeros(2)
        if self._boids_bfs_cache:
            target = self._boids_bfs_cache[0]
            diff = target - self.pos
            d = np.linalg.norm(diff)
            if d < env.cell_size * 0.5:
                self._boids_bfs_cache.pop(0)
                if self._boids_bfs_cache:
                    target = self._boids_bfs_cache[0]
                    diff = target - self.pos
                    d = np.linalg.norm(diff)
            if d > 0:
                cohesion_dir = diff / d
        else:
            # No path: fall back to a direct vector toward the guide.
            diff = guide.pos - self.pos
            d = np.linalg.norm(diff)
            if d > 0:
                cohesion_dir = diff / d

        # Alignment: match the guide's direction.
        gv = np.linalg.norm(guide.vel)
        alignment_dir = guide.vel / gv if gv > 0.1 else np.zeros(2)

        # Separation: push away from nearby group members.
        separation_dir = np.zeros(2)
        sep_r = env.cell_size * 1.5
        sep_count = 0
        for other in all_civilians:
            if other is self or other.state != FOLLOWING_GUIDE:
                continue
            if other.assigned_guide is not guide:
                continue
            diff_sep = self.pos - other.pos
            dn = np.linalg.norm(diff_sep)
            if 0 < dn < sep_r:
                separation_dir += (diff_sep / dn) * (1.0 - dn / sep_r)
                sep_count += 1
        if sep_count > 0:
            separation_dir /= sep_count

        # Combine the three forces. Cohesion is strongest because the
        # primary task is to follow the guide; separation prevents
        # piling up; alignment provides a smoother trajectory.
        steer = cohesion_dir * 1.0 + alignment_dir * 0.5 + separation_dir * 1.5

        norm = np.linalg.norm(steer)
        if norm > 0:
            steer = (steer / norm) * speed
        else:
            steer = cohesion_dir * speed

        return steer

    def _move(self, env, speed, all_civilians=None):
        """Apply velocity, with wall collision and universal separation.

        Universal separation applies to every civilian regardless of
        state. It prevents agents from overlapping and produces
        natural queueing at exits and doorways.
        """
        x, y = self.pos

        # Universal separation: push away from nearby civilians.
        # Manhattan pre-filter avoids the per-pair sqrt for distant ones.
        if all_civilians is not None:
            sep_force = np.zeros(2)
            sep_r = env.cell_size * 2.0  # Two-cell radius.
            px, py = self.pos[0], self.pos[1]
            for other in all_civilians:
                if other is self or other.state in (ESCAPED, DEAD):
                    continue
                dx = px - other.pos[0]
                dy = py - other.pos[1]
                if abs(dx) > sep_r or abs(dy) > sep_r:
                    continue
                dn = np.sqrt(dx * dx + dy * dy)
                if 0 < dn < sep_r:
                    strength = (1.0 - dn / sep_r) ** 2
                    sep_force[0] += (dx / dn) * strength
                    sep_force[1] += (dy / dn) * strength
            self.vel = self.vel + sep_force * speed * 0.4

        # Clamp the velocity magnitude so the separation force cannot
        # push agents through thin walls.
        vmag = np.linalg.norm(self.vel)
        max_vel = speed * 2.0
        if vmag > max_vel:
            self.vel = (self.vel / vmag) * max_vel

        vx, vy = self.vel

        if env.is_walkable(x + vx, y + vy):
            self.pos = np.array([x + vx, y + vy])
        elif env.is_walkable(x + vx, y):
            self.pos = np.array([x + vx, y])
        elif env.is_walkable(x, y + vy):
            self.pos = np.array([x, y + vy])
        else:
            self._pick_random_cardinal()

        # Stuck detection.
        moved = np.linalg.norm(self.pos - self._prev_pos)
        if moved < 0.5:
            self._stuck_ticks += 1
        else:
            self._stuck_ticks = 0
            self._prev_pos = self.pos.copy()

        if self._stuck_ticks >= 10:
            self._pick_random_cardinal()
            self._stuck_ticks = 0

    def _pick_random_cardinal(self):
        """Pick a random cardinal direction (up, down, left, or right)."""
        dirs = [
            np.array([1.0, 0.0]),
            np.array([-1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, -1.0]),
        ]
        self._walk_dir = dirs[self._rng.integers(4)]

    # ----------------------------------------------------------------
    # Perception
    # ----------------------------------------------------------------

    def _calc_vision(self, smoke_density):
        """Return (people_vision, exit_vision) attenuated by smoke.

        Civilians have unprotected eyes, so smoke causes irritation
        and tearing and vision drops rapidly with smoke density.
        """
        smoke_factor = max(0.15, 1.0 - smoke_density)
        people_v = self.people_vision_radius * smoke_factor
        exit_v = self.exit_vision_radius * smoke_factor
        return people_v, exit_v

    def _calc_speed(self, in_fire, in_smoke):
        """Return the per-tick movement speed for the current conditions."""
        has_path = self.state in (FLEEING_ROOM, MOVING_TO_EXIT, FOLLOWING)
        following_guide = self.state == FOLLOWING_GUIDE
        if in_fire:
            return self.speed * 1.8
        elif in_smoke:
            if following_guide:
                # Guide is leading: the agent is motivated and moves
                # at 85 percent of base speed even in smoke.
                return self.speed * 0.85
            elif has_path:
                return self.speed * 0.7
            else:
                return self.speed * 0.3
        return self.speed

    def _see_others_fleeing(self, all_civilians):
        """Return True if a fleeing civilian is visible nearby.

        Captures the alert-by-imitation effect: if someone nearby is
        evacuating, the agent infers there is danger and starts to
        react. Requires line of sight; walls block the cue.
        """
        env = getattr(self, "_env_ref", None)
        for other in all_civilians:
            if other is self:
                continue
            if other.state in (IDLE, REACTING, ESCAPED, DEAD):
                continue
            d = np.linalg.norm(self.pos - other.pos)
            if d < self.people_vision_radius * 0.5:
                if env is not None and self._has_los(other.pos, env):
                    return True
                elif env is None:
                    return True  # Fallback before env is cached.
        return False

    def _find_goal_leader(self, all_civilians, radius):
        """Return the nearest visible goal-directed civilian.

        A goal-directed civilian is one in MOVING_TO_EXIT or
        FLEEING_ROOM, the states that look 'confident'. People follow
        the confident person rather than the panicking crowd.
        """
        env = getattr(self, "_env_ref", None)
        best, best_d = None, float("inf")
        for other in all_civilians:
            if other is self:
                continue
            if other.state not in (MOVING_TO_EXIT, FLEEING_ROOM):
                continue
            d = np.linalg.norm(self.pos - other.pos)
            if d < radius and d < best_d:
                if env is not None and self._has_los(other.pos, env):
                    best, best_d = other, d
        return best

    def _sense_danger(self, env, smoke, fire):
        """Return True if smoke or fire is detectable within one cell."""
        gr, gc = env.world_to_grid(self.pos[0], self.pos[1])
        rows, cols = smoke.density.shape
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r, c = gr + dr, gc + dc
                if 0 <= r < rows and 0 <= c < cols:
                    if smoke.density[r, c] > 0.1 or fire.density[r, c] > 0.05:
                        return True
        return False

    def _has_los(self, target_pos, env):
        """Line-of-sight raycast. Returns False if a wall blocks the line."""
        diff = target_pos - self.pos
        dist = np.linalg.norm(diff)
        if dist < env.cell_size:
            return True
        direction = diff / dist
        steps = int(dist / env.cell_size) + 1
        for i in range(1, min(steps + 1, 20)):
            cx = self.pos[0] + direction[0] * env.cell_size * i
            cy = self.pos[1] + direction[1] * env.cell_size * i
            if not env.is_walkable(cx, cy):
                return False
        return True

    def _spot_exit(self, env, eff_vision):
        """Return the nearest exit within vision range and line of sight."""
        best, best_d = None, float("inf")
        for ex, ey in env.exits:
            d = np.hypot(self.pos[0] - ex, self.pos[1] - ey)
            if d < eff_vision and d < best_d:
                if self._has_los(np.array([ex, ey]), env):
                    best, best_d = (ex, ey), d
        return best

    def _sense_crowd_direction(self, all_civilians, radius):
        """Rule 4: average direction of nearby visible moving civilians.

        Requires line of sight; walls block the cue. Returns None if
        there are fewer than two qualifying neighbours.
        """
        avg_dir = np.zeros(2)
        count = 0

        for other in all_civilians:
            if other is self or other.state in (IDLE, REACTING, ESCAPED, DEAD):
                continue
            d = np.linalg.norm(self.pos - other.pos)
            if d < radius and d > 0:
                other_speed = np.linalg.norm(other.vel)
                if other_speed > 0.5:
                    if self._has_los(other.pos, self._env_ref):
                        avg_dir += other.vel / other_speed
                        count += 1

        if count >= 2:
            norm = np.linalg.norm(avg_dir)
            if norm > 0:
                return avg_dir / norm
        return None

    # ----------------------------------------------------------------
    # Pathfinding
    # ----------------------------------------------------------------

    def _find_doorway(self, env):
        """BFS for the nearest cell that opens onto a corridor.

        A corridor cell is one with at least a six-cell straight run
        in some cardinal direction. Searches up to 25 cells.
        """
        gr, gc = env.world_to_grid(self.pos[0], self.pos[1])
        rows, cols = env.grid.shape

        def longest_run(r, c):
            best = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                n, nr, nc = 0, r + dr, c + dc
                while (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and env.grid[nr, nc] != CellType.WALL
                ):
                    n += 1
                    nr += dr
                    nc += dc
                best = max(best, n)
            return best

        if longest_run(gr, gc) >= 6:
            return None

        visited = {(gr, gc)}
        queue = deque([(gr, gc, 0)])
        while queue:
            r, c, d = queue.popleft()
            if d > 25:
                continue
            if d >= 1 and longest_run(r, c) >= 6:
                wx, wy = env.grid_to_world(r, c)
                return np.array([wx, wy], dtype=float)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    if env.grid[nr, nc] != CellType.WALL:
                        visited.add((nr, nc))
                        queue.append((nr, nc, d + 1))
        return None


# Spawning


def spawn_civilians(
    env: Environment, n: int, speed: float = 1.5, seed: int = 99
) -> list[Civilian]:
    """Create n civilians at room-clustered floor positions."""
    positions = env.spawn_positions(n, seed=seed)
    rng = np.random.default_rng(seed)
    return [Civilian(x, y, speed, rng) for x, y in positions]
