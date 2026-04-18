"""
Civilian agent — paper-grounded 5-rule model.

Academic basis:
  - Potency system: Kasereka et al. 2018 (energy 50-100, decreases in smoke/fire, no recovery)
  - Reaction delay: Proulx 2001 (pre-evacuation time 0.5-2s)
  - Random walk: Standard ABM approach (Senanayake et al. 2024 review)
  - Disorientation in smoke: Smart Cities MDPI 2024 (random direction change proportional to density)
  - Herding: Helbing et al. 2000 (follow average crowd direction, not individual)
  - Speed in smoke: Jin 1978 (visibility vs walking speed)
  - BFS pathfinding: Dijkstra variant (Kasereka 2018)

5 Rules:
  1. No target → random walk (pick cardinal direction, walk until wall)
  2. Has target → follow BFS path (push through smoke)
  3. In smoke → disorientation (random direction change, probability ∝ density)
  4. Nearby civilians moving → herding (bias toward average crowd direction)
  5. Potency ≤ 0 → dead
"""

import numpy as np
from evacsim.environment import Environment, CellType
from evacsim.smoke import SmokeSystem
from evacsim.fire import FireSystem
from evacsim.guide import bfs_path


# ── States (simplified) ─────────────────────────────────────────────
IDLE = 0            # At desk, unaware
REACTING = 1        # Noticed danger, frozen (reaction delay)
FLEEING_ROOM = 2    # Following BFS path to corridor doorway
WANDERING = 3       # Random walk, no goal
WALL_FOLLOWING = 4  # Blind in smoke, hand-on-wall
FOLLOWING = 5       # Herding — biased toward crowd direction
FOLLOWING_GUIDE = 6 # In sweeper's group, using boids to follow
MOVING_TO_EXIT = 7  # Following BFS path to exit
QUEUED = 8          # At exit, waiting in line to escape
ESCAPED = 9
DEAD = 10


class Civilian:

    def __init__(self, x: float, y: float, speed: float,
                 rng: np.random.Generator):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.array([0.0, 0.0])
        self.speed = speed
        self._rng = rng

        # Two vision ranges (Jin 1978, Clark et al. 1985):
        # People: large moving objects, easy to spot (Jin — reflecting objects)
        # Exits: small static signs, easy to miss in panic (Clark — exit sign visibility)
        self.people_vision_radius = 70.0  # 7 cells (~20m) — see people in clear air
        self.exit_vision_radius = 25.0    # 2.5 cells (~7.5m) — see exit signs in clear air

        self.state = IDLE

        # Escape tracking: tick at which this civilian transitions to ESCAPED.
        # Remains None if the civilian never escapes (died or still in building).
        # Used for evacuation-curve analysis (cumulative escape rate over time).
        self.escape_tick = None

        # Reaction delay: 0.5-2s at 30fps (Proulx 2001)
        self._reaction_delay = int(rng.uniform(15, 60))
        self._reaction_counter = 0

        # Potency: energy/health system (Kasereka 2018)
        # Random 50-100. Decreases in smoke/fire. NO recovery. At 0 = dead.
        self.potency = rng.uniform(50.0, 100.0)

        # Herding commitment: once following a crowd, commit for N ticks
        self._herding_commitment = 0
        self._herding_leader = None  # Specific civilian we're following

        # BFS waypoints + target
        self._waypoints = []
        self._target_exit = None

        # Random walk: current direction (cardinal)
        self._walk_dir = None

        # Exit told by guide (set before reaction finishes, used after)
        self._guide_told_exit = None

        # Wall-following hand rule
        self._wall_hand = 1 if rng.random() < 0.5 else -1

        # Stuck detection
        self._prev_pos = self.pos.copy()
        self._stuck_ticks = 0

        # For sweeper guide (Phase 2)
        self.assigned_guide = None
        self._boids_bfs_cache = []  # Cached BFS path for boids wall fallback
        self._boids_bfs_tick = 0    # When cache was last updated

    # ================================================================
    # MAIN UPDATE
    # ================================================================

    def update(self, env: Environment, smoke: SmokeSystem,
               fire: FireSystem, all_civilians: list) -> None:

        if self.state in (ESCAPED, DEAD):
            return

        # Store references for helper methods
        self._env_ref = env
        # Store fire grid for BFS calls within this tick
        self._fire_grid = fire.density

        x, y = self.pos
        local_smoke = smoke.get_density_at(x, y)
        local_fire = fire.get_density_at(x, y)
        in_smoke = local_smoke > 0.3
        in_fire = local_fire > 0.1

        # ── IDLE: slow ambient movement + wait for danger ──
        # Pre-evacuation period (Proulx 2001): civilians are not yet
        # aware of the emergency, so they move at a CASUAL pace without
        # any destination — modelling normal office activity (walking
        # between desks/bathroom/colleagues). They do NOT detect exits
        # (early return below bypasses exit-spotting logic) but they
        # DO notice fire/smoke/fleeing others through the triggers below,
        # which is how pre-evacuation transitions happen in real life.
        # IDLE civilians are also EXCLUDED from being seen as "fleeing"
        # by others (line ~610 in _see_others_fleeing), so their ambient
        # movement does not cause cascading false alarms.
        if self.state == IDLE:
            # Alert triggers — same as before
            if (in_fire or in_smoke
                    or self._sense_danger(env, smoke, fire)
                    or self._see_others_fleeing(all_civilians)):
                self.state = REACTING
                self._reaction_counter = 0
                return

            # Ambient movement: slow casual pace (0.3× normal speed).
            # Much slower than evacuating civilians (1.5 unit/tick) so the
            # visual distinction remains clear and so IDLE civilians do not
            # accidentally reach an exit cell before realising the danger.
            idle_speed = self.speed * 0.3

            # Occasional direction change — natural-looking ambient motion.
            # 2% chance per tick ≈ change direction every ~50 ticks.
            if self._walk_dir is None or self._rng.random() < 0.02:
                self._pick_random_cardinal()

            # Random walk sets self.vel; _move applies wall collision +
            # separation from other civilians (reused existing logic).
            self._do_random_walk(env, idle_speed, local_smoke)
            self._move(env, idle_speed, all_civilians)
            return

        # ── Rule 5: Potency damage (no recovery) ──
        # Applied BEFORE state-specific logic so that REACTING civilians
        # still take fire/smoke damage. Fire burns you whether you're
        # frozen in panic or not (Kasereka 2018).
        # Fire and smoke damage STACK — both hazards affect the civilian
        # independently. Using elif would produce the perverse result that
        # a cell with mild fire + heavy smoke deals less damage than heavy
        # smoke alone (fire triggers, smoke gets skipped). Additive matches
        # physical reality: heat exposure and asphyxiation are separate
        # injury mechanisms.
        if in_fire:
            self.potency -= 3.0 * local_fire
        if in_smoke:
            self.potency -= local_smoke

        if self.potency <= 0:
            self.state = DEAD
            self.vel = np.zeros(2)
            return

        # ── REACTING: freeze, then move ──
        if self.state == REACTING:
            self._reaction_counter += 1
            if self._reaction_counter >= self._reaction_delay:
                self._start_moving(env)
            return

        # ── Vision & speed ──
        people_vision, exit_vision = self._calc_vision(local_smoke)
        speed = self._calc_speed(in_fire, in_smoke)

        # ── Opportunistic: spot exit (uses exit_vision — small signs, easy to miss) ──
        # FOLLOWING_GUIDE civilians CAN spot exits — if you see an exit door,
        # you go through it. You don't wait for the guide to announce it.
        if exit_vision > 0 and self.state not in (MOVING_TO_EXIT, QUEUED):
            exit_pos = self._spot_exit(env, exit_vision)
            if exit_pos is not None:
                # If in a guide's group, leave the group first
                if self.state == FOLLOWING_GUIDE and self.assigned_guide is not None:
                    try:
                        self.assigned_guide._group.remove(self)
                    except ValueError:
                        pass
                    self.assigned_guide = None
                self._set_exit_target_force(env, exit_pos)

        # ── Rule 4: Herding — follow someone who knows the way ──
        if people_vision > 0 and self.state == WANDERING:
            leader = self._find_goal_leader(all_civilians, people_vision)
            if leader is not None:
                # Found someone heading to exit — follow them
                self._herding_leader = leader
                self.state = FOLLOWING
                self._herding_commitment = 20
            else:
                # No goal-directed person, but check crowd direction as fallback
                crowd_dir = self._sense_crowd_direction(all_civilians, people_vision)
                if crowd_dir is not None:
                    self._walk_dir = crowd_dir
                    self._herding_leader = None
                    self.state = FOLLOWING
                    self._herding_commitment = 20

        # ── State behaviors ──

        if self.state == FLEEING_ROOM:
            # Rule 2: follow BFS path to doorway
            self._do_follow_path(env, speed, fallback=WANDERING)

        elif self.state == MOVING_TO_EXIT:
            # Rule 2: follow BFS path to exit
            if self._target_exit is not None:
                dist = np.linalg.norm(self.pos - np.array(self._target_exit))
                if dist < env.cell_size:
                    # Don't escape instantly — enter queue
                    self.state = QUEUED
                    self.vel = np.zeros(2)
                    return
            self._do_follow_path(env, speed, fallback=WANDERING)

        elif self.state == QUEUED:
            # Waiting at exit — simulation engine will process the queue
            # Still take potency damage while waiting!
            self.vel = np.zeros(2)

        elif self.state == FOLLOWING:
            leader = self._herding_leader

            # Following a specific leader vs crowd direction
            if leader is not None:
                # Leader died, escaped, or we lost line of sight?
                if leader.state in (ESCAPED, DEAD):
                    leader = None
                    self._herding_leader = None
                elif people_vision > 0:
                    d = np.linalg.norm(self.pos - leader.pos)
                    if d > people_vision or not self._has_los(leader.pos, env):
                        leader = None
                        self._herding_leader = None

            if leader is not None:
                # Follow the specific leader (direct vector — not boids)
                diff = leader.pos - self.pos
                dist = np.linalg.norm(diff)
                if dist > 1.0:
                    self.vel = (diff / dist) * speed
            elif self._walk_dir is not None:
                # Fallback: follow crowd direction
                self.vel = self._walk_dir * speed

            # Stuck? Give up
            if self._stuck_ticks >= 8:
                self.state = WANDERING
                self._walk_dir = None
                self._herding_leader = None
                self._herding_commitment = 0
            else:
                self._herding_commitment = max(0, self._herding_commitment - 1)

                if self._herding_commitment > 0:
                    # Still committed — keep following, but check for exits
                    if exit_vision > 0:
                        exit_pos = self._spot_exit(env, exit_vision)
                        if exit_pos is not None:
                            self._set_exit_target(env, exit_pos)
                elif people_vision > 0:
                    # Commitment expired — re-check for leader or crowd
                    new_leader = self._find_goal_leader(all_civilians, people_vision)
                    if new_leader is not None:
                        self._herding_leader = new_leader
                        self._herding_commitment = 20
                    else:
                        crowd_dir = self._sense_crowd_direction(all_civilians, people_vision)
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

                # Give up following if:
                # (a) far from guide for too long (lost in corridors), OR
                # (b) stuck for a long time regardless of distance (pinned
                #     against a wall next to guide — neither can move).
                # Without case (b), pinned civilians could stay in
                # FOLLOWING_GUIDE forever until they die of smoke.
                far_and_stuck = (dist_to_guide > env.cell_size * 20
                                 and self._stuck_ticks >= 8)
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
                    # Hybrid BFS + Boids:
                    # - BFS gives direction (handles walls, doorways)
                    # - Boids separation prevents piling up
                    self.vel = self._hybrid_boids_follow(
                        guide, all_civilians, speed, env)

        elif self.state == WALL_FOLLOWING:
            # Blind in smoke: hand-on-wall
            self._do_wall_follow(env, speed)
            if people_vision > 10 and not in_smoke:
                self.state = WANDERING
                self._walk_dir = None

        elif self.state == WANDERING:
            if in_smoke and people_vision < 5:
                # Blind → wall-follow
                self.state = WALL_FOLLOWING
                self._do_wall_follow(env, speed)
            else:
                # Rule 1: random walk
                # Rule 3: disorientation in smoke
                self._do_random_walk(env, speed, local_smoke)

        # ── Apply movement ──
        self._move(env, speed, all_civilians)

    # ================================================================
    # STATE TRANSITIONS
    # ================================================================

    def _start_moving(self, env):
        """After reaction delay: if guide told us an exit, go there.
        Otherwise find doorway or start wandering."""
        if self._guide_told_exit is not None:
            self._set_exit_target(env, self._guide_told_exit)
            self._guide_told_exit = None
            return

        doorway = self._find_doorway(env)
        if doorway is not None:
            fg = getattr(self, '_fire_grid', None)
            self._waypoints = bfs_path(env, self.pos, doorway, fire_grid=fg)
            self.state = FLEEING_ROOM
        else:
            self.state = WANDERING
            self._pick_random_cardinal()

    def _set_exit_target(self, env, exit_pos):
        """Set exit as goal with BFS path. Won't override FOLLOWING_GUIDE."""
        if self.state == FOLLOWING_GUIDE:
            return
        self._target_exit = exit_pos
        fg = getattr(self, '_fire_grid', None)
        self._waypoints = bfs_path(env, self.pos, np.array(exit_pos), fire_grid=fg)
        self.state = MOVING_TO_EXIT

    def _set_exit_target_force(self, env, exit_pos):
        """Force-set exit target (used when guide releases group)."""
        self._target_exit = exit_pos
        fg = getattr(self, '_fire_grid', None)
        self._waypoints = bfs_path(env, self.pos, np.array(exit_pos), fire_grid=fg)
        self.state = MOVING_TO_EXIT
        self.assigned_guide = None

    # ================================================================
    # MOVEMENT RULES
    # ================================================================

    def _do_random_walk(self, env, speed, smoke_density):
        """Rule 1 + Rule 3: Random walk with disorientation in smoke.
        Pick a cardinal direction, walk until wall. In smoke, random
        chance to change direction (disorientation)."""

        # Rule 3: Disorientation — smoke makes you lose your way
        if smoke_density > 0.2:
            # Higher smoke = more chance to randomly change direction
            if self._rng.random() < smoke_density * 0.3:
                self._pick_random_cardinal()

        # No direction yet? Pick one
        if self._walk_dir is None:
            self._pick_random_cardinal()

        self.vel = self._walk_dir * speed

    def _do_follow_path(self, env, speed, fallback):
        """Rule 2: Follow BFS waypoints."""
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
        """Blind in smoke: hand-on-wall rule."""
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

        # No wall nearby — walk forward
        norm = np.linalg.norm(self.vel)
        if norm > 0:
            self.vel = (self.vel / norm) * speed
        else:
            self._pick_random_cardinal()
            self.vel = self._walk_dir * speed

    def _hybrid_boids_follow(self, guide, all_civilians, speed, env):
        """Hybrid BFS + Boids for following a sweeper guide.

        Adapted from Reynolds' boids (1987) for indoor environments:
          - COHESION: replaced with BFS pathfinding toward guide
            (handles walls, doorways, corridors automatically)
          - ALIGNMENT: match guide's movement direction
          - SEPARATION: push away from nearby group members
            (prevents piling up at doorways and corridors)

        BFS handles navigation. Boids handles group spacing.
        """
        # ── COHESION via BFS (not direct vector) ──
        # Recompute path every 8 ticks. NOTE: we intentionally do NOT also
        # recompute when cache is empty — an empty cache (BFS returned no
        # path) would otherwise cause BFS to run every single tick, which
        # is O(rows*cols) per civilian per tick when the guide is
        # unreachable. Strict 8-tick throttle keeps cost bounded.
        self._boids_bfs_tick += 1
        if self._boids_bfs_tick >= 8:
            self._boids_bfs_tick = 0
            fg = getattr(self, '_fire_grid', None)
            self._boids_bfs_cache = bfs_path(env, self.pos, guide.pos, fire_grid=fg)

        # Follow BFS waypoints
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
            # No path — direct vector as fallback
            diff = guide.pos - self.pos
            d = np.linalg.norm(diff)
            if d > 0:
                cohesion_dir = diff / d

        # ── ALIGNMENT: match guide's direction ──
        gv = np.linalg.norm(guide.vel)
        alignment_dir = guide.vel / gv if gv > 0.1 else np.zeros(2)

        # ── SEPARATION: push away from nearby group members ──
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

        # ── Combine (weights from lecture: alignment*1, cohesion*0.1, separation*2)
        # Adjusted for leader-following: cohesion strongest (follow the guide)
        steer = cohesion_dir * 1.0 + alignment_dir * 0.5 + separation_dir * 1.5

        norm = np.linalg.norm(steer)
        if norm > 0:
            steer = (steer / norm) * speed
        else:
            steer = cohesion_dir * speed

        return steer

    def _move(self, env, speed, all_civilians=None):
        """Apply velocity with wall collision + universal separation.

        Universal separation (Reynolds 1987) applies to ALL civilians
        regardless of state. Prevents overlapping, creates natural
        queueing at exits and doorways."""
        x, y = self.pos

        # ── Universal separation: push away from nearby civilians ──
        # Quick distance check first (manhattan) to skip far civilians early
        if all_civilians is not None:
            sep_force = np.zeros(2)
            sep_r = env.cell_size * 2.0  # 2 cells radius
            px, py = self.pos[0], self.pos[1]
            for other in all_civilians:
                if other is self or other.state in (ESCAPED, DEAD):
                    continue
                # Fast manhattan pre-filter (avoids sqrt for distant civilians)
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

        # Clamp velocity magnitude to prevent separation force pushing
        # civilians through thin walls (Reynolds 1987 — max speed limit)
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

        # Stuck detection
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
        """Pick a random cardinal direction (up/down/left/right)."""
        dirs = [
            np.array([1.0, 0.0]),
            np.array([-1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, -1.0]),
        ]
        self._walk_dir = dirs[self._rng.integers(4)]

    # ================================================================
    # PERCEPTION
    # ================================================================

    def _calc_vision(self, smoke_density):
        """Calculate two vision ranges (Jin 1978, Clark et al. 1985).
        Returns (people_vision, exit_vision).
        Civilians: eyes unprotected, smoke causes irritation + tearing.
        Vision drops rapidly with smoke density."""
        smoke_factor = max(0.15, 1.0 - smoke_density)
        people_v = self.people_vision_radius * smoke_factor
        exit_v = self.exit_vision_radius * smoke_factor
        return people_v, exit_v

    def _calc_speed(self, in_fire, in_smoke):
        has_path = self.state in (FLEEING_ROOM, MOVING_TO_EXIT, FOLLOWING)
        following_guide = self.state == FOLLOWING_GUIDE
        if in_fire:
            return self.speed * 1.8
        elif in_smoke:
            if following_guide:
                # Guide is leading — motivated, move at 85% speed even in smoke
                return self.speed * 0.85
            elif has_path:
                return self.speed * 0.7
            else:
                return self.speed * 0.3
        return self.speed

    def _see_others_fleeing(self, all_civilians):
        """Kasereka 2018: 'when they see a person who evacuates the place,
        they shall run towards the nearest exit door.'
        Must have line of sight — can't see through walls."""
        env = getattr(self, '_env_ref', None)
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
                    return True  # Fallback if env not set yet
        return False

    def _find_goal_leader(self, all_civilians, radius):
        """Find nearest VISIBLE civilian who has a goal (MOVING_TO_EXIT or FLEEING_ROOM).
        These are people who look like they know where they're going.
        In real life: you follow the confident person, not the panicking crowd."""
        env = getattr(self, '_env_ref', None)
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
        """Detect smoke/fire within 1 cell."""
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
        """Line of sight check — can we see the target without walls blocking?"""
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
        """Find nearest exit within vision AND line of sight."""
        best, best_d = None, float("inf")
        for ex, ey in env.exits:
            d = np.hypot(self.pos[0] - ex, self.pos[1] - ey)
            if d < eff_vision and d < best_d:
                if self._has_los(np.array([ex, ey]), env):
                    best, best_d = (ex, ey), d
        return best

    def _sense_crowd_direction(self, all_civilians, radius):
        """Rule 4: Herding — average direction of nearby VISIBLE moving civilians.
        Requires line of sight — can't see through walls (Helbing 2000)."""
        avg_dir = np.zeros(2)
        count = 0

        for other in all_civilians:
            if other is self or other.state in (IDLE, REACTING, ESCAPED, DEAD):
                continue
            d = np.linalg.norm(self.pos - other.pos)
            if d < radius and d > 0:
                other_speed = np.linalg.norm(other.vel)
                if other_speed > 0.5:
                    # Must have line of sight — can't see through walls
                    if self._has_los(other.pos, self._env_ref):
                        avg_dir += other.vel / other_speed
                        count += 1

        if count >= 2:
            norm = np.linalg.norm(avg_dir)
            if norm > 0:
                return avg_dir / norm
        return None

    # ================================================================
    # PATHFINDING
    # ================================================================

    def _find_doorway(self, env):
        """BFS to nearest corridor cell (6+ straight-line walk)."""
        from collections import deque
        gr, gc = env.world_to_grid(self.pos[0], self.pos[1])
        rows, cols = env.grid.shape

        def longest_run(r, c):
            best = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                n, nr, nc = 0, r + dr, c + dc
                while 0 <= nr < rows and 0 <= nc < cols and env.grid[nr, nc] != CellType.WALL:
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


# ════════════════════════════════════════════════════════════════════
# SPAWNING
# ════════════════════════════════════════════════════════════════════

def spawn_civilians(env: Environment, n: int, speed: float = 1.5,
                    seed: int = 99) -> list[Civilian]:
    """Create n civilians at random floor positions."""
    positions = env.spawn_positions(n, seed=seed)
    rng = np.random.default_rng(seed)
    return [Civilian(x, y, speed, rng) for x, y in positions]
