"""Pygame visual mode for the simulation.

Loads a map and runs the simulation in real time with a Pygame window.
Civilians are coloured by state, fire and smoke are animated GIFs,
guides are drawn as orange diamonds with a health bar above each.

Keyboard controls:
    1 / 2 / 5 / 0    Set simulation speed multiplier (1x, 2x, 5x, 10x).
    ESC              Quit.
"""

import sys
import pygame
from PIL import Image
from config import (
    SimConfig,
    VISUAL_CIVILIAN_NUMBER,
    VISUAL_GUIDES_ENABLED,
    VISUAL_GUIDE_POLICY,
    VISUAL_GUIDE_COUNT,
    VISUAL_FIRE_POSITION,
)
from evacsim.environment import Environment, CellType
from evacsim.smoke import SmokeSystem
from evacsim.fire import FireSystem
from evacsim.civilian import (
    spawn_civilians,
    IDLE,
    REACTING,
    FLEEING_ROOM,
    WANDERING,
    WALL_FOLLOWING,
    FOLLOWING,
    FOLLOWING_GUIDE,
    MOVING_TO_EXIT,
    QUEUED,
    ESCAPED,
    DEAD,
)
from evacsim.guide import spawn_beacon_guides, spawn_sweeper_guides, spawn_router_guides

# --------------------------------------------------------------------
# Step 1: load the map
# --------------------------------------------------------------------

if len(sys.argv) > 1:
    map_file = sys.argv[1]
else:
    map_file = "maps/bottleneck_A.txt"

config = SimConfig(map_file=map_file, fire_position=VISUAL_FIRE_POSITION)
env = Environment(config)


# --------------------------------------------------------------------
# Step 2: auto-scale window to fit the screen
# --------------------------------------------------------------------

pygame.init()

# Query the desktop size so the window fits on smaller screens.
# Leave a 10% width margin and a 15% height margin to allow for the
# taskbar and window chrome.
display_info = pygame.display.Info()
max_w = int(display_info.current_w * 0.95)
max_h = int(display_info.current_h * 0.9)

# Pick the largest integer SCALE that keeps the window within the
# available area. Falls back to 1 on very small screens.
base_w = env.cols * config.cell_size
base_h = env.rows * config.cell_size
SCALE = max(1, int(min(max_w / base_w, max_h / base_h)))

CELL = int(config.cell_size * SCALE)
WIN_W = env.cols * CELL
WIN_H = env.rows * CELL


# --------------------------------------------------------------------
# Step 3: load tile images
# --------------------------------------------------------------------

wall_tile = pygame.transform.scale(pygame.image.load("assets/wall6.png"), (CELL, CELL))
floor_tile = pygame.transform.scale(
    pygame.image.load("assets/floor3.png"), (CELL, CELL)
)
door_tile = pygame.transform.scale(pygame.image.load("assets/door2.png"), (CELL, CELL))

TILES = {
    CellType.FLOOR: floor_tile,
    CellType.WALL: wall_tile,
    CellType.EXIT: door_tile,
}


# --------------------------------------------------------------------
# Step 4: load animated GIF frames
# --------------------------------------------------------------------


def load_gif_frames(path, size):
    """Load every frame of a GIF, scale to size, return as Pygame surfaces."""
    frames = []
    gif = Image.open(path)
    for i in range(gif.n_frames):
        gif.seek(i)
        rgba = gif.convert("RGBA")
        raw = rgba.tobytes()
        surf = pygame.image.fromstring(raw, rgba.size, "RGBA")
        surf = pygame.transform.scale(surf, size)
        frames.append(surf)
    return frames


fire_frames = load_gif_frames("assets/fire.gif", (CELL, CELL))
smoke_frames = load_gif_frames("assets/smoke gif.gif", (CELL, CELL))


# --------------------------------------------------------------------
# Step 5: build fire and smoke systems
# --------------------------------------------------------------------

smoke_system = SmokeSystem(env, spread_rate=0.02)
fire_system = FireSystem(env, spread_rate=0.005)
# Fire-and-smoke coupling: smoke is sourced from fire cells.
smoke_system._fire = fire_system


# --------------------------------------------------------------------
# Step 5b: spawn civilians and guides
# --------------------------------------------------------------------

civilians = spawn_civilians(env, n=VISUAL_CIVILIAN_NUMBER, speed=1.5, seed=99)

guides = []
if VISUAL_GUIDES_ENABLED:
    n = VISUAL_GUIDE_COUNT if VISUAL_GUIDE_COUNT is not None else len(env.exits)
    if VISUAL_GUIDE_POLICY == "beacon":
        guides = spawn_beacon_guides(env, n_guides=n, speed=2.0, seed=77)
    elif VISUAL_GUIDE_POLICY == "sweeper":
        guides = spawn_sweeper_guides(
            env, n_guides=n, speed=2.0, seed=77, n_civilians=VISUAL_CIVILIAN_NUMBER
        )
    elif VISUAL_GUIDE_POLICY == "router":
        guides = spawn_router_guides(
            env, n_guides=n, speed=2.0, seed=77, n_civilians=VISUAL_CIVILIAN_NUMBER
        )

# Drawing radii in pixels.
CIV_RADIUS = max(4, CELL // 4)
GUIDE_RADIUS = max(6, CELL // 3)


# --------------------------------------------------------------------
# Step 6: create the window
# --------------------------------------------------------------------

screen = pygame.display.set_mode((WIN_W, WIN_H))
pygame.display.set_caption(f"Map: {map_file}")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)


# --------------------------------------------------------------------
# Step 7: pre-draw the static map onto an offscreen surface
# --------------------------------------------------------------------

map_surface = pygame.Surface((WIN_W, WIN_H))

for r in range(env.rows):
    for c in range(env.cols):
        cell_type = env.grid[r, c]
        tile = TILES.get(cell_type, floor_tile)
        map_surface.blit(tile, (c * CELL, r * CELL))


# --------------------------------------------------------------------
# Step 8: animation loop
# --------------------------------------------------------------------

fire_frame_idx = 0
smoke_frame_idx = 0
frame_timer = 0
FRAME_DELAY = 4
tick = 0
sim_speed = 1  # 1x, 2x, 5x, 10x. Press 1 / 2 / 5 / 0 to change.

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_1:
                sim_speed = 1
            elif event.key == pygame.K_2:
                sim_speed = 2
            elif event.key == pygame.K_5:
                sim_speed = 5
            elif event.key == pygame.K_0:
                sim_speed = 10

    # Run several simulation steps per frame to support speed-up.
    for _ in range(sim_speed):
        smoke_system.step()
        fire_system.step()
        tick += 1
        # Update guides first (they may broadcast to civilians).
        for guide in guides:
            guide.update(env, smoke_system, fire_system, civilians)
        for civ in civilians:
            civ.update(env, smoke_system, fire_system, civilians)

        # Exit-queue processing: at most two civilians per exit per tick.
        import numpy as np

        EXIT_THROUGHPUT = 2
        exit_queues = {}
        for i, civ in enumerate(civilians):
            if civ.state != QUEUED:
                continue
            if civ._target_exit is None:
                civ.state = ESCAPED
                continue
            best_exit, best_d = None, float("inf")
            for ex, ey in env.exits:
                d = np.hypot(civ.pos[0] - ex, civ.pos[1] - ey)
                if d < best_d:
                    best_d = d
                    best_exit = (ex, ey)
            if best_exit:
                exit_queues.setdefault(best_exit, []).append((best_d, i))
        for q in exit_queues.values():
            q.sort(key=lambda x: x[0])
            for rank, (_, idx) in enumerate(q):
                if rank < EXIT_THROUGHPUT:
                    civilians[idx].state = ESCAPED

    # Draw the static map.
    screen.blit(map_surface, (0, 0))

    # Draw smoke.
    smoke_grid = smoke_system.get_density_grid()
    for r in range(env.rows):
        for c in range(env.cols):
            d = smoke_grid[r, c]
            if d > 0.02:
                frame = smoke_frames[smoke_frame_idx].copy()
                alpha = int(min(d, 1.0) * 200)
                frame.set_alpha(alpha)
                screen.blit(frame, (c * CELL, r * CELL))

    # Draw fire.
    fire_grid = fire_system.get_density_grid()
    for r in range(env.rows):
        for c in range(env.cols):
            fd = fire_grid[r, c]
            if fd > 0.1:
                frame = fire_frames[fire_frame_idx].copy()
                alpha = int(min(fd, 1.0) * 255)
                frame.set_alpha(alpha)
                screen.blit(frame, (c * CELL, r * CELL))

    # Draw civilians, coloured by state. Skip escaped civilians.
    for civ in civilians:
        if civ.state == ESCAPED:
            continue
        sx = int(civ.pos[0] * SCALE)
        sy = int(civ.pos[1] * SCALE)
        if civ.state == DEAD:
            colour = (200, 30, 30)  # Red: dead.
        elif civ.state == QUEUED:
            colour = (255, 255, 255)  # White: waiting in the exit queue.
        elif civ.state == MOVING_TO_EXIT:
            colour = (50, 220, 50)  # Green: heading to an exit.
        elif civ.state == FLEEING_ROOM:
            colour = (255, 150, 50)  # Orange: fleeing the current room.
        elif civ.state == FOLLOWING_GUIDE:
            colour = (0, 200, 200)  # Cyan: in a guide's group.
        elif civ.state == FOLLOWING:
            colour = (255, 200, 50)  # Yellow: herding (following crowd).
        elif civ.state == WALL_FOLLOWING:
            colour = (200, 100, 255)  # Purple: blind, hand-on-wall.
        elif civ.state == REACTING:
            colour = (255, 255, 150)  # Light yellow: noticed danger, processing.
        elif civ.state == IDLE:
            colour = (180, 180, 180)  # Grey: at desk, unaware.
        else:
            colour = (50, 120, 255)  # Blue: wandering.
        pygame.draw.circle(screen, colour, (sx, sy), CIV_RADIUS)
        pygame.draw.circle(screen, (0, 0, 0), (sx, sy), CIV_RADIUS, 1)

    # Draw guides as diamonds. Orange when on duty, red when self-evacuating.
    for guide in guides:
        if not guide.alive:
            continue
        gx = int(guide.pos[0] * SCALE)
        gy = int(guide.pos[1] * SCALE)
        r = GUIDE_RADIUS

        if guide._self_evacuating:
            fill_colour = (220, 50, 50)  # Red: escaping.
        else:
            fill_colour = (255, 140, 0)  # Orange: on duty.

        # Diamond shape.
        points = [(gx, gy - r), (gx + r, gy), (gx, gy + r), (gx - r, gy)]
        pygame.draw.polygon(screen, fill_colour, points)
        pygame.draw.polygon(screen, (0, 0, 0), points, 2)

        # Health bar above the guide.
        bar_w = CELL // 2
        bar_h = 3
        bar_x = gx - bar_w // 2
        bar_y = gy - r - 5
        health = guide.health
        pygame.draw.rect(screen, (60, 60, 60), (bar_x, bar_y, bar_w, bar_h))
        if health > 0.6:
            bar_col = (50, 200, 50)
        elif health > 0.3:
            bar_col = (255, 200, 50)
        else:
            bar_col = (220, 50, 50)
        pygame.draw.rect(screen, bar_col, (bar_x, bar_y, int(bar_w * health), bar_h))

        # Radius circle, different visual for each policy type.
        if not guide._self_evacuating:
            from evacsim.guide import BeaconGuide

            if isinstance(guide, BeaconGuide):
                # Beacon: broadcast radius (sound, passes through walls).
                brad = int(guide.get_broadcast_radius(smoke_system) * SCALE)
                broadcast_surf = pygame.Surface((brad * 2, brad * 2), pygame.SRCALPHA)
                pygame.draw.circle(
                    broadcast_surf, (255, 140, 0, 30), (brad, brad), brad
                )
                screen.blit(broadcast_surf, (gx - brad, gy - brad))
            else:
                # Sweeper or Router: visual recruit radius (smaller, requires LOS).
                vrad = int(guide.get_visual_recruit_radius(smoke_system) * SCALE)
                recruit_surf = pygame.Surface((vrad * 2, vrad * 2), pygame.SRCALPHA)
                pygame.draw.circle(
                    recruit_surf, (100, 200, 255, 25), (vrad, vrad), vrad
                )
                screen.blit(recruit_surf, (gx - vrad, gy - vrad))

    # Advance GIF animation frames.
    frame_timer += 1
    if frame_timer >= FRAME_DELAY:
        frame_timer = 0
        fire_frame_idx = (fire_frame_idx + 1) % len(fire_frames)
        smoke_frame_idx = (smoke_frame_idx + 1) % len(smoke_frames)

    # HUD: tick counter, speed multiplier, and per-state counts.
    smoke_cells = int((smoke_grid > 0.02).sum())
    fire_cells = int((fire_grid > 0.1).sum())
    idle = sum(1 for c in civilians if c.state == IDLE)
    reacting = sum(1 for c in civilians if c.state == REACTING)
    fleeing = sum(1 for c in civilians if c.state == FLEEING_ROOM)
    wandering = sum(1 for c in civilians if c.state == WANDERING)
    wall_f = sum(1 for c in civilians if c.state == WALL_FOLLOWING)
    following = sum(1 for c in civilians if c.state == FOLLOWING)
    to_exit = sum(1 for c in civilians if c.state == MOVING_TO_EXIT)
    escaped = sum(1 for c in civilians if c.state == ESCAPED)
    dead = sum(1 for c in civilians if c.state == DEAD)
    guides_alive = sum(1 for g in guides if g.alive)
    policy_label = VISUAL_GUIDE_POLICY.upper() if VISUAL_GUIDES_ENABLED else "NONE"
    alive_civs = [c for c in civilians if c.state not in (ESCAPED, DEAD)]
    avg_potency = sum(c.potency for c in alive_civs) / max(1, len(alive_civs))
    in_group = sum(1 for c in civilians if c.state == FOLLOWING_GUIDE)
    queued = sum(1 for c in civilians if c.state == QUEUED)
    hud_text = f"T:{tick}  {sim_speed}x  Policy:{policy_label}  Guides:{guides_alive}  Idle:{idle}  Flee:{fleeing}  Wander:{wandering}  Follow:{following}  Group:{in_group}  Exit:{to_exit}  Queue:{queued}  Escaped:{escaped}  Dead:{dead}  HP:{avg_potency:.0f}"
    label = font.render(hud_text, True, (255, 255, 255))
    bg = pygame.Surface(
        (label.get_width() + 10, label.get_height() + 4), pygame.SRCALPHA
    )
    bg.fill((0, 0, 0, 160))
    screen.blit(bg, (5, 5))
    screen.blit(label, (10, 7))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
