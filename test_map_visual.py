import sys
import pygame
from PIL import Image
from config import SimConfig
from evacsim.environment import Environment, CellType
from evacsim.smoke import SmokeSystem
from evacsim.fire import FireSystem


# =====================================================================
# STEP 1: LOAD THE MAP
# =====================================================================

if len(sys.argv) > 1:
    map_file = sys.argv[1]
else:
    map_file = "maps/bottleneck.txt"

config = SimConfig(map_file=map_file)
env = Environment(config)


# =====================================================================
# STEP 2: CALCULATE WINDOW SIZE
# =====================================================================

SCALE = 3
CELL = int(config.cell_size * SCALE)
WIN_W = env.cols * CELL
WIN_H = env.rows * CELL


# =====================================================================
# STEP 3: START PYGAME AND LOAD TILE IMAGES
# =====================================================================

pygame.init()

wall_tile = pygame.transform.scale(pygame.image.load("assets/wall6.png"), (CELL, CELL))
floor_tile = pygame.transform.scale(pygame.image.load("assets/floor3.png"), (CELL, CELL))
door_tile = pygame.transform.scale(pygame.image.load("assets/door2.png"), (CELL, CELL))

TILES = {
    CellType.FLOOR: floor_tile,
    CellType.WALL: wall_tile,
    CellType.EXIT: door_tile,
}


# =====================================================================
# STEP 4: LOAD GIF FRAMES
# =====================================================================

def load_gif_frames(path, size):
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


# =====================================================================
# STEP 5: CREATE FIRE AND SMOKE SYSTEMS (diffusion)
# =====================================================================

smoke_system = SmokeSystem(env, spread_rate=0.02)
fire_system = FireSystem(env, spread_rate=0.005)


# =====================================================================
# STEP 6: CREATE THE WINDOW
# =====================================================================

screen = pygame.display.set_mode((WIN_W, WIN_H))
pygame.display.set_caption(f"Map: {map_file}")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)


# =====================================================================
# STEP 7: PRE-DRAW THE STATIC MAP
# =====================================================================

map_surface = pygame.Surface((WIN_W, WIN_H))

for r in range(env.rows):
    for c in range(env.cols):
        cell_type = env.grid[r, c]
        tile = TILES.get(cell_type, floor_tile)
        map_surface.blit(tile, (c * CELL, r * CELL))


# =====================================================================
# STEP 8: ANIMATION LOOP
# =====================================================================

fire_frame_idx = 0
smoke_frame_idx = 0
frame_timer = 0
FRAME_DELAY = 4
tick = 0

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    # Update
    smoke_system.step()
    fire_system.step()
    tick += 1

    # Draw map
    screen.blit(map_surface, (0, 0))

    # Draw smoke
    smoke_grid = smoke_system.get_density_grid()
    for r in range(env.rows):
        for c in range(env.cols):
            d = smoke_grid[r, c]
            if d > 0.02:
                frame = smoke_frames[smoke_frame_idx].copy()
                alpha = int(min(d, 1.0) * 200)
                frame.set_alpha(alpha)
                screen.blit(frame, (c * CELL, r * CELL))

    # Draw fire
    fire_grid = fire_system.get_density_grid()
    for r in range(env.rows):
        for c in range(env.cols):
            fd = fire_grid[r, c]
            if fd > 0.1:
                frame = fire_frames[fire_frame_idx].copy()
                alpha = int(min(fd, 1.0) * 255)
                frame.set_alpha(alpha)
                screen.blit(frame, (c * CELL, r * CELL))

    # Advance animation
    frame_timer += 1
    if frame_timer >= FRAME_DELAY:
        frame_timer = 0
        fire_frame_idx = (fire_frame_idx + 1) % len(fire_frames)
        smoke_frame_idx = (smoke_frame_idx + 1) % len(smoke_frames)

    # HUD
    smoke_cells = int((smoke_grid > 0.02).sum())
    fire_cells = int((fire_grid > 0.1).sum())
    hud_text = f"Tick: {tick}   Smoke: {smoke_cells}   Fire: {fire_cells}"
    label = font.render(hud_text, True, (255, 255, 255))
    bg = pygame.Surface((label.get_width() + 10, label.get_height() + 4), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 160))
    screen.blit(bg, (5, 5))
    screen.blit(label, (10, 7))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
