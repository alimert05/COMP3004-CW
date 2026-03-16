import sys
import pygame
from PIL import Image
from config import SimConfig
from evacsim.environment import Environment, CellType
from evacsim.smoke import SmokeSystem


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
CELL = int(config.cell_size * SCALE)  # 30 pixels per cell
WIN_W = env.cols * CELL
WIN_H = env.rows * CELL


# =====================================================================
# STEP 3: START PYGAME AND LOAD TILE IMAGES
# =====================================================================

pygame.init()

# Load each tile image and scale it to our cell size (30x30)
wall_tile = pygame.transform.scale(pygame.image.load("assets/wall6.png"), (CELL, CELL))
floor_tile = pygame.transform.scale(
    pygame.image.load("assets/floor3.png"), (CELL, CELL)
)
door_tile = pygame.transform.scale(pygame.image.load("assets/door2.png"), (CELL, CELL))

# Map each cell type to its tile image
TILES = {
    CellType.FLOOR: floor_tile,
    CellType.WALL: wall_tile,
    CellType.EXIT: door_tile,
}


# =====================================================================
# STEP 4: LOAD FIRE GIF FRAMES (4 frames)
# =====================================================================


def load_gif_frames(path, size):
    """Open a GIF, extract each frame, convert to Pygame surface."""
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

print(f"Loaded {len(fire_frames)} fire frames, {len(smoke_frames)} smoke frames")


# =====================================================================
# STEP 5: CREATE THE SMOKE SYSTEM
# =====================================================================

# ── Smoke system ──
# spread_rate=0.03  → 3% chance per tick (slow, realistic)
# wind blows smoke to the right
# start_delay=60    → smoke doesn't spread for the first 60 ticks (evacuation window)
smoke_system = SmokeSystem(
    env,
    spread_rate=0.005,
    seed=42,
    wind_direction="right",
    wind_strength=1.8,
    start_delay=60,
)

# ── Fire spread ──
import numpy as np

fire_density = np.zeros((env.rows, env.cols), dtype=float)
walkable = env.grid != CellType.WALL
fire_sources = list(env.fire_sources)
for r, c in fire_sources:
    fire_density[r, c] = 1.0

FIRE_SPREAD_CHANCE = 0.001  # 0.8% per tick — fire is MUCH slower than smoke
FIRE_START_DELAY = 60  # same delay as smoke before fire starts spreading
FIRE_WIND_BIAS = 1.3  # fire slightly biased rightward too

fire_rng = np.random.default_rng(123)  # different seed from smoke
fire_tick = 0


def spread_fire():
    """Probabilistic fire spread with wind bias — slower than smoke."""
    global fire_density, fire_tick
    fire_tick += 1

    # During delay, fire stays at source only
    if fire_tick <= FIRE_START_DELAY:
        for r, c in fire_sources:
            fire_density[r, c] = 1.0
        return

    d = fire_density
    rows, cols = d.shape
    has_fire = d > 0.0

    # Check each direction with wind bias
    from_above = np.zeros((rows, cols), dtype=bool)
    from_below = np.zeros((rows, cols), dtype=bool)
    from_left = np.zeros((rows, cols), dtype=bool)
    from_right = np.zeros((rows, cols), dtype=bool)

    from_above[1:, :] = has_fire[:-1, :]
    from_below[:-1, :] = has_fire[1:, :]
    from_left[:, 1:] = has_fire[:, :-1]
    from_right[:, :-1] = has_fire[:, 1:]

    rolls = fire_rng.random(size=(rows, cols, 4))
    catches = np.zeros((rows, cols), dtype=bool)
    catches |= from_above & (rolls[:, :, 0] < FIRE_SPREAD_CHANCE)
    catches |= from_below & (rolls[:, :, 1] < FIRE_SPREAD_CHANCE)
    catches |= from_left & (
        rolls[:, :, 2] < FIRE_SPREAD_CHANCE * FIRE_WIND_BIAS
    )  # wind right
    catches |= from_right & (rolls[:, :, 3] < FIRE_SPREAD_CHANCE)

    new_fire = walkable & (~has_fire) & catches
    d[new_fire] = fire_rng.uniform(0.1, 0.3, size=int(new_fire.sum()))

    # Existing fire grows slowly
    growing = (d > 0) & (d < 1.0) & walkable
    growth = fire_rng.uniform(0.003, 0.01, size=(rows, cols))
    d[growing] = np.minimum(1.0, d[growing] + growth[growing])

    for r, c in fire_sources:
        d[r, c] = 1.0
    fire_density = d


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
FRAME_DELAY = 4  # change animation frame every 6 ticks
tick = 0

running = True
while running:
    # --- Handle events ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    # --- Update fire and smoke ---
    spread_fire()  # fire spreads first
    smoke_system.update_fire_cells(fire_density)  # tell smoke where fire is
    smoke_system.step()  # smoke spreads from ALL fire cells
    tick += 1

    # --- Draw the static map ---
    screen.blit(map_surface, (0, 0))

    # --- Draw smoke overlay (using smoke GIF frames) ---
    smoke_grid = smoke_system.get_density_grid()
    for r in range(env.rows):
        for c in range(env.cols):
            d = smoke_grid[r, c]
            if d > 0.02:
                # Pick the smoke frame and set transparency based on density
                frame = smoke_frames[smoke_frame_idx].copy()
                # Higher density = more visible (alpha 30 to 200)
                alpha = int(min(d, 1.0) * 200)
                frame.set_alpha(alpha)
                screen.blit(frame, (c * CELL, r * CELL))

    # --- Draw fire overlay (using fire GIF frames) ---
    for r in range(env.rows):
        for c in range(env.cols):
            fd = fire_density[r, c]
            if fd > 0.1:
                frame = fire_frames[fire_frame_idx].copy()
                alpha = int(min(fd, 1.0) * 255)
                frame.set_alpha(alpha)
                screen.blit(frame, (c * CELL, r * CELL))

    # --- Advance animation frames ---
    frame_timer += 1
    if frame_timer >= FRAME_DELAY:
        frame_timer = 0
        fire_frame_idx = (fire_frame_idx + 1) % len(fire_frames)
        smoke_frame_idx = (smoke_frame_idx + 1) % len(smoke_frames)

    # --- HUD ---
    smoke_cells = int((smoke_grid > 0.02).sum())
    fire_cells = int((fire_density > 0.1).sum())
    hud_text = f"Tick: {tick}   Smoke cells: {smoke_cells}   Fire cells: {fire_cells}"
    label = font.render(hud_text, True, (255, 255, 255))
    bg = pygame.Surface(
        (label.get_width() + 10, label.get_height() + 4), pygame.SRCALPHA
    )
    bg.fill((0, 0, 0, 160))
    screen.blit(bg, (5, 5))
    screen.blit(label, (10, 7))

    # --- Push to screen ---
    pygame.display.flip()
    clock.tick(30)

pygame.quit()
print(f"\nFinished at tick {tick}")
print(f"Smoke cells: {(smoke_grid > 0.02).sum()}")
print(f"Fire cells: {(fire_density > 0.1).sum()}")
