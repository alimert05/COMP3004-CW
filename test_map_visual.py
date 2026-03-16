import sys
import pygame
from PIL import Image
from config import SimConfig
from evacsim.environment import Environment, CellType


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
# STEP 3: LOAD TILE IMAGES
# =====================================================================

# Start pygame first (needed before loading images)
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
# STEP 4: LOAD THE FIRE GIF FRAMES
# =====================================================================

fire_frames = []
gif = Image.open("assets/fire.gif")

for frame_num in range(gif.n_frames):
    gif.seek(frame_num)
    frame_rgba = gif.convert("RGBA")
    raw_data = frame_rgba.tobytes()
    frame_surface = pygame.image.fromstring(raw_data, frame_rgba.size, "RGBA")
    frame_surface = pygame.transform.scale(frame_surface, (CELL, CELL))
    fire_frames.append(frame_surface)

print(f"Loaded tiles and {len(fire_frames)} fire frames")


# =====================================================================
# STEP 5: CREATE THE PYGAME WINDOW
# =====================================================================

screen = pygame.display.set_mode((WIN_W, WIN_H))
pygame.display.set_caption(f"Map: {map_file}")
clock = pygame.time.Clock()


# =====================================================================
# STEP 6: PRE-DRAW THE STATIC MAP USING TILE IMAGES
# =====================================================================

# Draw tiles once onto a separate surface (fast to copy each frame)
map_surface = pygame.Surface((WIN_W, WIN_H))

for r in range(env.rows):
    for c in range(env.cols):
        cell_type = env.grid[r, c]
        tile = TILES.get(cell_type, floor_tile)  # default to floor if unknown
        map_surface.blit(tile, (c * CELL, r * CELL))


# =====================================================================
# STEP 7: ANIMATION LOOP
# =====================================================================

frame_index = 0
frame_timer = 0
FRAME_DELAY = 6

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    # --- Draw the static map (tiles) ---
    screen.blit(map_surface, (0, 0))

    # --- Draw fire GIF at every fire source position ---
    for fr, fc in env.fire_sources:
        fx = fc * CELL
        fy = fr * CELL
        screen.blit(fire_frames[frame_index], (fx, fy))

    # --- Advance the fire animation ---
    frame_timer += 1
    if frame_timer >= FRAME_DELAY:
        frame_timer = 0
        frame_index = (frame_index + 1) % len(fire_frames)

    # --- Push to screen ---
    pygame.display.flip()
    clock.tick(15)

pygame.quit()
print(f"Map loaded: {env.rows} rows x {env.cols} cols")
print(f"Exits: {len(env.exits)}  Fire sources: {len(env.fire_sources)}")
