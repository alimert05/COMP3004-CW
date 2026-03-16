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
# STEP 3: DEFINE COLOURS
# =====================================================================

COLOURS = {
    CellType.FLOOR: (220, 220, 220),
    CellType.WALL: (40, 40, 40),
    CellType.EXIT: (0, 200, 0),
}


# =====================================================================
# STEP 4: LOAD THE FIRE GIF FRAMES
# =====================================================================

# Pygame can't animate GIFs on its own, so we use PIL (Pillow)
# to extract each frame, then convert to Pygame surfaces

fire_frames = []  # will hold each frame as a Pygame surface

# Open the GIF with PIL
gif = Image.open("assets/fire.gif")

# Loop through every frame in the GIF
for frame_num in range(gif.n_frames):
    # Seek to this frame
    gif.seek(frame_num)

    # Convert to RGBA (so transparency works)
    frame_rgba = gif.convert("RGBA")

    # Convert PIL image → raw bytes → Pygame surface
    raw_data = frame_rgba.tobytes()
    frame_surface = pygame.image.fromstring(
        raw_data, frame_rgba.size, "RGBA"
    )

    # Scale the frame to fit our cell size (30x30 pixels)
    frame_surface = pygame.transform.scale(frame_surface, (CELL, CELL))

    fire_frames.append(frame_surface)

print(f"Loaded {len(fire_frames)} fire frames, each scaled to {CELL}x{CELL}")


# =====================================================================
# STEP 5: CREATE THE PYGAME WINDOW
# =====================================================================

pygame.init()
screen = pygame.display.set_mode((WIN_W, WIN_H))
pygame.display.set_caption(f"Map: {map_file}")
clock = pygame.time.Clock()


# =====================================================================
# STEP 6: PRE-DRAW THE STATIC MAP (walls, floor, exits)
# =====================================================================

map_surface = pygame.Surface((WIN_W, WIN_H))

for r in range(env.rows):
    for c in range(env.cols):
        colour = COLOURS.get(env.grid[r, c], (220, 220, 220))
        pygame.draw.rect(map_surface, colour, (c * CELL, r * CELL, CELL, CELL))

# Grid lines
for r in range(env.rows + 1):
    pygame.draw.line(map_surface, (180, 180, 180), (0, r * CELL), (WIN_W, r * CELL))
for c in range(env.cols + 1):
    pygame.draw.line(map_surface, (180, 180, 180), (c * CELL, 0), (c * CELL, WIN_H))


# =====================================================================
# STEP 7: ANIMATION LOOP
# =====================================================================

frame_index = 0      # which fire frame we're showing right now
frame_timer = 0      # counts ticks to slow down the animation
FRAME_DELAY = 6      # show each fire frame for 6 ticks (~0.4 seconds at 15fps)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    # --- Draw the static map ---
    screen.blit(map_surface, (0, 0))

    # --- Draw fire GIF at every fire source position ---
    for fr, fc in env.fire_sources:
        fx = fc * CELL   # pixel x
        fy = fr * CELL   # pixel y
        screen.blit(fire_frames[frame_index], (fx, fy))

    # --- Advance the fire animation ---
    frame_timer += 1
    if frame_timer >= FRAME_DELAY:
        frame_timer = 0
        # Move to next frame, loop back to 0 when we reach the end
        frame_index = (frame_index + 1) % len(fire_frames)

    # --- Push to screen ---
    pygame.display.flip()
    clock.tick(15)

pygame.quit()
print(f"Map loaded: {env.rows} rows x {env.cols} cols")
print(f"Exits: {len(env.exits)}  Fire sources: {len(env.fire_sources)}")
