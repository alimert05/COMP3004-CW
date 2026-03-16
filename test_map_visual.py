# sys.argv lets us read command-line arguments
# so we can do: python test_map_visual.py maps/open_plan.txt
import sys

# pygame is the library that opens a window and draws graphics
import pygame

# Import our own classes
from config import SimConfig
from evacsim.environment import Environment, CellType


# =====================================================================
# STEP 1: LOAD THE MAP
# =====================================================================

# Check if user passed a filename as argument
# sys.argv[0] is always the script name itself
# sys.argv[1] would be the first argument the user typed
if len(sys.argv) > 1:
    map_file = sys.argv[1]  # use what the user typed
else:
    map_file = "maps/bottleneck.txt"  # default map

# Create a config object with that map file
config = SimConfig(map_file=map_file)

# Create the environment — this reads the .txt file and builds
# the grid matrix inside it
env = Environment(config)

# At this point:
#   env.grid    = 2D numpy array of integers (0=floor, 1=wall, 2=exit)
#   env.rows    = 20
#   env.cols    = 50
#   env.exits   = [(x1,y1), (x2,y2)] list of exit world positions
#   env.fire_sources = [(r,c)] list of fire grid positions


# =====================================================================
# STEP 2: CALCULATE WINDOW SIZE
# =====================================================================

# SCALE makes everything bigger on screen
# cell_size=10 means each cell is 10 world-units
# SCALE=3 means we draw each world-unit as 3 pixels
# So each cell becomes 10 * 3 = 30 pixels on screen
SCALE = 3
CELL = int(config.cell_size * SCALE)  # 10 * 3 = 30 pixels per cell

# Total window size in pixels
WIN_W = env.cols * CELL  # 50 cells * 30 pixels = 1500 pixels wide
WIN_H = env.rows * CELL  # 20 cells * 30 pixels = 600 pixels tall


# =====================================================================
# STEP 3: DEFINE COLOURS
# =====================================================================

# Each colour is (Red, Green, Blue) from 0-255
COLOURS = {
    CellType.FLOOR: (220, 220, 220),  # light grey for walkable floor
    CellType.WALL: (40, 40, 40),  # dark grey for walls
    CellType.EXIT: (0, 200, 0),  # green for exits
}
FIRE_COLOUR = (255, 70, 0)  # orange for fire source


# =====================================================================
# STEP 4: CREATE THE PYGAME WINDOW
# =====================================================================

# Start up pygame (must be called before anything else)
pygame.init()

# Create a window of size WIN_W x WIN_H pixels
screen = pygame.display.set_mode((WIN_W, WIN_H))

# Set the text in the title bar
pygame.display.set_caption(f"Map: {map_file}")


# =====================================================================
# STEP 5: DRAW THE MAP
# =====================================================================

# Loop through every cell in the grid
for r in range(env.rows):  # r goes 0, 1, 2, ... 19
    for c in range(env.cols):  # c goes 0, 1, 2, ... 49

        # Look up what colour this cell type should be
        # env.grid[r, c] gives us 0, 1, or 2
        # COLOURS.get() finds the matching colour
        # If not found, default to light grey
        colour = COLOURS.get(env.grid[r, c], (220, 220, 220))

        # Draw a filled rectangle on the screen
        # Arguments: surface, colour, (x, y, width, height)
        # x position = column * cell pixel size
        # y position = row * cell pixel size
        pygame.draw.rect(screen, colour, (c * CELL, r * CELL, CELL, CELL))

# Draw fire sources on top (orange squares)
# We draw these separately because fire cells were converted
# to FLOOR in the grid, so they'd appear grey otherwise
for fr, fc in env.fire_sources:
    pygame.draw.rect(screen, FIRE_COLOUR, (fc * CELL, fr * CELL, CELL, CELL))

# Draw thin grid lines between cells
# Horizontal lines (one per row)
for r in range(env.rows + 1):
    # Draw a line from left edge to right edge at this row's y position
    pygame.draw.line(
        screen,
        (180, 180, 180),
        (0, r * CELL),  # start point (x=0, y=row position)
        (WIN_W, r * CELL),
    )  # end point (x=window width, y=same)

# Vertical lines (one per column)
for c in range(env.cols + 1):
    pygame.draw.line(
        screen, (180, 180, 180), (c * CELL, 0), (c * CELL, WIN_H)  # start point
    )  # end point


# =====================================================================
# STEP 6: SHOW EVERYTHING ON SCREEN
# =====================================================================

# Everything we drew above was to a hidden buffer
# flip() pushes it all to the actual screen at once
# Without this, the window would stay black
pygame.display.flip()


# =====================================================================
# STEP 7: WAIT FOR USER TO CLOSE THE WINDOW
# =====================================================================

# This loop keeps the window open until the user closes it
# Without this loop, the window would flash and immediately disappear
running = True
while running:
    # pygame.event.get() returns a list of things that happened
    # (mouse clicks, key presses, window close, etc.)
    for event in pygame.event.get():

        # Did the user click the X button to close the window?
        if event.type == pygame.QUIT:
            running = False

        # Did the user press the Escape key?
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

# Clean up pygame and close the window
pygame.quit()

# Print a summary to the terminal
print(f"Map loaded: {env.rows} rows x {env.cols} cols")
print(f"Exits: {len(env.exits)}  Fire sources: {len(env.fire_sources)}")
