# numpy lets us create and work with 2D arrays (matrices)
import numpy as np

# IntEnum lets us give names to numbers (so we write WALL instead of 1)
from enum import IntEnum

# Import our config class
from config import SimConfig


# This gives friendly names to the numbers we store in the grid
# Instead of remembering "1 means wall", we write CellType.WALL
class CellType(IntEnum):
    FLOOR = 0        # walkable empty space
    WALL = 1         # solid wall, can't walk through
    EXIT = 2         # exit door, civilians escape here
    FIRE_SOURCE = 3  # where fire starts (we'll convert to FLOOR after noting position)


class Environment:
    def __init__(self, config: SimConfig):
        # Store how big each cell is (10 world-units)
        self.cell_size: float = config.cell_size

        # The grid matrix — will be filled by _load_map()
        # This will become a 2D numpy array like:
        #   [[1, 1, 1, 1, 1],
        #    [1, 0, 0, 0, 1],
        #    [1, 0, 0, 2, 1],
        #    [1, 1, 1, 1, 1]]
        self.grid: np.ndarray = np.array([])

        # How many rows and columns the grid has
        self.rows: int = 0
        self.cols: int = 0

        # Total size in world-units (cols * cell_size, rows * cell_size)
        self.width: float = 0.0
        self.height: float = 0.0

        # List of exit positions in world coordinates
        # Example: [(495.0, 35.0), (5.0, 185.0)] — two exit doors
        self.exits: list[tuple[float, float]] = []

        # List of fire source positions in grid coordinates
        # Example: [(3, 28)] — row 3, column 28
        self.fire_sources: list[tuple[int, int]] = []

        # Now actually load the map file and fill everything above
        self._load_map(config.map_file)

    def _load_map(self, path: str) -> None:
        # Dictionary that translates characters from the text file into numbers
        # When we see '.' in the file, store 0 in the grid
        # When we see '#' in the file, store 1 in the grid
        # etc.
        char_map = {
            ".": CellType.FLOOR,       # '.' becomes 0
            "#": CellType.WALL,        # '#' becomes 1
            "E": CellType.EXIT,        # 'E' becomes 2
            "F": CellType.FIRE_SOURCE, # 'F' becomes 3
        }

        # Open the text file and read all lines
        # Example result: ["##########", "#...E....#", "#........#", "##########"]
        with open(path) as f:
            lines = [line.rstrip("\n") for line in f]
            # rstrip("\n") removes the newline character at the end of each line

        # Count how many rows = how many lines in the file
        self.rows = len(lines)  # e.g. 20

        # Count how many columns = length of the longest line
        self.cols = max(len(line) for line in lines)  # e.g. 50

        # Create the grid matrix, filled with WALL (1) by default
        # np.full((20, 50), 1) creates a 20-row, 50-column matrix of all 1s
        # If any line in the text file is shorter, those cells stay as walls
        self.grid = np.full((self.rows, self.cols), CellType.WALL, dtype=int)

        # Now loop through every character in the text file
        # and put the right number in the grid
        for r, line in enumerate(lines):
            # r = row number (0, 1, 2, ...)
            # line = the text for that row, like "#...E....#"
            for c, ch in enumerate(line):
                # c = column number (0, 1, 2, ...)
                # ch = one character, like '#' or '.' or 'E'

                # Look up the character in our dictionary
                # If character not found, default to WALL
                self.grid[r, c] = char_map.get(ch, CellType.WALL)

        # Now scan the grid to find all EXIT cells
        # np.argwhere(condition) returns a list of [row, col] pairs
        # where the condition is True
        for r, c in np.argwhere(self.grid == CellType.EXIT):
            # Convert grid position to world position and save it
            self.exits.append(self.grid_to_world(int(r), int(c)))

        # Find all FIRE_SOURCE cells
        for r, c in np.argwhere(self.grid == CellType.FIRE_SOURCE):
            # Save the grid position
            self.fire_sources.append((int(r), int(c)))

            # IMPORTANT: change the fire cell to FLOOR in the grid
            # The fire cell should still be walkable — people can walk
            # through it. We just remember where the fire is separately.
            self.grid[r, c] = CellType.FLOOR

        # Calculate total world size
        # 50 columns × 10 units per cell = 500 world-units wide
        self.width = self.cols * self.cell_size
        # 20 rows × 10 units per cell = 200 world-units tall
        self.height = self.rows * self.cell_size

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        """Convert a grid cell to the centre point in world coordinates.

        Example: grid cell (row=3, col=5) with cell_size=10
            x = (5 + 0.5) * 10 = 55.0
            y = (3 + 0.5) * 10 = 35.0
            Returns (55.0, 35.0)

        The +0.5 gives us the CENTRE of the cell, not the top-left corner.
        """
        x = (col + 0.5) * self.cell_size
        y = (row + 0.5) * self.cell_size
        return x, y

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Convert a world position back to a grid cell.

        Example: world position (55.0, 35.0) with cell_size=10
            col = int(55.0 / 10) = 5
            row = int(35.0 / 10) = 3
            Returns (3, 5)

        max/min clamps the result so it never goes outside the grid.
        """
        col = max(0, min(int(x / self.cell_size), self.cols - 1))
        row = max(0, min(int(y / self.cell_size), self.rows - 1))
        return row, col

    def spawn_positions(self, n: int, seed: int = 0) -> list[tuple[float, float]]:
        """Pick n random FLOOR cells and return their world-coordinate centres.

        Rules:
          - Only FLOOR cells (not walls, not exits, not fire sources)
          - At least 3 cells away from any exit (so civilians don't start at the door)
          - Each cell is used at most once (no two civilians in the same cell)

        Returns a list of (x, y) world positions, one per civilian.
        """
        rng = np.random.default_rng(seed)

        # Find all floor cells
        floor_cells = list(zip(*np.where(self.grid == CellType.FLOOR)))

        # Remove cells too close to exits (within 3 cells Manhattan distance)
        exit_grid = [self.world_to_grid(ex, ey) for ex, ey in self.exits]
        safe_cells = []
        for r, c in floor_cells:
            too_close = False
            for er, ec in exit_grid:
                if abs(r - er) + abs(c - ec) < 4:
                    too_close = True
                    break
            # Also skip fire source cells
            if (r, c) in self.fire_sources:
                too_close = True
            if not too_close:
                safe_cells.append((r, c))

        # Pick n unique cells at random
        n = min(n, len(safe_cells))
        indices = rng.choice(len(safe_cells), size=n, replace=False)

        # Convert to world coordinates (centre of each cell)
        positions = []
        for idx in indices:
            r, c = safe_cells[idx]
            positions.append(self.grid_to_world(r, c))

        return positions

    def is_walkable(self, x: float, y: float) -> bool:
        """Check if an agent can stand at this world position.

        Returns False if:
          - The position is outside the map
          - The position is inside a wall
        Returns True if it's floor or exit.
        """
        # First check: is it inside the map boundaries?
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False

        # Convert world position to grid cell
        r, c = self.world_to_grid(x, y)

        # Check: is that cell NOT a wall?
        return self.grid[r, c] != CellType.WALL
