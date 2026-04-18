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

        # If fire_position is set in config, override map's F markers
        fire_pos = getattr(config, 'fire_position', None)
        if fire_pos is not None:
            self.fire_sources.clear()
            pos = self._resolve_fire_position(fire_pos)
            if pos is not None:
                self.fire_sources.append(pos)

    def _resolve_fire_position(self, position: str) -> tuple:
        """Convert named fire position to grid coordinates.
        Finds the nearest FLOOR cell to the target position."""
        margin_r = max(3, self.rows // 6)
        margin_c = max(3, self.cols // 6)

        positions = {
            "center":       (self.rows // 2, self.cols // 2),
            "top_left":     (margin_r, margin_c),
            "top_right":    (margin_r, self.cols - margin_c),
            "bottom_left":  (self.rows - margin_r, margin_c),
            "bottom_right": (self.rows - margin_r, self.cols - margin_c),
        }

        if position not in positions:
            # Fail loudly on typos rather than silently running with no
            # fire (which would corrupt results for that condition).
            raise ValueError(
                f"Unknown fire_position {position!r}. "
                f"Valid options: {sorted(positions.keys())}"
            )

        target_r, target_c = positions[position]

        # Find nearest floor cell to the target
        best = None
        best_dist = float("inf")
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r, c] == CellType.FLOOR:
                    d = abs(r - target_r) + abs(c - target_c)
                    if d < best_dist:
                        best_dist = d
                        best = (r, c)
        return best

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

    def detect_rooms(self) -> list[list[tuple[int, int]]]:
        """Detect enclosed rooms in the map using flood-fill.

        A room = a connected group of floor cells that is enclosed by walls
        on most sides and smaller than a corridor. Corridors are long thin
        areas; rooms are compact.

        Returns list of rooms, each room is a list of (row, col) grid cells.
        """
        rows, cols = self.grid.shape
        visited = np.zeros((rows, cols), dtype=bool)

        # Mark walls and exits as visited so we skip them
        visited[self.grid == CellType.WALL] = True

        rooms = []
        corridors = []

        for r in range(rows):
            for c in range(cols):
                if visited[r, c]:
                    continue

                # Flood-fill to find connected region
                region = []
                queue = [(r, c)]
                visited[r, c] = True

                while queue:
                    cr, cc = queue.pop(0)
                    region.append((cr, cc))

                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                            if self.grid[nr, nc] != CellType.WALL:
                                visited[nr, nc] = True
                                queue.append((nr, nc))

                if len(region) < 4:
                    continue  # Too tiny

                # Classify: room vs corridor
                # Get bounding box
                min_r = min(cell[0] for cell in region)
                max_r = max(cell[0] for cell in region)
                min_c = min(cell[1] for cell in region)
                max_c = max(cell[1] for cell in region)
                height = max_r - min_r + 1
                width = max_c - min_c + 1

                # Corridors are very elongated (aspect ratio > 4) or very large
                aspect = max(width, height) / max(1, min(width, height))
                area = len(region)

                if aspect > 5 or area > 100:
                    corridors.append(region)
                else:
                    rooms.append(region)

        # If no rooms detected (e.g. open plan), treat the whole map
        # as one space — the old random approach will be used
        return rooms

    def spawn_positions(self, n: int, seed: int = 0) -> list[tuple[float, float]]:
        """Spawn n civilians clustered inside rooms.

        Instead of scattering randomly across the whole map:
          1. Detect rooms in the map
          2. Distribute civilians proportionally to room size
          3. Cluster civilians near the center of each room (at desks)
          4. Small random offset so they're not on top of each other

        Falls back to random floor cells if no rooms detected.
        """
        rng = np.random.default_rng(seed)
        rooms = self.detect_rooms()

        # Filter rooms: skip fire-source rooms, skip tiny rooms
        exit_grid = set(self.world_to_grid(ex, ey) for ex, ey in self.exits)
        fire_set = set(self.fire_sources)

        valid_rooms = []
        for room in rooms:
            # Skip rooms that contain fire sources
            has_fire = any((r, c) in fire_set for r, c in room)
            # Skip rooms that contain exits (they're doorways/lobbies)
            has_exit = any((r, c) in exit_grid for r, c in room)
            if not has_fire and not has_exit and len(room) >= 6:
                valid_rooms.append(room)

        if not valid_rooms:
            # Fallback: random floor cells (old approach)
            return self._spawn_random(n, seed)

        # Distribute civilians proportionally to room size
        total_cells = sum(len(room) for room in valid_rooms)
        positions = []

        for room in valid_rooms:
            # How many civilians in this room (proportional to size)
            room_share = len(room) / total_cells
            room_n = max(1, round(n * room_share))

            # Pick cells near the center of the room
            room_arr = np.array(room)
            center_r = room_arr[:, 0].mean()
            center_c = room_arr[:, 1].mean()

            # Sort cells by distance to center (inner cells first)
            dists = np.sqrt((room_arr[:, 0] - center_r) ** 2 +
                            (room_arr[:, 1] - center_c) ** 2)
            sorted_idx = np.argsort(dists)

            # Pick the inner cells (not edge cells near walls)
            inner_cells = [room[sorted_idx[i]]
                           for i in range(min(len(sorted_idx), room_n * 2))]

            # Randomly pick from inner cells
            rng.shuffle(inner_cells)
            chosen = inner_cells[:room_n]

            for r, c in chosen:
                wx, wy = self.grid_to_world(r, c)
                # Small random offset within the cell (not exactly center)
                ox = rng.uniform(-self.cell_size * 0.3, self.cell_size * 0.3)
                oy = rng.uniform(-self.cell_size * 0.3, self.cell_size * 0.3)
                positions.append((wx + ox, wy + oy))

        # If we got too many or too few, adjust
        if len(positions) > n:
            rng.shuffle(positions)
            positions = positions[:n]
        elif len(positions) < n:
            # Fill remaining with random floor cells
            extra = self._spawn_random(n - len(positions), seed + 1000)
            positions.extend(extra)

        return positions[:n]

    def _spawn_random(self, n: int, seed: int) -> list[tuple[float, float]]:
        """Old random spawning as fallback."""
        rng = np.random.default_rng(seed)
        floor_cells = list(zip(*np.where(self.grid == CellType.FLOOR)))

        exit_grid = [self.world_to_grid(ex, ey) for ex, ey in self.exits]
        safe_cells = []
        for r, c in floor_cells:
            too_close = False
            for er, ec in exit_grid:
                if abs(r - er) + abs(c - ec) < 4:
                    too_close = True
                    break
            if (r, c) in self.fire_sources:
                too_close = True
            if not too_close:
                safe_cells.append((r, c))

        n = min(n, len(safe_cells))
        indices = rng.choice(len(safe_cells), size=n, replace=False)

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
