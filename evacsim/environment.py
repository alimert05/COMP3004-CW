"""Environment: grid loader, cell types, room detection, and spawning.

The environment owns the static layout of the building. It loads a
plain-text map, classifies cells (FLOOR, WALL, EXIT, FIRE_SOURCE),
provides world-to-grid coordinate conversion, detects rooms via
flood-fill, and spawns civilians at clustered floor positions.
"""

import numpy as np
from enum import IntEnum

from config import SimConfig


class CellType(IntEnum):
    """Symbolic constants for cell classification on the grid."""
    FLOOR = 0        # Walkable empty space.
    WALL = 1         # Solid wall, impassable.
    EXIT = 2         # Exit cell; civilians escape on contact.
    FIRE_SOURCE = 3  # Fire origin; converted to FLOOR after position is recorded.


class Environment:
    """Static building layout, walkability, and spawning.

    Loaded once per simulation from a text-format map file.
    """

    def __init__(self, config: SimConfig):
        self.cell_size: float = config.cell_size

        # Grid matrix populated by _load_map. Stored as a 2D numpy
        # array of CellType integers, with shape (rows, cols).
        self.grid: np.ndarray = np.array([])

        self.rows: int = 0
        self.cols: int = 0

        # Total dimensions in world units (cols * cell_size, rows * cell_size).
        self.width: float = 0.0
        self.height: float = 0.0

        # Exit positions in world coordinates,
        # for example [(495.0, 35.0), (5.0, 185.0)].
        self.exits: list[tuple[float, float]] = []

        # Fire-source positions in grid coordinates.
        self.fire_sources: list[tuple[int, int]] = []

        # Load the map and populate every field above.
        self._load_map(config.map_file)

        # If a fire_position is set in config, override the F markers
        # from the map file with a single configured source.
        fire_pos = getattr(config, 'fire_position', None)
        if fire_pos is not None:
            self.fire_sources.clear()
            pos = self._resolve_fire_position(fire_pos)
            if pos is not None:
                self.fire_sources.append(pos)

    def _resolve_fire_position(self, position: str) -> tuple:
        """Convert a named fire position to grid coordinates.

        Returns the nearest FLOOR cell to the named target so that the
        fire is always placed on a walkable cell rather than a wall.
        """
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
            # Fail loudly on a typo rather than silently running with
            # no fire, which would corrupt that experimental cell.
            raise ValueError(
                f"Unknown fire_position {position!r}. "
                f"Valid options: {sorted(positions.keys())}"
            )

        target_r, target_c = positions[position]

        # Find the nearest floor cell to the target using Manhattan
        # distance.
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
        """Parse a text map file into the grid, exit list, and fire-source list."""
        # Character-to-cell-type translation.
        char_map = {
            ".": CellType.FLOOR,
            "#": CellType.WALL,
            "E": CellType.EXIT,
            "F": CellType.FIRE_SOURCE,
        }

        with open(path) as f:
            lines = [line.rstrip("\n") for line in f]

        self.rows = len(lines)
        self.cols = max(len(line) for line in lines)

        # Default-fill with walls; any line shorter than the maximum
        # width keeps the missing cells as walls.
        self.grid = np.full((self.rows, self.cols), CellType.WALL, dtype=int)

        for r, line in enumerate(lines):
            for c, ch in enumerate(line):
                self.grid[r, c] = char_map.get(ch, CellType.WALL)

        # Collect exit cells in world coordinates.
        for r, c in np.argwhere(self.grid == CellType.EXIT):
            self.exits.append(self.grid_to_world(int(r), int(c)))

        # Collect fire-source cells; convert them back to FLOOR on the
        # grid so they remain walkable (the source is tracked
        # separately).
        for r, c in np.argwhere(self.grid == CellType.FIRE_SOURCE):
            self.fire_sources.append((int(r), int(c)))
            self.grid[r, c] = CellType.FLOOR

        self.width = self.cols * self.cell_size
        self.height = self.rows * self.cell_size

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        """Convert a grid cell to its centre point in world coordinates.

        With cell_size 10 and (row, col) = (3, 5):
            x = (5 + 0.5) * 10 = 55.0
            y = (3 + 0.5) * 10 = 35.0
        The 0.5 offset places the result at the centre of the cell.
        """
        x = (col + 0.5) * self.cell_size
        y = (row + 0.5) * self.cell_size
        return x, y

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Convert a world position to the grid cell containing it.

        Output is clamped so it never goes outside the grid.
        """
        col = max(0, min(int(x / self.cell_size), self.cols - 1))
        row = max(0, min(int(y / self.cell_size), self.rows - 1))
        return row, col

    def detect_rooms(self) -> list[list[tuple[int, int]]]:
        """Detect enclosed rooms by flood-filling connected floor regions.

        A region is classified as a room if it is compact (small area
        and aspect ratio at most 5) and as a corridor otherwise.
        Returns the list of rooms only; corridors are discarded.
        """
        rows, cols = self.grid.shape
        visited = np.zeros((rows, cols), dtype=bool)

        # Walls and exits are treated as visited so the search skips them.
        visited[self.grid == CellType.WALL] = True

        rooms = []
        corridors = []

        for r in range(rows):
            for c in range(cols):
                if visited[r, c]:
                    continue

                # Flood-fill the connected region containing (r, c).
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
                    continue  # Too tiny to count as a room.

                # Classify by bounding box and area.
                min_r = min(cell[0] for cell in region)
                max_r = max(cell[0] for cell in region)
                min_c = min(cell[1] for cell in region)
                max_c = max(cell[1] for cell in region)
                height = max_r - min_r + 1
                width = max_c - min_c + 1

                # Corridors are very elongated (aspect ratio over 5) or
                # very large (area over 100).
                aspect = max(width, height) / max(1, min(width, height))
                area = len(region)

                if aspect > 5 or area > 100:
                    corridors.append(region)
                else:
                    rooms.append(region)

        return rooms

    def spawn_positions(self, n: int, seed: int = 0) -> list[tuple[float, float]]:
        """Spawn n civilians clustered inside detected rooms.

        Steps:
            1. Detect rooms via flood-fill.
            2. Distribute civilians proportionally to room size.
            3. Place each civilian near the centre of its assigned room
               (closer to the centroid than the walls).
            4. Add a small random offset so civilians do not stack on
               top of each other.

        Falls back to random floor cells when no rooms are detected
        (for example on open-plan maps).
        """
        rng = np.random.default_rng(seed)
        rooms = self.detect_rooms()

        # Filter rooms: skip any that contain a fire source or an exit
        # (the latter would be a doorway or lobby, not a habitable room),
        # and skip rooms with fewer than six cells.
        exit_grid = set(self.world_to_grid(ex, ey) for ex, ey in self.exits)
        fire_set = set(self.fire_sources)

        valid_rooms = []
        for room in rooms:
            has_fire = any((r, c) in fire_set for r, c in room)
            has_exit = any((r, c) in exit_grid for r, c in room)
            if not has_fire and not has_exit and len(room) >= 6:
                valid_rooms.append(room)

        if not valid_rooms:
            return self._spawn_random(n, seed)

        total_cells = sum(len(room) for room in valid_rooms)
        positions = []

        for room in valid_rooms:
            # Number of civilians to place in this room, proportional
            # to its area.
            room_share = len(room) / total_cells
            room_n = max(1, round(n * room_share))

            # Sort cells by distance to the room centroid so inner
            # cells are picked before edge cells (avoids placing
            # civilians right next to walls).
            room_arr = np.array(room)
            center_r = room_arr[:, 0].mean()
            center_c = room_arr[:, 1].mean()
            dists = np.sqrt((room_arr[:, 0] - center_r) ** 2 +
                            (room_arr[:, 1] - center_c) ** 2)
            sorted_idx = np.argsort(dists)

            # Take the inner cells, then shuffle and pick room_n.
            inner_cells = [room[sorted_idx[i]]
                           for i in range(min(len(sorted_idx), room_n * 2))]
            rng.shuffle(inner_cells)
            chosen = inner_cells[:room_n]

            for r, c in chosen:
                wx, wy = self.grid_to_world(r, c)
                # Small random offset within the cell so civilians do
                # not all stand exactly on the centre point.
                ox = rng.uniform(-self.cell_size * 0.3, self.cell_size * 0.3)
                oy = rng.uniform(-self.cell_size * 0.3, self.cell_size * 0.3)
                positions.append((wx + ox, wy + oy))

        # Adjust if proportional rounding produced too many or too few.
        if len(positions) > n:
            rng.shuffle(positions)
            positions = positions[:n]
        elif len(positions) < n:
            extra = self._spawn_random(n - len(positions), seed + 1000)
            positions.extend(extra)

        return positions[:n]

    def _spawn_random(self, n: int, seed: int) -> list[tuple[float, float]]:
        """Fallback random-cell spawn used when room detection finds nothing."""
        rng = np.random.default_rng(seed)
        floor_cells = list(zip(*np.where(self.grid == CellType.FLOOR)))

        # Skip cells too close to an exit or to a fire source.
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
        """Return True if the world position is inside the map and not a wall."""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False

        r, c = self.world_to_grid(x, y)
        return self.grid[r, c] != CellType.WALL
