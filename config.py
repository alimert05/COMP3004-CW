from dataclasses import dataclass, field


@dataclass
class SimConfig:
    map_file: str = "maps/bottleneck_A.txt"
    cell_size: float = 10.0

    # Fire start position as (row, col) grid coordinates
    # If None, uses F markers in the map file
    # Named positions: "center", "top_left", "top_right", "bottom_left", "bottom_right"
    fire_position: str = "center"


# --- Simulation settings ---

CIVILIAN_NUMBER = 50

# Guide policy: "none", "beacon", "sweeper", "router"
GUIDE_POLICY = "sweeper"

# Enable/disable guides
GUIDES_ENABLED = True

# Number of guides (None = 1 per exit)
GUIDE_COUNT = None
