from dataclasses import dataclass


@dataclass
class SimConfig:
    # Map
    map_file: str = "maps/bottleneck_A.txt"
    cell_size: float = 10.0
    fire_position: str = "top_left"

    # Civilians
    n_civilians: int = 50
    civilian_speed: float = 1.5

    # Guides
    guide_policy: str = "beacon"  # "none", "beacon", "sweeper", "router"
    n_guides: int = None  # None = 1 per exit
    guide_speed: float = 2.0

    # Smoke/Fire
    smoke_spread_rate: float = 0.02
    fire_spread_rate: float = 0.005

    # Simulation limits
    max_ticks: int = 3000  # Stop after this many ticks (~100s at 30fps)
    seed: int = 42

    # Exit throughput (max civilians escaping per exit per tick)
    exit_throughput: int = 2


# --- Visual mode settings (used by test_map_visual.py only) ---

CIVILIAN_NUMBER = 100
GUIDE_POLICY = "router"
GUIDES_ENABLED = True
GUIDE_COUNT = None
