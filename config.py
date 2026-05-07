from dataclasses import dataclass


@dataclass
class SimConfig:
    # Map
    map_file: str = "maps/bottleneck_A.txt"
    cell_size: float = 10.0
    fire_position: str = (
        "center"  # "center", "top_right", "top_left", "bottom_right", "bottom_left"
    )

    # Civilians
    n_civilians: int = 50
    civilian_speed: float = 1.5

    # Guides
    guide_policy: str = "beacon"  # "none", "beacon", "sweeper", "router"
    n_guides: int = 4  # Fixed at 4 for fair comparison across maps.
    guide_speed: float = 2.0

    # Smoke and fire spread coefficients (per-tick CA spread rates).
    smoke_spread_rate: float = 0.02
    fire_spread_rate: float = 0.005

    # Simulation limits.
    max_ticks: int = 6000  # Safety cap derived from a one-off
    # saturation probe: worst-case smoke-damage
    # coverage about 4,500 ticks, plus civilian
    # death time (around 300 ticks at the damage
    # threshold), plus a 1,200-tick safety buffer.
    # Simulations almost always terminate via the
    # all-resolved check well before this cap.
    seed: int = 42

    # Exit throughput: maximum civilians escaping per exit per tick.
    exit_throughput: int = 2


# Visual-mode settings used by test_map_visual.py only. These are not
# read by experiments/runner.py, which constructs SimConfig objects
# directly. The VISUAL_ prefix marks the scope so changes here do not
# affect the headless experiment.

VISUAL_CIVILIAN_NUMBER = 100
VISUAL_GUIDE_POLICY = "router"  # "none", "beacon", "sweeper", "router"
VISUAL_GUIDES_ENABLED = True
VISUAL_GUIDE_COUNT = None  # Default: one guide per exit (typically 4).
VISUAL_FIRE_POSITION = (
    "center"  # "center", "top_left", "top_right", "bottom_left", "bottom_right"
)
