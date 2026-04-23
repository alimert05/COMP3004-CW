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
    n_guides: int = 4  # Fixed at 4 for fair comparison across maps
    guide_speed: float = 2.0

    # Smoke/Fire
    smoke_spread_rate: float = 0.02
    fire_spread_rate: float = 0.005

    # Simulation limits
    max_ticks: int = 6000  # Safety cap chosen from measure_saturation.py probe:
    # worst-case smoke damage coverage ~4,500 ticks,
    # plus civilian death time (~300 ticks at damage
    # threshold), plus ~1,200-tick safety buffer.
    # In practice, simulations terminate via `all_done`
    # well before this cap.
    seed: int = 42

    # Exit throughput (max civilians escaping per exit per tick)
    exit_throughput: int = 2


# --- Visual mode settings (used by test_map_visual.py only) ---
# These are NOT read by experiments/runner.py or the debug tests - those
# construct SimConfig objects directly. The VISUAL_ prefix makes the scope
# unambiguous so nobody expects edits here to affect the full experiment.

VISUAL_CIVILIAN_NUMBER = 100
VISUAL_GUIDE_POLICY = "none"
VISUAL_GUIDES_ENABLED = True
VISUAL_GUIDE_COUNT = None  # default one guide per exit (4)
VISUAL_FIRE_POSITION = "center"  # "center", "top_left", "top_right",
# "bottom_left", "bottom_right"
