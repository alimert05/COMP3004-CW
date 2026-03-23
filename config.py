# We import 'dataclass' — a shortcut that auto-creates __init__ for us
from dataclasses import dataclass


# @dataclass means: "make this class a simple data holder"
# It automatically creates __init__ so we don't have to write one
@dataclass
class SimConfig:
    # Which map file to load. Default is the bottleneck map
    map_file: str = "maps/bottleneck.txt"

    # How big each grid cell is in "world units"
    # Our grid is like graph paper — this says each square is 10x10 units
    cell_size: float = 10.0


CIVILIAN_NUMBER = 50
