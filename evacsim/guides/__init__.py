"""Guide agents package (split from the monolithic evacsim/guide.py).

Public API re-exports — every name that was importable from
``evacsim.guide`` remains importable from ``evacsim.guides``.
"""
from evacsim.guides.common import (
    bfs_path,
    dijkstra_congestion_path,
    predict_spread,
    dbscan_cluster,
    build_density_grid,
)
from evacsim.guides.base import Guide
from evacsim.guides.beacon import BeaconGuide, spawn_beacon_guides
from evacsim.guides.sweeper import SweeperGuide, spawn_sweeper_guides
from evacsim.guides.router import (
    RouterBlackboard,
    RouterGuide,
    spawn_router_guides,
)

__all__ = [
    "bfs_path",
    "dijkstra_congestion_path",
    "predict_spread",
    "dbscan_cluster",
    "build_density_grid",
    "Guide",
    "BeaconGuide", "spawn_beacon_guides",
    "SweeperGuide", "spawn_sweeper_guides",
    "RouterBlackboard", "RouterGuide", "spawn_router_guides",
]
