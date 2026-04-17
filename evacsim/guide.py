"""Backwards-compatibility shim.

The guide implementation was split across ``evacsim/guides/`` (common,
base, beacon, sweeper, router) during the pre-experiment refactor.
This shim preserves the old ``from evacsim.guide import ...`` import
paths used by simulation.py, civilian.py, test_map_visual.py, and the
debug tests, so nothing outside the package needs to change.
"""
from evacsim.guides import *  # noqa: F401, F403
from evacsim.guides import (  # explicit re-exports for static analysis
    bfs_path,
    dijkstra_congestion_path,
    predict_spread,
    dbscan_cluster,
    build_density_grid,
    Guide,
    BeaconGuide,
    spawn_beacon_guides,
    SweeperGuide,
    spawn_sweeper_guides,
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
