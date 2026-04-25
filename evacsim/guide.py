"""Compatibility shim for the legacy import path.

The guide implementation lives in evacsim/guides/ as a package
(common, base, beacon, sweeper, router). This shim re-exports every
public name so that the older 'from evacsim.guide import ...' import
path continues to work for simulation.py, civilian.py,
test_map_visual.py, and the debug tests.
"""
from evacsim.guides import *  # noqa: F401, F403
from evacsim.guides import (  # Explicit re-exports for static analysis.
    bfs_path,
    dijkstra_congestion_path,
    predict_spread,
    dbscan_cluster,
    build_density_grid,
    Guide,
    BeaconGuide,
    spawn_beacon_guides,
    SweeperBlackboard,
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
    "SweeperBlackboard", "SweeperGuide", "spawn_sweeper_guides",
    "RouterBlackboard", "RouterGuide", "spawn_router_guides",
]
