"""Guide agents package.

Re-exports the public API: the guide policy classes (BeaconGuide,
SweeperGuide, RouterGuide), their blackboards (SweeperBlackboard,
RouterBlackboard), the spawn helpers, the Guide base class, and the
shared path-finding and clustering utilities.
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
from evacsim.guides.sweeper import (
    SweeperBlackboard,
    SweeperGuide,
    spawn_sweeper_guides,
)
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
    "SweeperBlackboard", "SweeperGuide", "spawn_sweeper_guides",
    "RouterBlackboard", "RouterGuide", "spawn_router_guides",
]
