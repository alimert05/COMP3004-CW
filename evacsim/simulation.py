"""
Headless simulation engine — no Pygame, pure computation.

Usage:
    from config import SimConfig
    from evacsim.simulation import Simulation

    config = SimConfig(map_file="maps/bottleneck_A.txt", guide_policy="beacon")
    result = Simulation(config).run()
    print(result)  # {'TV': 35, 'TM': 15, 'MP': 62.3, 'MT': 450.2, ...}
"""

import numpy as np
from config import SimConfig
from evacsim.environment import Environment
from evacsim.smoke import SmokeSystem
from evacsim.fire import FireSystem
from evacsim.civilian import (
    spawn_civilians, IDLE, REACTING, QUEUED, ESCAPED, DEAD
)
from evacsim.guide import (
    spawn_beacon_guides, spawn_sweeper_guides, spawn_router_guides
)
from evacsim.metrics import MetricsCollector


class Simulation:
    """Run a complete evacuation simulation headlessly."""

    def __init__(self, config: SimConfig):
        self.config = config

        # Build environment
        self.env = Environment(config)

        # Build smoke & fire
        self.smoke = SmokeSystem(self.env, spread_rate=config.smoke_spread_rate,
                                  seed=config.seed)
        self.fire = FireSystem(self.env, spread_rate=config.fire_spread_rate,
                                seed=config.seed + 1)
        # Fire–smoke coupling: smoke is continuously sourced from fire cells
        # (Makmul 2020 source-term CA). Must be set AFTER fire is created.
        self.smoke._fire = self.fire

        # Spawn civilians
        self.civilians = spawn_civilians(
            self.env, n=config.n_civilians,
            speed=config.civilian_speed, seed=config.seed + 100
        )

        # Spawn guides (fixed count for fair comparison)
        n_guides = config.n_guides if config.n_guides is not None else 4

        self.guides = []
        if config.guide_policy == "beacon":
            self.guides = spawn_beacon_guides(
                self.env, n_guides=n_guides,
                speed=config.guide_speed, seed=config.seed + 200
            )
        elif config.guide_policy == "sweeper":
            self.guides = spawn_sweeper_guides(
                self.env, n_guides=n_guides,
                speed=config.guide_speed, seed=config.seed + 200,
                n_civilians=config.n_civilians
            )
        elif config.guide_policy == "router":
            self.guides = spawn_router_guides(
                self.env, n_guides=n_guides,
                speed=config.guide_speed, seed=config.seed + 200,
                n_civilians=config.n_civilians
            )
        # "none" → empty list

        # Metrics
        self.metrics = MetricsCollector()

        # Track which civilians already had events recorded
        self._escaped_set = set()
        self._dead_set = set()
        self._started_set = set()

        self.tick = 0

    def step(self) -> bool:
        """Run one simulation tick. Returns True if simulation should continue."""
        self.tick += 1

        # 1. Spread smoke and fire
        self.smoke.step()
        self.fire.step()

        # 2. Update guides
        for guide in self.guides:
            guide.update(self.env, self.smoke, self.fire, self.civilians)

        # 3. Update civilians
        for civ in self.civilians:
            civ.update(self.env, self.smoke, self.fire, self.civilians)

        # 4. Exit processing — throughput-limited queue
        self._process_exit_queues()

        # 5. Record metrics events
        for i, civ in enumerate(self.civilians):
            # Track when civilians start moving
            if i not in self._started_set and civ.state not in (IDLE, REACTING):
                self._started_set.add(i)
                self.metrics.record_start(i, self.tick)

            # Track escapes
            if civ.state == ESCAPED and i not in self._escaped_set:
                self._escaped_set.add(i)
                self.metrics.record_escape(i, self.tick, civ.potency)

            # Track deaths
            if civ.state == DEAD and i not in self._dead_set:
                self._dead_set.add(i)
                self.metrics.record_death(i, self.tick)

        # 6. Check termination
        all_done = all(c.state in (ESCAPED, DEAD) for c in self.civilians)
        max_reached = self.tick >= self.config.max_ticks

        return not (all_done or max_reached)

    def _process_exit_queues(self):
        """Process exit queues with throughput limit.

        Each exit can only let exit_throughput civilians through per tick.
        Civilians closest to the exit go first. Others wait (QUEUED state).
        Waiting civilians still take potency damage — they can die in line."""
        import numpy as np

        throughput = self.config.exit_throughput

        # Group queued civilians by their target exit
        exit_queues = {}  # (ex, ey) → list of (distance, civ_index)
        for i, civ in enumerate(self.civilians):
            if civ.state != QUEUED:
                continue
            if civ._target_exit is None:
                # No target exit — just escape
                civ.state = ESCAPED
                continue

            tx, ty = civ._target_exit
            # Find which exit cell they're at
            best_exit = None
            best_d = float("inf")
            for ex, ey in self.env.exits:
                d = np.hypot(civ.pos[0] - ex, civ.pos[1] - ey)
                if d < best_d:
                    best_d = d
                    best_exit = (ex, ey)

            if best_exit is not None:
                if best_exit not in exit_queues:
                    exit_queues[best_exit] = []
                exit_queues[best_exit].append((best_d, i))

        # For each exit, let only N closest through
        for exit_pos, queue in exit_queues.items():
            # Sort by distance (closest first)
            queue.sort(key=lambda x: x[0])

            for rank, (dist, civ_idx) in enumerate(queue):
                if rank < throughput:
                    # This civilian escapes this tick
                    self.civilians[civ_idx].state = ESCAPED
                # Others stay QUEUED — will try again next tick

    def run(self) -> dict:
        """Run simulation to completion. Returns metrics dict."""
        while self.step():
            pass

        result = self.metrics.summary(
            total_civilians=len(self.civilians),
            final_tick=self.tick
        )

        # Add config info to result
        result["map"] = self.config.map_file
        result["fire_position"] = self.config.fire_position
        result["policy"] = self.config.guide_policy
        result["n_civilians"] = self.config.n_civilians
        result["seed"] = self.config.seed
        result["n_guides"] = len(self.guides)
        # "Survived" = didn't die in fire (either still alive OR safely escaped)
        result["guides_survived"] = sum(
            1 for g in self.guides if g.alive or getattr(g, "escaped", False)
        )

        return result


def run_single(config: SimConfig) -> dict:
    """Convenience function for multiprocessing."""
    return Simulation(config).run()
