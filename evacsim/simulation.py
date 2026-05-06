"""Headless simulation engine.

A pure-computation runner with no Pygame dependency. The visual mode
in test_map_visual.py reuses the same component classes but drives its
own loop.

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
from evacsim.guides import (
    spawn_beacon_guides, spawn_sweeper_guides, spawn_router_guides
)
from evacsim.metrics import MetricsCollector


class Simulation:
    """Run a complete evacuation simulation headlessly."""

    def __init__(self, config: SimConfig):
        self.config = config

        # Build the environment from the map file.
        self.env = Environment(config)

        # Build smoke and fire systems with deterministic seeds derived
        # from the master seed.
        self.smoke = SmokeSystem(self.env, spread_rate=config.smoke_spread_rate,
                                  seed=config.seed)
        self.fire = FireSystem(self.env, spread_rate=config.fire_spread_rate,
                                seed=config.seed + 1)
        # Fire-and-smoke coupling: smoke is continuously sourced from
        # fire cells. Must be set after both systems are constructed.
        self.smoke._fire = self.fire

        # Spawn civilians with a separate seed offset so that civilian
        # randomness is independent of smoke and fire randomness.
        self.civilians = spawn_civilians(
            self.env, n=config.n_civilians,
            speed=config.civilian_speed, seed=config.seed + 100
        )

        # Spawn guides. Count is fixed across policies for fairness.
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
        # The "none" policy spawns no guides (empty list).

        # Metrics collector.
        self.metrics = MetricsCollector()

        # Track which civilians have already had each event recorded
        # so the metric collector receives one event per civilian.
        self._escaped_set = set()
        self._dead_set = set()
        self._started_set = set()

        self.tick = 0

    def step(self) -> bool:
        """Run one simulation tick. Return True if the run should continue."""
        self.tick += 1

        # 1. Advance the cellular automata one tick.
        self.smoke.step()
        self.fire.step()

        # 2. Update guides.
        for guide in self.guides:
            guide.update(self.env, self.smoke, self.fire, self.civilians)

        # 3. Update civilians.
        for civ in self.civilians:
            civ.update(self.env, self.smoke, self.fire, self.civilians)

        # 4. Process exit queues with throughput limiting.
        self._process_exit_queues()

        # 5. Record metric events (one per civilian for each event type).
        for i, civ in enumerate(self.civilians):
            # Mark when each civilian leaves the IDLE/REACTING phase.
            if i not in self._started_set and civ.state not in (IDLE, REACTING):
                self._started_set.add(i)
                self.metrics.record_start(i, self.tick)

            # Record escapes.
            if civ.state == ESCAPED and i not in self._escaped_set:
                self._escaped_set.add(i)
                self.metrics.record_escape(i, self.tick, civ.potency)

            # Record deaths.
            if civ.state == DEAD and i not in self._dead_set:
                self._dead_set.add(i)
                self.metrics.record_death(i, self.tick)

        # 6. Termination check: every civilian resolved OR cap reached.
        all_done = all(c.state in (ESCAPED, DEAD) for c in self.civilians)
        max_reached = self.tick >= self.config.max_ticks

        return not (all_done or max_reached)

    def _process_exit_queues(self):
        """Process exit queues with a per-exit per-tick throughput limit.

        Each exit lets at most exit_throughput civilians through per
        tick; the closest civilians go first. Civilians beyond that
        cap stay in QUEUED and try again next tick. Queued civilians
        still take potency damage and can die in line.
        """
        throughput = self.config.exit_throughput

        # Group queued civilians by their target exit.
        exit_queues = {}  # (ex, ey) -> list of (distance, civ_index)
        for i, civ in enumerate(self.civilians):
            if civ.state != QUEUED:
                continue
            if civ._target_exit is None:
                # No target recorded; let them escape directly.
                civ.state = ESCAPED
                civ.escape_tick = self.tick
                continue

            tx, ty = civ._target_exit
            # Identify which exit cell this civilian is closest to.
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

        # For each exit, let only the throughput nearest civilians
        # through.
        for exit_pos, queue in exit_queues.items():
            queue.sort(key=lambda x: x[0])

            for rank, (dist, civ_idx) in enumerate(queue):
                if rank < throughput:
                    self.civilians[civ_idx].state = ESCAPED
                    self.civilians[civ_idx].escape_tick = self.tick
                # Otherwise the civilian stays QUEUED until next tick.

    def run(self) -> dict:
        """Run the simulation to completion. Return the summary metrics dict."""
        while self.step():
            pass

        result = self.metrics.summary(
            total_civilians=len(self.civilians),
            final_tick=self.tick
        )

        # Add configuration metadata to the result row.
        result["map"] = self.config.map_file
        result["fire_position"] = self.config.fire_position
        result["policy"] = self.config.guide_policy
        result["n_civilians"] = self.config.n_civilians
        result["seed"] = self.config.seed
        result["n_guides"] = len(self.guides)
        # 'Survived' here means did not die: either still alive on the
        # patrol or successfully self-evacuated through an exit.
        result["guides_survived"] = sum(
            1 for g in self.guides if g.alive or getattr(g, "escaped", False)
        )

        # Escape-tick bins for evacuation curves: 12 bins of 500 ticks
        # each covering the 0 to 6000 range. Each bin holds the count
        # of civilians who escaped during that window. The analysis
        # script uses these to build cumulative escape curves per
        # policy.
        bin_edges = [0, 500, 1000, 1500, 2000, 2500, 3000,
                     3500, 4000, 4500, 5000, 5500, 6000]
        escape_ticks = [
            c.escape_tick for c in self.civilians if c.escape_tick is not None
        ]
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            count = sum(1 for t in escape_ticks if lo <= t < hi)
            result[f"escaped_{lo}_{hi}"] = count

        return result


def run_single(config: SimConfig) -> dict:
    """Convenience wrapper used by the multiprocessing runner."""
    return Simulation(config).run()
