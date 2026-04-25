"""Per-civilian metric collection for the simulation.

Four primary outcome metrics are reported per run:

    TV    Total alive (number of civilians who escaped successfully).
    TM    Total dead (number of civilians killed by fire or smoke).
    MP    Mean potency of survivors at the moment they escaped.
    MT    Mean time to escape, in ticks, measured from the moment the
          civilian first started moving.

Two additional fields are reported alongside:

    evacuation_time  Tick at which the last civilian escaped or died.
    survival_rate    TV divided by (TV + TM); the primary outcome.
"""


class MetricsCollector:
    """Records per-civilian escape and death events during a run."""

    def __init__(self):
        self._escape_times = []  # (tick, potency) for each civilian who escaped.
        self._death_times = []   # tick for each civilian who died.
        self._start_tick = {}    # civilian_id -> tick when they began evacuating.

    def record_start(self, civ_id: int, tick: int):
        """Record when a civilian starts evacuating (leaves IDLE)."""
        if civ_id not in self._start_tick:
            self._start_tick[civ_id] = tick

    def record_escape(self, civ_id: int, tick: int, potency: float):
        """Record a civilian escaping. Time is measured from the start tick."""
        start = self._start_tick.get(civ_id, 0)
        self._escape_times.append((tick - start, potency))

    def record_death(self, civ_id: int, tick: int):
        """Record a civilian dying."""
        self._death_times.append(tick)

    def summary(self, total_civilians: int, final_tick: int) -> dict:
        """Compute and return all metrics as a dictionary."""
        tv = len(self._escape_times)       # Total alive (escaped).
        tm = len(self._death_times)        # Total dead.
        still_alive = total_civilians - tv - tm  # Still in the building.

        # Mean potency of survivors at escape.
        if tv > 0:
            mp = sum(p for _, p in self._escape_times) / tv
        else:
            mp = 0.0

        # Mean time-to-escape over civilians who escaped. If nobody
        # escaped, fall back to the final tick so the field is still
        # comparable across runs.
        if tv > 0:
            mt = sum(t for t, _ in self._escape_times) / tv
        else:
            mt = float(final_tick)

        # Survival rate among resolved civilians (escaped or dead).
        total_resolved = tv + tm
        survival_rate = tv / total_resolved if total_resolved > 0 else 0.0

        return {
            "TV": tv,
            "TM": tm,
            "MP": round(mp, 2),
            "MT": round(mt, 2),
            "still_alive": still_alive,
            "evacuation_time": final_tick,
            "survival_rate": round(survival_rate, 4),
            "total_civilians": total_civilians,
        }
