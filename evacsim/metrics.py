"""
Evaluation metrics — based on Kasereka et al. 2018.

Four core metrics:
  TV — Total alive (escaped successfully)
  TM — Total dead
  MP — Mean potency of survivors (health at escape)
  MT — Mean time to escape (ticks)

Additional:
  evacuation_time — tick when last civilian escaped or died
  survival_rate — TV / (TV + TM)
"""


class MetricsCollector:
    """Collects per-civilian escape data during simulation."""

    def __init__(self):
        self._escape_times = []     # (tick, potency) for each escaped civilian
        self._death_times = []      # tick for each dead civilian
        self._start_tick = {}       # civilian_id → tick when they started moving

    def record_start(self, civ_id: int, tick: int):
        """Record when a civilian starts evacuating (leaves IDLE)."""
        if civ_id not in self._start_tick:
            self._start_tick[civ_id] = tick

    def record_escape(self, civ_id: int, tick: int, potency: float):
        """Record a civilian escaping."""
        start = self._start_tick.get(civ_id, 0)
        self._escape_times.append((tick - start, potency))

    def record_death(self, civ_id: int, tick: int):
        """Record a civilian dying."""
        self._death_times.append(tick)

    def summary(self, total_civilians: int, final_tick: int) -> dict:
        """Compute all metrics. Returns dict."""
        tv = len(self._escape_times)       # Total alive
        tm = len(self._death_times)        # Total dead
        still_alive = total_civilians - tv - tm  # Still in building

        # Mean potency of survivors
        if tv > 0:
            mp = sum(p for _, p in self._escape_times) / tv
        else:
            mp = 0.0

        # Mean escape time
        if tv > 0:
            mt = sum(t for t, _ in self._escape_times) / tv
        else:
            mt = float(final_tick)  # Nobody escaped

        # Survival rate
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
