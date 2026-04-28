# COMP3004 — Evacuation Simulation

An agent-based fire evacuation simulator that compares three guide agent
policies (Beacon, Sweeper, Router) across a 14,400-run factorial experiment
covering eight topologies, five fire positions, and three crowd sizes.

## 1. Author

Ali Karaoglu (20587851)
COMP3004 Designing Intelligent Agents — University of Nottingham
Academic year 2025/2026

## 2. Policies at a glance

- **Beacon** — passive broadcaster. Stands two cells inward from an
  assigned exit and announces the exit to every civilian within
  hearing range. Hearing probability decays with distance and with
  smoke at both endpoints.
- **Sweeper** — reactive patroller. Walks an assigned territory in
  alternating-half order, recruits civilians visually within
  line-of-sight, and escorts them to the nearest safe exit using
  a three-layer Dijkstra → safety-validator → BFS path cascade.
- **Router** — hybrid reactive-deliberative. Globally scans for
  civilian clusters via DBSCAN, predicts fire and smoke spread 30
  ticks ahead, routes through density-weighted Dijkstra paths, and
  re-evaluates the destination exit every 20 ticks under a
  multi-criteria score (distance, congestion, smoke, fire).

## 3. Requirements

- Python 3.10 or newer (developed on 3.14.3)
- Packages listed in `requirements.txt`

Install with:

```bash
pip install -r requirements.txt
```

## 4. Smoke test

Verify the full pipeline works end-to-end with a 4-run sanity check
(approximately 45 seconds on a modern 8-core machine):

```bash
python -m experiments.runner --smoke
```

This runs one simulation per policy on `bottleneck_A` at crowd size 100,
streams results to `results/experiment_results.csv`, and exits.

## 5. Visual demo

Run the simulation with a Pygame window showing civilians, guides, fire,
and smoke in real time:

```bash
python test_map_visual.py
```

The map file and policy are selected via `VISUAL_*` constants at the bottom
of `config.py`. Change `VISUAL_GUIDE_POLICY` to `"none"`, `"beacon"`,
`"sweeper"`, or `"router"` to see different policies, and
`VISUAL_CIVILIAN_NUMBER` to vary crowd size.

## 6. Reproducing the experiment

The full factorial (8 maps × 5 fire positions × 4 policies × 3 crowd sizes
× 30 trials = 14,400 runs):

```bash
python -m experiments.runner
```

Results stream to `results/experiment_results.csv` as each run completes
(one row per run), so a crash preserves every completed run on disk.

Alternative modes:

| Mode          | Command                                                    | Runs   |
|---------------|------------------------------------------------------------|--------|
| Smoke         | `python -m experiments.runner --smoke`                     | 4      |
| Pilot         | `python -m experiments.runner --pilot`                     | 960    |
| Full          | `python -m experiments.runner`                             | 14,400 |
| Single policy | `python -m experiments.runner --policy router`             | 3,600  |
| Single map    | `python -m experiments.runner --map maps/bottleneck_A.txt` | 1,800  |

Worker count defaults to 8; override with `--workers N`.

## 7. Performance notes

Approximate wall-clock times on an 8-core machine:

| Mode  | Runs   | Time        |
|-------|--------|-------------|
| Smoke | 4      | ~45 sec     |
| Pilot | 960    | ~30 min     |
| Full  | 14,400 | ~7-8 hours  |

Per-run CPU cost scales with policy complexity and crowd size; Router at
n = 150 civilians is the slowest configuration. Disk output for the full
run is approximately 2 MB.

## 8. Reproducibility

The 14,400-run dataset in `results/experiment_results.csv` is fully
reproducible:

- Seeds are derived deterministically from the trial index via
  `zlib.crc32`, which is stable across processes, Python versions,
  and operating systems.
- A re-run of `python -m experiments.runner` produces the same numbers
  byte for byte.
- The seed scheme is fully paired across factorial conditions: trial N
  represents the same civilian cohort under every (map, fire, policy,
  crowd) combination, so within-cell and across-cell comparisons are
  matched on stochastic noise.

## 9. Reproducing the analysis

After `results/experiment_results.csv` exists, generate all statistical
tables and figures:

```bash
python -m experiments.analysis
```

Outputs are written to `results/analysis/` by default. Override with
`--input` or `--output`.

The analysis produces:

- One-way ANOVA on policy (survival rate)
- Two-way ANOVA on policy × map_type, policy × n_civilians, policy × fire_position
- Partial η² effect sizes
- Pairwise Tukey HSD for policy contrasts, with Cohen's d
- Box plots, line charts, policy × map heatmap
- Per-policy summary CSVs

## 10. Configuration

Key `SimConfig` fields (in `config.py`):

| Field               | Default                       | Purpose                                                            |
|---------------------|-------------------------------|--------------------------------------------------------------------|
| `map_file`          | `"maps/bottleneck_A.txt"`     | Map file path                                                      |
| `cell_size`         | `10.0`                        | World units per grid cell (≈ 3 m)                                  |
| `fire_position`     | `"center"`                    | One of `center`, `top_left`, `top_right`, `bottom_left`, `bottom_right` |
| `n_civilians`       | `50`                          | Number of civilians spawned                                        |
| `civilian_speed`    | `1.5`                         | Civilian movement, world units per tick                            |
| `guide_policy`      | `"beacon"`                    | One of `none`, `beacon`, `sweeper`, `router`                       |
| `n_guides`          | `4`                           | Number of guides spawned                                           |
| `guide_speed`       | `2.0`                         | Guide movement, world units per tick                               |
| `smoke_spread_rate` | `0.02`                        | Per-tick spread coefficient for the smoke CA                       |
| `fire_spread_rate`  | `0.005`                       | Per-tick spread coefficient for the fire CA                        |
| `max_ticks`         | `6000`                        | Hard simulation cap                                                |
| `seed`              | `42`                          | RNG seed for the run                                               |
| `exit_throughput`   | `2`                           | Maximum civilians escaping per exit per tick                       |

## 11. Map file format

Maps in `maps/*.txt` are plain ASCII grids using one character per cell:

| Character | Meaning                                              |
|-----------|------------------------------------------------------|
| `.`       | Floor (walkable)                                     |
| `#`       | Wall                                                 |
| `E`       | Exit                                                 |
| `F`       | Fire source (overridden by `fire_position` in config) |

A 36×80 map is 36 lines of 80 characters each. Lines shorter than the
maximum row are padded with walls. Every map in the experiment has
four exits.

## 12. Output format

Each row in `results/experiment_results.csv` represents one simulation run.

Key columns:

| Column              | Meaning                                                                       |
|---------------------|-------------------------------------------------------------------------------|
| `map`               | Map file path                                                                 |
| `fire_position`     | Fire origin (`center`, `top_left`, `top_right`, `bottom_left`, `bottom_right`) |
| `policy`            | `none`, `beacon`, `sweeper`, `router`                                         |
| `n_civilians`       | Crowd size (50, 100, or 150)                                                  |
| `trial`             | Trial index within the (map, fire, policy, crowd) cell                        |
| `seed`              | Deterministic seed derived from `trial`                                       |
| `TV`                | Total alive: civilians who escaped (Kasereka 2018)                            |
| `TM`                | Total dead: civilians killed by fire or smoke (Kasereka 2018)                 |
| `MP`                | Mean potency (remaining health) of survivors at escape                        |
| `MT`                | Mean time to escape, in ticks                                                 |
| `survival_rate`     | `TV / (TV + TM)` — primary outcome metric                                     |
| `evacuation_time`   | Tick when the last civilian resolved (escaped or died)                        |
| `still_alive`       | Civilians neither escaped nor dead at `max_ticks` (usually 0)                 |
| `n_guides`          | Number of guides spawned (fixed at 4)                                         |
| `guides_survived`   | Guides alive or successfully escaped at end of run                            |
| `escaped_{lo}_{hi}` | Civilians who escaped in that 500-tick window (12 bins, 0–6000)               |

One simulation tick corresponds to roughly one second of simulated time.

## 13. File layout

```
COMP3004-CW/
├── README.md                  This file
├── requirements.txt           Python dependencies
├── config.py                  SimConfig dataclass
├── test_headless.py           Single-run headless test
├── test_map_visual.py         Pygame visual mode
├── evacsim/
│   ├── environment.py         Grid, walls, exits, map loader
│   ├── smoke.py               Smoke cellular automaton
│   ├── fire.py                Fire cellular automaton
│   ├── civilian.py            Civilian agent
│   ├── simulation.py          Simulation engine
│   ├── metrics.py             Metric collector (TV/TM/MP/MT)
│   ├── guide.py               Thin shim re-exporting guide classes
│   └── guides/
│       ├── common.py          BFS, Dijkstra, DBSCAN, predict_spread
│       ├── base.py            Guide base class
│       ├── beacon.py          BeaconGuide
│       ├── sweeper.py         SweeperGuide
│       └── router.py          RouterGuide + RouterBlackboard
├── experiments/
│   ├── runner.py              Factorial runner (14,400 runs)
│   └── analysis.py            ANOVA, Tukey, figures
├── maps/                      8 map files (bottleneck A-D, open_plan A-D)
└── results/                   Experiment output (CSV + analysis)
```

## 14. Platform notes

- **Windows**: multiprocessing in `experiments/runner.py` is already
  guarded with `if __name__ == "__main__":`. No extra action needed.
- **Linux / macOS**: use `python3` in place of `python` if the system
  default interpreter is Python 2.
- **Headless systems**: `experiments/analysis.py` sets
  `matplotlib.use("Agg")`, so it works without a display.
- **Pygame on Linux**: if the visual demo fails to open a window, install
  SDL2 development headers (for example, `apt install libsdl2-dev` on
  Debian-based distributions).
