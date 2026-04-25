# COMP3004 — Evacuation Simulation

An agent-based fire evacuation simulator that compares three guide agent
policies (Beacon, Sweeper, Router) across a 14,400-run factorial experiment
covering eight topologies, five fire positions, and three crowd sizes.

## 1. Author

Ali Karaoglu (20587851)
COMP3004 Designing Intelligent Agents — University of Nottingham
Academic year 2025/2026

## 2. Requirements

- Python 3.10 or newer (developed on 3.13.2)
- Packages listed in `requirements.txt`

Install with:

```bash
pip install -r requirements.txt
```

## 3. Smoke test

Verify the full pipeline works end-to-end with a 4-run sanity check
(approximately 45 seconds on a modern 8-core machine):

```bash
python -m experiments.runner --smoke
```

This runs one simulation per policy on `bottleneck_A` at crowd size 100,
streams results to `results/experiment_results.csv`, and exits.

## 4. Visual demo

Run the simulation with a Pygame window showing civilians, guides, fire,
and smoke in real time:

```bash
python test_map_visual.py
```

The map file and policy are selected via `VISUAL_*` constants at the bottom
of `config.py`. Change `VISUAL_GUIDE_POLICY` to `"none"`, `"beacon"`,
`"sweeper"`, or `"router"` to see different policies, and
`VISUAL_CIVILIAN_NUMBER` to vary crowd size.

## 5. Reproducing the experiment

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

## 6. Performance notes

Approximate wall-clock times on an 8-core machine:

| Mode  | Runs   | Time        |
|-------|--------|-------------|
| Smoke | 4      | ~45 sec     |
| Pilot | 960    | ~30 min     |
| Full  | 14,400 | ~7-8 hours  |

Per-run CPU cost scales with policy complexity and crowd size; Router at
n = 150 civilians is the slowest configuration. Disk output for the full
run is approximately 2 MB.

Seeds are deterministic via `zlib.crc32(trial_index)`, so re-running the
full experiment produces identical numbers across sessions, machines, and
Python versions.

## 7. Reproducing the analysis

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

## 8. Output format

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

## 9. File layout

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

## 10. Related documents

- **Main report**: `Agents_Report.pdf` (submitted separately via Moodle)
- **Supplementary PDFs**: `agent_specs.pdf`, `statistics.pdf` (submitted as a supplementary zip)
- **Video demonstration**: `video_COMP3004_20587851.mp4`
  (submitted separately)

## 11. Platform notes

- **Windows**: multiprocessing in `experiments/runner.py` is already
  guarded with `if __name__ == "__main__":`. No extra action needed.
- **Linux / macOS**: use `python3` in place of `python` if the system
  default interpreter is Python 2.
- **Headless systems**: `experiments/analysis.py` sets
  `matplotlib.use("Agg")`, so it works without a display.
- **Pygame on Linux**: if the visual demo fails to open a window, install
  SDL2 development headers (for example, `apt install libsdl2-dev` on
  Debian-based distributions).
