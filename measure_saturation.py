"""
Measures how many ticks it takes for fire and smoke CA propagation to
cover every floor cell at the damage threshold on each (map x fire
position) combination. Used to set an empirically-justified max_ticks
safety cap for the factorial experiment.

Background:
  The experiment terminates a simulation when either all civilians are
  resolved (ESCAPED or DEAD) or max_ticks is reached. If max_ticks is
  too low, civilians may remain unresolved at timeout, creating a
  right-censoring issue in the survival/time metrics.

  Fire and smoke do NOT depend on civilians - their CA propagation is
  purely a function of the density grid, walls, and fire-source cells.
  So we can measure saturation time directly by running only the CA,
  without any agent logic.

  Once smoke/fire cover every floor cell at their damage thresholds
  (smoke >= 0.3, fire >= 0.1), any civilian in any cell will be taking
  damage and will die within ~100 ticks (smoke: 0.3+ potency/tick;
  fire: 0.3+ potency/tick; starting potency 50-100).

  So a safe max_ticks is:  (worst case damage-threshold coverage time)
                           +  (civilian death buffer, ~100 ticks).

Usage:
  python measure_saturation.py
"""
from config import SimConfig
from evacsim.environment import Environment, CellType
from evacsim.smoke import SmokeSystem
from evacsim.fire import FireSystem
import time


MAPS = [
    "maps/bottleneck_A.txt", "maps/bottleneck_B.txt",
    "maps/bottleneck_C.txt", "maps/bottleneck_D.txt",
    "maps/open_plan_A.txt",  "maps/open_plan_B.txt",
    "maps/open_plan_C.txt",  "maps/open_plan_D.txt",
]
FIRE_POSITIONS = ["center", "top_left", "top_right",
                  "bottom_left", "bottom_right"]

MAX_PROBE_TICKS       = 20000   # upper bound on probe length
SMOKE_DAMAGE_THRESH   = 0.3     # civilian takes smoke damage above this
FIRE_DAMAGE_THRESH    = 0.1     # civilian takes fire damage above this
SATURATION_THRESH     = 0.95    # "effectively full" density
CIVILIAN_DEATH_BUFFER = 100     # ticks for a civilian at damage threshold to die


def probe_one_scenario(map_file, fire_pos):
    """Run CA propagation for one (map, fire_pos) and return
    (t_smoke_dmg, t_fire_dmg, t_smoke_sat) - ticks at which each
    threshold was first reached across every floor cell. Returns
    None for any threshold not reached within MAX_PROBE_TICKS."""
    config = SimConfig(map_file=map_file, fire_position=fire_pos,
                       n_civilians=0, seed=42)
    env = Environment(config)

    # Floor cells are those where civilians could be stuck.
    # Walls and exits don't need hazard coverage.
    floor_mask = (env.grid != CellType.WALL) & (env.grid != CellType.EXIT)

    fire = FireSystem(env, spread_rate=0.005, seed=1)
    smoke = SmokeSystem(env, spread_rate=0.02, seed=2, fire=fire)

    t_smoke_dmg = None
    t_fire_dmg  = None
    t_smoke_sat = None

    for tick in range(1, MAX_PROBE_TICKS + 1):
        smoke.step()
        fire.step()

        # An efficient coverage check: every floor cell meets threshold.
        # The mask | ~floor_mask trick counts non-floor cells as "already met"
        # so .all() returns True exactly when all floor cells meet the threshold.
        if t_smoke_dmg is None and \
           ((smoke.density >= SMOKE_DAMAGE_THRESH) | ~floor_mask).all():
            t_smoke_dmg = tick
        if t_fire_dmg is None and \
           ((fire.density >= FIRE_DAMAGE_THRESH) | ~floor_mask).all():
            t_fire_dmg = tick
        if t_smoke_sat is None and \
           ((smoke.density >= SATURATION_THRESH) | ~floor_mask).all():
            t_smoke_sat = tick

        # Short-circuit once all three thresholds are crossed
        if t_smoke_dmg and t_fire_dmg and t_smoke_sat:
            break

    return t_smoke_dmg, t_fire_dmg, t_smoke_sat


def main():
    t_start = time.time()
    results = []

    print(f"Probing {len(MAPS)} maps x {len(FIRE_POSITIONS)} fire positions "
          f"= {len(MAPS) * len(FIRE_POSITIONS)} scenarios")
    print(f"Max probe length: {MAX_PROBE_TICKS} ticks\n")
    print(f"  {'map':<16} {'fire_pos':<13} "
          f"{'smoke_dmg':>9} {'fire_dmg':>9} {'smoke_sat':>9}")
    print(f"  {'-'*16} {'-'*13} {'-'*9} {'-'*9} {'-'*9}")

    for map_file in MAPS:
        for fire_pos in FIRE_POSITIONS:
            t_smoke_dmg, t_fire_dmg, t_smoke_sat = \
                probe_one_scenario(map_file, fire_pos)

            map_name = map_file.split('/')[-1].replace('.txt', '')
            results.append({
                'map': map_name, 'fire': fire_pos,
                't_smoke_dmg': t_smoke_dmg,
                't_fire_dmg':  t_fire_dmg,
                't_smoke_sat': t_smoke_sat,
            })

            def fmt(v):
                return f"{v:>9}" if v is not None else f"{'>probe':>9}"
            print(f"  {map_name:<16} {fire_pos:<13} "
                  f"{fmt(t_smoke_dmg)} {fmt(t_fire_dmg)} {fmt(t_smoke_sat)}")

    elapsed = time.time() - t_start
    print(f"\nElapsed: {elapsed:.1f}s  (over {len(results)} scenarios)\n")

    # Summary
    smoke_dmg_times = [r['t_smoke_dmg'] for r in results if r['t_smoke_dmg']]
    fire_dmg_times  = [r['t_fire_dmg']  for r in results if r['t_fire_dmg']]
    smoke_sat_times = [r['t_smoke_sat'] for r in results if r['t_smoke_sat']]

    n_total = len(results)
    n_smoke_dmg = len(smoke_dmg_times)
    n_fire_dmg  = len(fire_dmg_times)
    n_smoke_sat = len(smoke_sat_times)

    print("Coverage counts (how many scenarios reached the threshold within probe length):")
    print(f"  smoke damage threshold (0.3) covered: {n_smoke_dmg}/{n_total}")
    print(f"  fire  damage threshold (0.1) covered: {n_fire_dmg}/{n_total}")
    print(f"  smoke 95%-saturation covered:         {n_smoke_sat}/{n_total}")

    if smoke_dmg_times:
        print(f"\nWorst case smoke damage-threshold coverage: {max(smoke_dmg_times)} ticks")
    if fire_dmg_times:
        print(f"Worst case fire  damage-threshold coverage: {max(fire_dmg_times)} ticks")
    if smoke_sat_times:
        print(f"Worst case smoke 95%-saturation coverage:  {max(smoke_sat_times)} ticks")

    # Recommended cap
    if smoke_dmg_times and fire_dmg_times:
        damage_cap = max(max(smoke_dmg_times), max(fire_dmg_times)) \
                     + CIVILIAN_DEATH_BUFFER
        print(f"\nRECOMMENDED max_ticks (damage coverage "
              f"+ {CIVILIAN_DEATH_BUFFER}-tick death buffer): {damage_cap}")

    if smoke_sat_times:
        sat_cap = max(smoke_sat_times) + CIVILIAN_DEATH_BUFFER
        print(f"CONSERVATIVE max_ticks (saturation + buffer):                       {sat_cap}")

    # Flag any scenarios that didn't reach thresholds
    unreached = [r for r in results
                 if r['t_smoke_dmg'] is None or r['t_fire_dmg'] is None]
    if unreached:
        print(f"\nWARNING: {len(unreached)} scenario(s) did not reach damage "
              f"thresholds within {MAX_PROBE_TICKS} ticks:")
        for r in unreached:
            print(f"  {r['map']:<16} {r['fire']:<13} "
                  f"smoke_dmg={r['t_smoke_dmg']}  fire_dmg={r['t_fire_dmg']}")
        print("  -> These maps have isolated floor regions the fire cannot reach. "
              "Investigate map connectivity.")


if __name__ == "__main__":
    main()
