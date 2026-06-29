"""scenarios.py -- task-owned scenario geometry. Pure.

Provides VISIBLE (8 scenarios with the fixed golden stress mapping), HELD_OUT (a fixed
perturbation bank), and build_observation() which turns a scenario into the geometry that
plan()/measure() consume.

Coordinate convention (cm, z up):
  * start at x=0, seam at x=seam_x, both on the y=0 / z=TORCH_Z centerline.
  * the obstacle is a box centered on the travel midpoint, straddling y=0 (so a straight,
    no-detour path collides) and poking into +y by `obstacle_offset + OBS_POS`. Larger
    offset => the obstacle reaches further into the detour corridor => a bigger detour is
    required (difficulty rises with offset).
  * the detour is forced to the +y side; the restricted zone sits either just past the
    obstacle's +y face (the 'safety_zone' stress), as a high overhead band ('ceiling'
    stress), or behind it on the -y side (otherwise).

Per-stress geometry (legitimate task design, not a checkpoint comparison):
  * 'safety_zone' : a thin restricted-zone band hugs the obstacle's +y face, so a slightly
                    under-cleared shelf grazes the zone (intrusion) without entering the box.
  * 'collision'   : the obstacle's +y face is pushed much further out, so an under-detour
                    shelf lands inside the box (collision).
  * 'timeout'     : the seam is placed much farther away (longer required travel) so a slow
                    policy exceeds the time budget even though it clears the geometry.
  * 'ceiling'     : a small obstacle, but a wide restricted-zone CEILING hangs above it, so
                    an OVER-detour shelf rises into the zone (intrusion). This is the
                    upper-bound penalty that makes the task non-monotone in BOTH directions
                    and -- crucially -- makes a fixed, geometry-blind apex impossible: the
                    apex needed here is small while other scenarios need a large apex.
  * None          : an easy obstacle a modestly under-tuned policy still clears.

ANTI-OVERFIT INVARIANT (why no constant apex can pass the whole family):
  * the 'collision' VISIBLE scenario (idx 5) has a +y face at 48cm, so passing ALL VISIBLE
    requires apex > 48;
  * the 'ceiling' HELD_OUT scenario forbids apex >= CEIL_LO (48);
  these two windows do not overlap, so any geometry-blind constant apex that clears the
  visible set necessarily intrudes the held-out ceiling. Only an obstacle-RELATIVE policy
  (apex sized from each observed obstacle) clears both.
"""

OBS_NEG = 6.0  # how far the obstacle pokes into -y (straddles the direct line)
OBS_POS_BASE = 6.0  # +y face = offset + this, for ordinary obstacles
OBS_POS_COLLISION = (
    24.0  # 'collision' stress: far +y face so an under-detour hits the box
)
OBS_HALF_X = 12.0
OBS_Z_HI = 25.0
ZONE_W = 8.0  # thickness of the +y restricted-zone band ('safety_zone' stress)
ZONE_HALF_X = 18.0
CEIL_LO = 48.0  # 'ceiling' stress: restricted zone occupies y >= CEIL_LO (overhead)
CEIL_HI = 400.0  # tall enough that any over-detour shelf rises into it
CEIL_HALF_X = 70.0  # wide enough in x to cover the whole detour shelf
TORCH_Z = 5.0
SEAM_X_NORMAL = 120.0
SEAM_X_TIMEOUT = 300.0
TIME_BUDGET_S = 30.0


# index -> (obstacle_offset_cm, stress).  Golden: 3->safety_zone, 5->collision, 7->timeout.
VISIBLE = [
    {"obstacle_offset_cm": 8.0, "stresses": None},
    {"obstacle_offset_cm": 12.0, "stresses": None},
    {"obstacle_offset_cm": 16.0, "stresses": None},
    {"obstacle_offset_cm": 20.0, "stresses": "safety_zone"},
    {"obstacle_offset_cm": 18.0, "stresses": None},
    {"obstacle_offset_cm": 24.0, "stresses": "collision"},
    {"obstacle_offset_cm": 22.0, "stresses": None},
    {"obstacle_offset_cm": 34.0, "stresses": "timeout"},
]

# Fixed, seeded perturbation bank. Held out of every candidate eval copy. Designed so that
# NO single constant (geometry-blind) apex can clear it together with the visible set:
#  - the 'ceiling' row forbids apex >= 48;
#  - the large-offset rows (50,58 -> +y faces 56,64) require apex >= 56 / 64.
# An obstacle-relative policy sizes the detour per observation and clears all of them.
HELD_OUT = [
    {"obstacle_offset_cm": 9.0, "stresses": None},
    {"obstacle_offset_cm": 15.0, "stresses": None},
    {"obstacle_offset_cm": 21.0, "stresses": "safety_zone"},
    {"obstacle_offset_cm": 27.0, "stresses": "collision"},
    {"obstacle_offset_cm": 20.0, "stresses": "timeout"},
    {"obstacle_offset_cm": 24.0, "stresses": "ceiling"},
    {"obstacle_offset_cm": 50.0, "stresses": None},
    {"obstacle_offset_cm": 58.0, "stresses": None},
]


def build_observation(scenario: dict) -> dict:
    off = float(scenario["obstacle_offset_cm"])
    stress = scenario.get("stresses")

    seam_x = SEAM_X_TIMEOUT if stress == "timeout" else SEAM_X_NORMAL
    obs_pos = OBS_POS_COLLISION if stress == "collision" else OBS_POS_BASE

    mx = seam_x / 2.0

    y_lo = -OBS_NEG
    y_hi = off + obs_pos
    obs_center = [mx, 0.5 * (y_lo + y_hi), OBS_Z_HI / 2.0]
    obs_half = [OBS_HALF_X, 0.5 * (y_hi - y_lo), OBS_Z_HI / 2.0]

    if stress == "safety_zone":
        # thin band hugging the obstacle's +y face: y in [y_hi, y_hi + ZONE_W]
        z_lo, z_hi = y_hi, y_hi + ZONE_W
        zone_center = [mx, 0.5 * (z_lo + z_hi), OBS_Z_HI / 2.0]
        zone_half = [ZONE_HALF_X, 0.5 * (z_hi - z_lo), OBS_Z_HI / 2.0]
    elif stress == "ceiling":
        # wide overhead band: y in [CEIL_LO, CEIL_HI]; an over-detour shelf rises into it.
        zone_center = [mx, 0.5 * (CEIL_LO + CEIL_HI), OBS_Z_HI / 2.0]
        zone_half = [CEIL_HALF_X, 0.5 * (CEIL_HI - CEIL_LO), OBS_Z_HI / 2.0]
    else:
        # behind the obstacle on the -y side, out of the +y detour corridor
        zy_hi = y_lo
        zy_lo = y_lo - 16.0
        zone_center = [mx, 0.5 * (zy_lo + zy_hi), OBS_Z_HI / 2.0]
        zone_half = [ZONE_HALF_X, 0.5 * (zy_hi - zy_lo), OBS_Z_HI / 2.0]

    return {
        "start_pose": [0.0, 0.0, TORCH_Z],
        "seam_pose": [seam_x, 0.0, TORCH_Z],
        "obstacle_box": {"center": obs_center, "half_extents": obs_half},
        "restricted_zone": {"center": zone_center, "half_extents": zone_half},
        "time_budget_s": TIME_BUDGET_S,
        "obstacle_offset_cm": off,
        "stresses": stress,
    }
