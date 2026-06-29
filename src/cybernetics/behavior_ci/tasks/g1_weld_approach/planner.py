"""planner.py -- pure trajectory planner for the welding-arm anti-gaming core.

The policy is an opaque CHECKPOINT (dict). plan() turns it into an actual TRAJECTORY
(a list of 3D waypoints + a travel speed). It does NOT emit any "clearance" number that
a grader trusts -- the environment (measure.py) independently re-derives every metric
from the geometry of the waypoints versus the per-scenario obstacle boxes.

------------------------------------------------------------------------------------
CHECKPOINT SCHEMA (all distances in cm, speed in m/s)
------------------------------------------------------------------------------------
  detour_mode        : 'relative' | 'absolute'
                       'relative' -> the lateral detour is computed FROM the observed
                                     obstacle geometry (genuinely obstacle-relative).
                       'absolute' -> a fixed canned apex independent of geometry
                                     (this is the over-fit / geometry-blind attack).
  detour_gain        : float   (relative mode) multiplies the obstacle's far (+y) face.
  clearance_margin_cm: float   (relative mode) additive safety margin past the obstacle.
  absolute_apex_cm   : float   (absolute mode) the fixed lateral apex to detour to.
  top_halfwidth_cm   : float   half-width (in the travel/x direction) of the flat top of
                               the detour. Must exceed the obstacle's x half-extent or the
                               sloped legs of the path will clip the box.
  approach_speed_mps : float   travel speed; sets elapsed time = path_length / speed and
                               the reachable arc-length within the time budget. Clamped to
                               [0, MAX_SPEED_MPS] so a policy cannot defeat the time budget
                               with a physically impossible torch speed.

The planner routes a TRAPEZOID: start -> P1 -> P2 -> seam, where P1/P2 are a flat lateral
"shelf" at height `apex` spanning the obstacle in x. The shelf height `apex` is the only
thing that decides whether the swept path clears the box / restricted zone, and it is a
real geometric coordinate, not a trusted scalar.

The design is deliberately NON-MONOTONE in the detour size:
  * too small an apex -> the shelf is inside the obstacle (collision) or grazes the
    restricted zone (intrusion);
  * too large an apex -> the wide lateral excursion destabilises the base (tilt > limit),
    the path is long (elapsed >= budget => timeout, and the torch is truncated by the time
    budget so it never reaches the seam => target miss), AND -- on scenarios with a
    "ceiling" restricted zone above the obstacle -- the shelf rises into the ceiling
    (intrusion).
There is therefore no single apex a policy can crank to pass; and crucially no FIXED apex
clears the whole scenario family, because some scenarios need a large detour while others
penalise one. The policy MUST size the detour to the actual observed obstacle.
"""

# Physical ceiling on torch travel speed (m/s). A real welding end-effector does not move
# at 100 m/s; clamping here stops "crank the speed" from trivially satisfying the timeout
# check while leaving every geometric safety check (collision/intrusion/tilt) unaffected.
MAX_SPEED_MPS = 2.0


def plan(checkpoint: dict, observation: dict) -> dict:
    start = list(observation["start_pose"])
    seam = list(observation["seam_pose"])
    obs = observation["obstacle_box"]
    oc = obs["center"]
    oh = obs["half_extents"]

    # Far (+y) face of the obstacle as seen in the observation geometry.
    obstacle_top_y = oc[1] + oh[1]

    mode = checkpoint.get("detour_mode", "relative")
    if mode == "absolute":
        # geometry-blind: a fixed apex regardless of where the obstacle actually is.
        apex = float(checkpoint["absolute_apex_cm"])
    else:
        gain = float(checkpoint.get("detour_gain", 1.0))
        margin = float(checkpoint.get("clearance_margin_cm", 0.0))
        apex = gain * obstacle_top_y + margin

    thw = float(checkpoint.get("top_halfwidth_cm", 30.0))
    speed = float(checkpoint.get("approach_speed_mps", 0.1))
    speed = max(0.0, min(speed, MAX_SPEED_MPS))

    mx = 0.5 * (start[0] + seam[0])
    z = start[2]

    p1 = [mx - thw, apex, z]
    p2 = [mx + thw, apex, z]

    waypoints = [list(start), p1, p2, list(seam)]
    return {"waypoints": waypoints, "speed_mps": speed}
