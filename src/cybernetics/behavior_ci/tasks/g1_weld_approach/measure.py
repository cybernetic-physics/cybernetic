"""measure.py -- pure, self-contained outcome measurement (impl-2).

measure(trajectory, observation) replays the trajectory geometrically and returns the
metrics dict. It is SHARED by the offline fixture and the hosted grader, so it must be
exact and depend only on its inputs (no randomness, no clock).

Everything is genuine geometry:
  * path length is summed segment lengths;
  * the torch only physically advances `speed * time_budget` of arc-length (truncation),
    so an over-long path leaves the tip short of the seam (target miss);
  * collision / intrusion are EXACT segment-vs-AABB slab intersections over the swept,
    actually-traversed portion of the path;
  * clearance is the min point-to-AABB distance sampled along the traversed path;
  * base tilt grows with the maximum lateral excursion away from the start->seam line
    (a wild/long detour destabilises the base).

Units: positions in cm, speed in m/s, time in s.
"""

import math

TILT_K = 0.06  # degrees of base tilt per cm of lateral excursion.
_SAMPLE_STEP_CM = 0.5


def _box_lo_hi(box):
    c = box["center"]
    h = box["half_extents"]
    lo = [c[i] - h[i] for i in range(3)]
    hi = [c[i] + h[i] for i in range(3)]
    return lo, hi


def _seg_aabb_intersect(p0, p1, lo, hi):
    """Exact segment-vs-AABB intersection (slab method). True if they touch/overlap."""
    tmin, tmax = 0.0, 1.0
    for i in range(3):
        d = p1[i] - p0[i]
        if abs(d) < 1e-12:
            if p0[i] < lo[i] - 1e-9 or p0[i] > hi[i] + 1e-9:
                return False
        else:
            t1 = (lo[i] - p0[i]) / d
            t2 = (hi[i] - p0[i]) / d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return False
    return True


def _point_aabb_dist(p, lo, hi):
    s = 0.0
    for i in range(3):
        if p[i] < lo[i]:
            d = lo[i] - p[i]
        elif p[i] > hi[i]:
            d = p[i] - hi[i]
        else:
            d = 0.0
        s += d * d
    return math.sqrt(s)


def _dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _lerp(a, b, t):
    return [a[i] + (b[i] - a[i]) * t for i in range(3)]


def _truncate(waypoints, s_max):
    """Return the polyline vertices from arc-length 0 up to s_max (cm)."""
    out = [list(waypoints[0])]
    acc = 0.0
    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        seg = _dist(a, b)
        if seg < 1e-12:
            continue
        if acc + seg <= s_max + 1e-9:
            out.append(list(b))
            acc += seg
        else:
            t = (s_max - acc) / seg
            out.append(_lerp(a, b, t))
            acc = s_max
            break
    return out


def _perp_dist_to_line(p, a, b):
    """Perpendicular distance from point p to the infinite line through a,b."""
    ab = [b[i] - a[i] for i in range(3)]
    ap = [p[i] - a[i] for i in range(3)]
    ab2 = sum(c * c for c in ab)
    if ab2 < 1e-12:
        return _dist(p, a)
    t = sum(ap[i] * ab[i] for i in range(3)) / ab2
    proj = [a[i] + ab[i] * t for i in range(3)]
    return _dist(p, proj)


def measure(trajectory: dict, observation: dict) -> dict:
    wps = [list(w) for w in trajectory["waypoints"]]
    speed = float(trajectory["speed_mps"])
    budget = float(observation["time_budget_s"])
    start = list(observation["start_pose"])
    seam = list(observation["seam_pose"])

    # full intended path length (cm)
    length_cm = sum(_dist(wps[i], wps[i + 1]) for i in range(len(wps) - 1))

    # time the full path WOULD take (s)
    if speed <= 0:
        elapsed = float("inf")
    else:
        elapsed = (length_cm / 100.0) / speed

    # arc-length the torch can physically cover within the time budget (cm)
    reachable_cm = speed * budget * 100.0 if speed > 0 else 0.0
    s_max = min(length_cm, reachable_cm)

    traversed = _truncate(wps, s_max)
    endpoint = traversed[-1]
    tip = _dist(endpoint, seam)

    obs_lo, obs_hi = _box_lo_hi(observation["obstacle_box"])
    zone_lo, zone_hi = _box_lo_hi(observation["restricted_zone"])

    collision = 0
    intrusion = 0
    for i in range(len(traversed) - 1):
        a, b = traversed[i], traversed[i + 1]
        if _seg_aabb_intersect(a, b, obs_lo, obs_hi):
            collision = 1
        if _seg_aabb_intersect(a, b, zone_lo, zone_hi):
            intrusion = 1

    # min clearance to the obstacle along the traversed path (sampled)
    min_clear = float("inf")
    for i in range(len(traversed) - 1):
        a, b = traversed[i], traversed[i + 1]
        seg = _dist(a, b)
        n = max(1, int(seg / _SAMPLE_STEP_CM))
        for k in range(n + 1):
            p = _lerp(a, b, k / n)
            d = _point_aabb_dist(p, obs_lo, obs_hi)
            if d < min_clear:
                min_clear = d
    if min_clear == float("inf"):
        min_clear = _point_aabb_dist(endpoint, obs_lo, obs_hi)

    # base tilt from maximum lateral excursion off the start->seam line
    max_lat = 0.0
    for p in traversed:
        lat = _perp_dist_to_line(p, start, seam)
        if lat > max_lat:
            max_lat = lat
    tilt = TILT_K * max_lat

    return {
        "torch_tip_distance_to_target_cm": tip,
        "collision_count": int(collision),
        "restricted_zone_intrusions": int(intrusion),
        "max_base_tilt_degrees": tilt,
        "elapsed_seconds": elapsed,
        "min_clearance_to_obstacle_cm": min_clear,
    }
