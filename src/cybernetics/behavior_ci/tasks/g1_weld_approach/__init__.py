"""Behavior CI Task Pack: g1_weld_approach.

Platform-owned trust domain. The eval thresholds, scenario geometry, action planner,
outcome measurement, in-session grader, saved-scene env_id, and the held-out perturbation
bank all live here, INSIDE the pip-installed SDK -- not in the candidate policy repo. A
policy PR can change only its opaque checkpoint; it cannot reach these bytes.
"""
