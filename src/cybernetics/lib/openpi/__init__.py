"""Public identifiers for inference-only OpenPI policies hosted by Cybernetics.

``pi0-droid`` returns one native ``[H, 8]`` absolute DROID joint-position and
gripper action chunk. It does not support training, SDE, or predicted video.
"""

PI0_DROID_BASE_MODEL = "pi0-droid"
PI0_DROID_ACTION_SPACE = "droid_joint_position"

__all__ = ["PI0_DROID_ACTION_SPACE", "PI0_DROID_BASE_MODEL"]
