"""Public identifiers for frozen OpenPI policies hosted by Cybernetics.

``pi0-droid`` returns one native ``[H, 8]`` absolute DROID joint-position and
gripper action chunk. A DSRL controller supplies one typed ``[32]`` action; the
SDK repeats it over PI0's ten-step initial flow-noise horizon while the hosted
PI0 weights remain immutable. The base runtime does not itself train, run SDE
rollouts, or predict video.
"""

from ...types.pi0_droid_dsrl_action import (
    PI0_DROID_DSRL_ACTION_SHAPE,
    PI0_DROID_INITIAL_FLOW_NOISE_CONTRACT_VERSION,
    PI0_DROID_INITIAL_FLOW_NOISE_SHAPE,
    Pi0DroidDsrlAction,
)

PI0_DROID_BASE_MODEL = "pi0-droid"
PI0_DROID_ACTION_SPACE = "droid_joint_position"

__all__ = [
    "PI0_DROID_ACTION_SPACE",
    "PI0_DROID_BASE_MODEL",
    "PI0_DROID_DSRL_ACTION_SHAPE",
    "PI0_DROID_INITIAL_FLOW_NOISE_CONTRACT_VERSION",
    "PI0_DROID_INITIAL_FLOW_NOISE_SHAPE",
    "Pi0DroidDsrlAction",
]
