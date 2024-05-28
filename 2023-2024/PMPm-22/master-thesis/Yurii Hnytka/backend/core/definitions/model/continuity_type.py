"""Continuity Type Enum"""


from enum import Enum


class ContinuityType(Enum):
    """Continuity Type"""

    CONTINUOUSLY_DIFFERENTIABLE = 0
    CONTINUOUS = 1
    DISCONTINUOUS = 2
