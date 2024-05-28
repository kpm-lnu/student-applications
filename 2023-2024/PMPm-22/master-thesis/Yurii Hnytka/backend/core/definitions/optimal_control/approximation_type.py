"""Approximation type Definition"""


from enum import Enum


class ApproximationType(str, Enum):
    """Approximation Type"""

    PIECEWISE_CONSTANT = 'piecewise-constant'
    PIECEWISE_LINEAR = 'piecewise-linear'
