from fenics import Constant, Expression
from dataclasses import dataclass
from collections import namedtuple
import numbers
import numpy as np

Velocity = namedtuple("Velocity", ["x", "y"])
Rectangle = namedtuple("Rectangle", ["x", "y"]) # Rectangle from (0; 0) to (x; y)

@dataclass
class AdrParams:
    diffusion: Constant
    reaction: Constant
    velocity: Velocity
    f_source: Expression
    plane: Rectangle
    N_mesh: int
    T_time: float
    M_time_iter: int
    u_exact: Expression

def exact(x, expression):
    if len(x) < 2:
        raise TypeError(f"len of x is less than 2, len(x) = {len(x)}")
    if len(x[0]) != len(x[1]):
        raise TypeError(f"len(x[0]): {len(x[0])} != len(x[1]): {len(x[1])}")
    res = []
    for a, b in zip(x[0], x[1]):
        val = expression(a, b)
        if not isinstance(val, numbers.Number):
            raise TypeError(f"Result value is not numbers, val = {val}, val type = {type(val)}")

        res.append(val)

    return res


def is_close(l, r):
    return abs(l - r) <  1.e-6

# def on_boundary(x, l, r, u, d):
#     return is_close(x[0], l) or is_close(x[0], r) or is_close(x[1], u) or is_close(x[1], d)

#     # return (
#     #     np.isclose(x[0], l) ors
#     #     np.isclose(x[0], r) or
#     #     np.isclose(x[1], u) or
#     #     np.isclose(x[1], d)
#     # )

def on_boundary(x, x0, x1, y0, y1, tol=1e-14):
    return np.logical_or.reduce((
        np.isclose(x[0], x0, atol=tol),
        np.isclose(x[0], x1, atol=tol),
        np.isclose(x[1], y0, atol=tol),
        np.isclose(x[1], y1, atol=tol)
    ))
