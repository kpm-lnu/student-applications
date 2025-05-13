from fenics import Constant, Expression
from dataclasses import dataclass
from collections import namedtuple
import subprocess
import os

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
    boundary: list [Expression]
    initial: Expression
    u_exact: Expression


def paraview_show_results(file_name: str):
    pvd_path = os.path.abspath(file_name)
    if os.path.exists(pvd_path):
        try:
            subprocess.Popen(["paraview", '-d',  pvd_path])
        except FileNotFoundError:
            print("Paraview is not installed!")
    else:
        print("Paraview file with results no found!")
