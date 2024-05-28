"""
Convection-diffusion-reaction pde:

  du/dt - D(dot(grad(u), grad(u))) + dot(v, grad(u)) + Ku = f

"""


import sys
import argparse
import matplotlib.pyplot as plt
from fenics import *
import numpy as np


def boundary(x, on_boundary):
    return on_boundary

# parser = argparse.ArgumentParser(
#                     prog='Convection-diffusion-reaction pde',
#                     description='Solve CDF pde',
#                     epilog='--plot   Show additional plots')
# parser.add_argument('-p', '--plot', action='store_true', default=False)
# parser.add_argument('-p', '--plot', action='store_true', default=False)
# args = parser.parse_args()
# SHOW_PLOT = args.plot

SHOW_PLOT = False


if len(sys.argv) == 1:
    sys.argv.append('1')

if sys.argv[1] == '1':
# Example 1:
    print('du/dt - 0.001(dot(grad(u), grad(u))) + dot([1,1], grad(u)) + u = 0')
# u_exact = e^((1/2)*(0.001*t - x - y))
# -----------------------------------------------------------------------------
    D = Constant(0.1)  # Diffusion
    # D = Constant(0.001)  # Diffusion
    K = Constant(1.0)    # Reaction
    v_x = Constant(1.0)  # Velocity x
    v_y = Constant(1.0)  # Velocity y
    b = Expression((('v_x', 'v_y')), v_x=v_x, v_y=v_y, degree=2, t=0)
    f = Expression('0', degree=1, alpha=D, t=0)
    u_exact = Expression('exp((alpha*t -x[0] - x[1])/2)', degree=2, alpha=D, t=0)
# -----------------------------------------------------------------------------

elif sys.argv[1] == '2':

# Example 2:
    print('du/dt - 0.001(dot(grad(u), grad(u))) + dot([1,2], grad(u)) + 2u = (1/2)*e^((1/2)*(0.001*t - x - y))')
# u_exact = e^((1/2)*(0.001*t - x - y))
# -----------------------------------------------------------------------------
    D = Constant(0.001)  # Diffusion
    K = Constant(2.0)    # Reaction
    v_x = Constant(1.0)  # Velocity x
    v_y = Constant(2.0)  # Velocity y
    b = Expression((('v_x', 'v_y')), v_x=v_x, v_y=v_y, degree=2, t=0)
    f = Expression('exp((alpha*t -x[0] - x[1])/2)/2', degree=1, alpha=D, t=0)
    u_exact = Expression('exp((alpha*t -x[0] - x[1])/2)', degree=2, alpha=D, t=0)
# -----------------------------------------------------------------------------

else:
    raise Exception(f"Enter the correct example number, not {sys.argv[1]}")


N = 20
T = 4.0
M = 20

# Create mesh and define function space
a = 2   # [0; a] x [0; a]
mesh = RectangleMesh(Point(0, 0), Point(a, a), N, N)

POW = 3
# Bubble
# V = FunctionSpace(mesh, "B", 3)
# Lagrange
V = FunctionSpace(mesh, "P", POW)


# Define boundary condition
bc = DirichletBC(V, u_exact, boundary)
# Define initial value
u_n = interpolate(u_exact, V)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

dt = T / M
k = Constant(dt)

# main weak formulation
F = ((u - u_n) / k)*v*dx + D*dot(grad(u), grad(v)) * \
    dx + (dot(grad(u), b) + K*u) * v*dx - f*v*dx

a, L = lhs(F), rhs(F)

# Time-stepping
u = Function(V)
t = 0

errors = []  # List to store the errors
vtkfile = File('results/solution-final.pvd')

for n in range(M):
    # Update current time
    t += dt
    u_exact.t = t
    b.t = t
    f.t = t

    # Compute solution
    solve(a == L, u, bc)

    vtkfile << u

    u_v_exact = interpolate(u_exact, V)
    # Compute error at vertices
    error_t = errornorm(u_v_exact, u, 'L2')
    errors.append(error_t)

    # Update previous solution
    u_n.assign(u)


time_avg_error = np.sum(errors) / M
print('Time-averaged L2 error:', time_avg_error)
print('Time-max L2 error:', np.max(errors))

# Hold plot
plt.plot(np.linspace(0, T, M), errors)
plt.xlabel('Time')
plt.ylabel('L2 Error')
plt.show()

# plot(u)
# plot(mesh)
# plt.show()

# plot(u_e)
# plt.show()
