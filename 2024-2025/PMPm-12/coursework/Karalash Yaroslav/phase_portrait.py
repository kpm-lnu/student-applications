import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

gamma_h = 0.04
lambda_hI = 0.025
r1 = 0.005
lambda_hS = 0.0001
r2 = 0.0025
mu_rI = 0.02
lambda_rI = 0.04
gamma_r = 0.05
theta = 0.5
alpha = 0.2
delta = 0.1

E2 = [0.336, 0.64569, 0.6, 1.2249] 

initial_conditions = [
    [E2[0] - 0.1, E2[1] - 0.12, E2[2] - 0.12, E2[3] - 0.1],
    [E2[0] - 0.1, E2[1] - 0.12, E2[2] - 0.1, E2[3] - 0.1],
    [E2[0] - 0.1, E2[1] - 0.1, E2[2] - 0.1, E2[3] - 0.1],
    [E2[0] - 0.1, E2[1] - 0.1, E2[2] - 0.12, E2[3] - 0.1],
]

t = np.linspace(0, 200, 20000)

def system(vars, t, gamma_h, lambda_hI, r1, lambda_hS, r2, mu_rI, lambda_rI, gamma_r, alpha, delta, theta):
    I_h, R_h, I_r, E = vars
    dIh_dt = gamma_h * I_r * (1 - I_h - R_h) + theta * E * (1 - I_h - R_h) - I_h * (lambda_hI + r1)
    dRh_dt = r1 * I_h - R_h * (lambda_hS + r2)
    dIr_dt = mu_rI * I_r - lambda_rI * I_r + gamma_r * I_r * (1 - I_r)
    dE_dt = alpha * I_r - delta * E
    return [dIh_dt, dRh_dt, dIr_dt, dE_dt]

trajectories = []
for ic in initial_conditions:
    sol = odeint(system, ic, t, args=(gamma_h, lambda_hI, r1, lambda_hS, r2, mu_rI, lambda_rI, gamma_r, alpha, delta, theta))
    trajectories.append(sol)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

colors = ['r', 'g', 'b', 'k']

for sol, color in zip(trajectories, colors):
    I_h, R_h, I_r, E = sol.T
    ax.plot(I_h, R_h, I_r, color=color, lw=1)

    skip = 200
    for i in range(0, len(I_h) - skip, skip):
        dx = I_h[i + skip] - I_h[i]
        dy = R_h[i + skip] - R_h[i]
        dz = I_r[i + skip] - I_r[i]

        norm = np.sqrt(dx**2 + dy**2 + dz**2)
        if norm == 0:
            continue

        ax.quiver(I_h[i], R_h[i], I_r[i],
                  dx / norm, dy / norm, dz / norm,
                  length=0.025, color=color, linewidth=1)

ax.set_xlabel('I_h')
ax.set_ylabel('R_h')
ax.set_zlabel('I_r')
ax.set_title('Фазовий портрет системи поблизу точки E₂')
ax.view_init(elev=25, azim=-140)
plt.tight_layout()
plt.show()
