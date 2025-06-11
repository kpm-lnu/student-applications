import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

os.makedirs("figures1_2", exist_ok=True)

λ, ρ, β, δ, p, c = 1.0, 0.01, 0.2, 0.1, 5.0, 2.4
A, B, C1 = 10, 15, 250
U0, I0, V0 = 998, 10, 10
T = 30
dt = 0.1
t = np.arange(0, T + dt, dt)
N_steps = len(t)

m_vals = np.zeros_like(t)
m_func = lambda t_val: np.interp(t_val, t, m_vals)

def system_rhs(t, y, m_func):
    U, I, V = y
    m = m_func(t)
    dU = λ - ρ * U - (1 - m) * β * U * V
    dI = (1 - m) * β * U * V - δ * I
    dV = p * I - c * V
    return [dU, dI, dV]

def adjoint_rhs(t, phi, U_func, I_func, V_func, m_func):
    phi1, phi2, phi3 = phi
    U = U_func(t)
    I = I_func(t)
    V = V_func(t)
    m = m_func(t)
    dphi1 = phi1 * (ρ + (1 - m) * β * V) - phi2 * (1 - m) * β * V
    dphi2 = phi2 * δ - phi3 * p - A
    dphi3 = phi1 * (1 - m) * β * U - phi2 * (1 - m) * β * U + phi3 * c - B
    return [dphi1, dphi2, dphi3]

def optimal_m(U, I, V, phi1, phi2):
    raw = -((phi1 - phi2) * β * U * V) / C1
    return np.clip(raw, 0.0, 1.0)

for iter_num in range(200):
    sol = solve_ivp(system_rhs, [0, T], [U0, I0, V0], t_eval=t, args=(m_func,))
    U, I, V = sol.y

    U_func = lambda t_val: np.interp(t_val, t, U)
    I_func = lambda t_val: np.interp(t_val, t, I)
    V_func = lambda t_val: np.interp(t_val, t, V)

    phi0 = [0, 0, 0]
    sol_adj = solve_ivp(adjoint_rhs, [T, 0], phi0, t_eval=t[::-1],
                        args=(U_func, I_func, V_func, m_func))
    phi1 = np.interp(t, t[::-1], sol_adj.y[0])
    phi2 = np.interp(t, t[::-1], sol_adj.y[1])
    phi3 = np.interp(t, t[::-1], sol_adj.y[2])

    m_new = optimal_m(U, I, V, phi1, phi2)

    if np.linalg.norm(m_new - m_vals, np.inf) < 1e-4:
        print(f"Збіжність досягнута на ітерації {iter_num}")
        break
    m_vals = m_new
    m_func = lambda t_val: np.interp(t_val, t, m_vals)
else:
    print("⚠️ Збіжність не досягнута за 200 ітерацій.")

J = np.trapz(A * I + B * V + 0.5 * C1 * m_vals**2, t)
print(f"Функціонал якості J(m) = {J:.4f}")

plt.figure(figsize=(8, 3))
plt.plot(t, m_vals, lw=2)
plt.xlabel("Час (дні)"); plt.ylabel("m(t)")
plt.title("Оптимальне керування $m(t)$")
plt.grid(); plt.tight_layout()
plt.savefig("figures1_2/optimal_m_control.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(t, U, label="U (вразливі)")
plt.plot(t, I, label="I (інфіковані)")
plt.plot(t, V, label="V (віруси)")
plt.xlabel("Час (дні)"); plt.ylabel("Кількість")
plt.title("Динаміка U(t), I(t), V(t)")
plt.legend(); plt.grid(); plt.tight_layout()
plt.savefig("figures1_2/trajectories.png", dpi=300)
plt.close()

print("Графіки збережено у папці 'figures'")
