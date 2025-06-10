import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt

# ----------------------------
# 1. Параметри моделі
mu_h = 0.04356
lambda_h = 0.03673
theta_h = 5.0e-14
gamma_h = 0.01587
delta_h = 3.0e-4
Nh = 32022600

mu_v = 1.13
lambda_v = 1.09589
theta_v = 4.5e-9
Nv = 256180800

# Параметри контролю
alpha1 = 0.7
alpha2 = 0.2
C1 = 0.01
C2 = 0.02

T = 1000
N = 1000
t = np.linspace(0, T, N)
dt = t[1] - t[0]
t_grid = t.copy()

# Початкові умови
Sh0, Ih0, Rh0 = Nh - 1, 1, 0
Sv0, Iv0 = Nv - 1, 1
y0 = [Sh0, Ih0, Rh0, Sv0, Iv0]

# ----------------------------
# 2. Без контролю
def deriv_no_control(y, t):
    Sh, Ih, Rh, Sv, Iv = y
    dSh = mu_h*Nh - lambda_h*Sh - theta_h*Iv*Sh + delta_h*Rh
    dIh = theta_h*Iv*Sh - (lambda_h + gamma_h)*Ih
    dRh = gamma_h*Ih - (lambda_h + delta_h)*Rh
    dSv = mu_v*Nv - lambda_v*Sv - theta_v*Iv*Sv
    dIv = theta_v*Iv*Sv - lambda_v*Iv
    return dSh, dIh, dRh, dSv, dIv

ret_no = odeint(deriv_no_control, y0, t)
Sh_no, Ih_no, Rh_no, Sv_no, Iv_no = ret_no.T

# ----------------------------
# 3. З контролем
def forward_backward_sweep(params, control_mask=(1,1), max_iter=50, tol=1e-6):
    mu_h, mu_v, lambda_h, lambda_v, theta_h, theta_v, \
    alpha1, alpha2, gamma_h, delta_h, C1, C2, Nh, Nv = params

    u1 = np.zeros(N)
    u2 = np.zeros(N)

    for _ in range(max_iter):
        u1_old, u2_old = u1.copy(), u2.copy()

        def state_rhs(ti, y):
            ui1 = np.interp(ti, t_grid, u1) * control_mask[0]
            ui2 = np.interp(ti, t_grid, u2) * control_mask[1]
            Sh, Ih, Rh, Sv, Iv = y
            dSh = mu_h*Nh - lambda_h*Sh - theta_h*(1 - alpha1*ui1 - alpha2*ui2)*Iv*Sh + delta_h*Rh
            dIh = theta_h*(1 - alpha1*ui1 - alpha2*ui2)*Iv*Sh - (lambda_h + gamma_h)*Ih
            dRh = gamma_h*Ih - (lambda_h + delta_h)*Rh
            dSv = mu_v*Nv - lambda_v*Sv - theta_v*Iv*Sv
            dIv = theta_v*Iv*Sv - lambda_v*Iv
            return [dSh, dIh, dRh, dSv, dIv]

        sol_fwd = solve_ivp(state_rhs, [0, T], y0, t_eval=t)
        Sh, Ih, Rh, Sv, Iv = sol_fwd.y

        def adjoint_rhs(ti, lam):
            idx = int((T - ti) / dt)
            idx = np.clip(idx, 0, N - 1)
            ui1, ui2 = u1[idx] * control_mask[0], u2[idx] * control_mask[1]
            Sh_i, Ih_i, Rh_i, Sv_i, Iv_i = Sh[idx], Ih[idx], Rh[idx], Sv[idx], Iv[idx]
            lam1, lam2, lam3, lam4, lam5 = lam

            dlam1 = -(-lam1*(lambda_h + theta_h*(1 - alpha1*ui1 - alpha2*ui2)*Iv_i)
                      + lam2*theta_h*(1 - alpha1*ui1 - alpha2*ui2)*Iv_i)
            dlam2 = -(1 + lam2*(-(lambda_h + gamma_h)) + lam3*gamma_h)
            dlam3 = -(lam1*delta_h + lam3*(-(lambda_h + delta_h)))
            dlam4 = -(lam4*(-(lambda_v + theta_v*Iv_i)) + lam5*theta_v*Iv_i)
            dlam5 = -(
                lam1*(-theta_h*(1 - alpha1*ui1 - alpha2*ui2)*Sh_i) +
                lam2*(theta_h*(1 - alpha1*ui1 - alpha2*ui2)*Sh_i) +
                lam5*(theta_v*Sv_i - lambda_v)
            )
            return [dlam1, dlam2, dlam3, dlam4, dlam5]

        sol_adj = solve_ivp(adjoint_rhs, [T, 0], [0, 0, 0, 0, 0], t_eval=t[::-1])
        lam = sol_adj.y[:, ::-1]

        common = theta_h * Iv * Sh
        u1 = np.clip(-alpha1 * common * (lam[0] - lam[1]) / C1, 0, 1) * control_mask[0]
        u2 = np.clip(-alpha2 * common * (lam[0] - lam[1]) / C2, 0, 1) * control_mask[1]

        if np.linalg.norm(u1 - u1_old) + np.linalg.norm(u2 - u2_old) < tol:
            break

    return t, Sh, Ih, Rh, Sv, Iv, u1, u2

params = (mu_h, mu_v, lambda_h, lambda_v, theta_h, theta_v,
          alpha1, alpha2, gamma_h, delta_h, C1, C2, Nh, Nv)

# З контролем
t_ctrl, Sh_ctrl, Ih_ctrl, Rh_ctrl, Sv_ctrl, Iv_ctrl, u1_ctrl, u2_ctrl = forward_backward_sweep(params, control_mask=(1, 1))
t_u1, _, Ih_u1, _, _, _, _, _ = forward_backward_sweep(params, control_mask=(1, 0))
t_u2, _, Ih_u2, _, _, _, _, _ = forward_backward_sweep(params, control_mask=(0, 1))

# ----------------------------
# 4. Побудова 9 графіків

plt.figure(figsize=(8, 5))
plt.plot(t, Sh_no, 'b--', label='Без контролю')
plt.plot(t_ctrl, Sh_ctrl, 'b-', label='З контролем')
plt.title('Сприйнятливі люди')
plt.xlabel('Час (дні)'); plt.ylabel('Кількість'); plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

plt.figure(figsize=(8, 5))
plt.plot(t, Ih_no, 'r--', label='I_h без контролю')
plt.plot(t_ctrl, Ih_ctrl, 'r-', label='I_h з контролем')
plt.plot(t, Rh_no, 'g--', label='R_h без контролю')
plt.plot(t_ctrl, Rh_ctrl, 'g-', label='R_h з контролем')
plt.title('Інфіковані та Одужалі люди')
plt.xlabel('Час (дні)'); plt.ylabel('Кількість'); plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

plt.figure(figsize=(8, 5))
plt.plot(t, Sv_no, 'b--', label='S_v без контролю')
plt.plot(t_ctrl, Sv_ctrl, 'b-', label='S_v з контролем')
plt.title('Сприйнятливі переносники')
plt.xlabel('Час (дні)'); plt.ylabel('Кількість'); plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

plt.figure(figsize=(8, 5))
plt.plot(t, Iv_no, 'r--', label='I_v без контролю')
plt.plot(t_ctrl, Iv_ctrl, 'r-', label='I_v з контролем')
plt.title('Інфіковані переносники')
plt.xlabel('Час (дні)'); plt.ylabel('Кількість'); plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

plt.figure(figsize=(10, 5))
plt.plot(t_ctrl, u1_ctrl, 'b-', label='u1(t) – Індивідуальний захист')
plt.plot(t_ctrl, u2_ctrl, 'r-', label='u2(t) – Гігієна')
plt.title('Оптимальні керуючі функції u1(t) та u2(t)')
plt.xlabel('Час (дні)'); plt.ylabel('Інтенсивність'); plt.ylim(-0.05, 1.05); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

plt.figure(figsize=(10, 5))
plt.plot(t, Ih_no, 'r--', label='$I_h$ без контролю')
plt.plot(t_ctrl, Ih_ctrl, 'r-', label='$I_h$ з контролем')
plt.plot(t, Rh_no, 'g--', label='$R_h$ без контролю')
plt.plot(t_ctrl, Rh_ctrl, 'g-', label='$R_h$ з контролем')
plt.title('Інфіковані ($I_h$) та Одужалі ($R_h$) люди')
plt.xlabel('Час (дні)'); plt.ylabel('Кількість осіб'); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

plt.figure(figsize=(8, 5))
plt.plot(t, Ih_no - Ih_ctrl, label='Різниця: без - з контролем')
plt.title('Ефективність контролю')
plt.xlabel('Час (дні)'); plt.ylabel('Різниця кількості інфікованих'); plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

plt.figure(figsize=(10, 5))
plt.plot(t_ctrl, u1_ctrl, label='u1 (індивідуальний захист)')
plt.plot(t_ctrl, u2_ctrl, label='u2 (гігієна)')
plt.title('Динаміка заходів індивідуального захисту та гігієни у часі')
plt.xlabel('Час (дні)'); plt.ylabel('Інтенсивність'); plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

plt.figure(figsize=(12, 6))
plt.plot(t, Ih_no, 'k--', label='Без контролю')
plt.plot(t_u1, Ih_u1, 'r-', label='Тільки u1')
plt.plot(t_u2, Ih_u2, 'b-', label='Тільки u2')
plt.plot(t_ctrl, Ih_ctrl, 'g-', label='u1 та u2')
plt.title('Порівняння ефективності контролю')
plt.xlabel('Час (дні)'); plt.ylabel('Інфіковані'); plt.legend(); plt.grid(); plt.tight_layout(); plt.show()
