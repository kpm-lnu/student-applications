import numpy as np
import time
import warnings
import os
import pickle
import hashlib
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.stats import qmc
from scipy.linalg import solve
from scipy.spatial.distance import pdist

import simulation_engine
from simulation_engine import (
    N_VARS, i_S_H, i_I_H, i_R_H, i_I_L, i_R_L, 
    i_S_R, i_I_R, i_S_A, i_I_A, i_W
)

warnings.filterwarnings('ignore', category=RuntimeWarning)

A1, A2 = 5000.0, 5000.0  
B1, B2 = 100.0, 100.0        
U_MAX = 0.5

DAYS_PER_MONTH = 30
N_MONTHS = 12
N_CONTROLS = 2 
TOTAL_PARAMS = N_MONTHS * N_CONTROLS
BOUNDS = [(0.0, U_MAX)] * TOTAL_PARAMS

CACHE_FILENAME = "opt_control_bcd_cache.pkl"

try:
    DEFAULT_PARAMS = simulation_engine.DEFAULT_PARAMS
except AttributeError:
    DEFAULT_PARAMS = {
        'gamma_H': 1 / 14, 'gamma_L': 1 / 10, 'omega_H': 1 / 180, 'omega_L': 1 / 180,
        'base_beta_R': 0.008, 'base_beta_H': 0.001, 'base_Lambda_R': 2.0, 'base_mu_R': 0.08,
        'base_beta_A': 0.01, 'base_beta_L': 0.005, 'base_beta_AL': 0.0001, 
        'base_Lambda_A': 5.0, 'base_mu_A': 0.08, 'gamma_A': 1 / 90, 
        'alpha_A': 0.1, 'base_delta_W': 0.03, 'm_R': 0.00001, 'm_A': 0.000001,
    }

class GERBF:
    def __init__(self, bounds):
        self.bounds = np.array(bounds)
        self.lb = self.bounds[:, 0]
        self.range = self.bounds[:, 1] - self.lb
        
    def _normalize(self, X): 
        return (np.array(X) - self.lb) / self.range
    
    def train(self, X, Y, dY):
        self.X_norm = self._normalize(X)
        self.N, self.D = self.X_norm.shape
        med = np.median(pdist(self.X_norm)) if np.median(pdist(self.X_norm)) > 1e-12 else 0.5
        self.gamma = 1.0 / (2.0 * med**2)
        
        self.y_std = np.std(Y) if np.std(Y) > 1e-12 else 1.0
        self.y_mean = np.mean(Y)
        Y_norm = (Y - self.y_mean) / self.y_std
        
        dY_norm = dY * (self.range / self.y_std)

        size = self.N * (self.D + 1)
        self.F = np.zeros(size)
        self.F[0::self.D+1] = Y_norm
        for d in range(self.D): 
            self.F[1+d::self.D+1] = dY_norm[:, d]
        
        M = np.zeros((size, size))
        diff = self.X_norm[:, np.newaxis, :] - self.X_norm[np.newaxis, :, :]
        phi = np.exp(-self.gamma * np.sum(diff**2, axis=-1))
        g2 = 2.0 * self.gamma
        
        M[0::self.D+1, 0::self.D+1] = phi
        for d in range(self.D):
            dphi = g2 * diff[:, :, d] * phi
            M[0::self.D+1, 1+d::self.D+1] = dphi
            M[1+d::self.D+1, 0::self.D+1] = -dphi
        for d1 in range(self.D):
            for d2 in range(self.D):
                M[1+d1::self.D+1, 1+d2::self.D+1] = g2 * (1.0*(d1==d2) - g2 * diff[:, :, d1] * diff[:, :, d2]) * phi
        
        success = False
        for reg_base in [1e-6, 1e-5, 1e-4, 1e-3]:
            reg_vec = np.zeros(size)
            reg_vec[:self.N] = reg_base              
            reg_vec[self.N:] = reg_base * 1000.0        
            try:
                self.W = solve(M + np.diag(reg_vec), self.F, assume_a='pos')
                if np.max(np.abs(self.W)) < 1e7: 
                    success = True
                    break
            except np.linalg.LinAlgError: 
                pass
            
        if not success:
            reg_vec = np.ones(size) * 1e-2
            self.W = np.linalg.lstsq(M + np.diag(reg_vec), self.F, rcond=None)[0]

    def predict(self, x):
        diff = self._normalize(np.atleast_2d(x))[:, np.newaxis, :] - self.X_norm
        phi = np.exp(-self.gamma * np.sum(diff**2, axis=-1))
        k = np.zeros(self.N * (self.D + 1))
        k[0::self.D+1] = phi[0]
        for d in range(self.D): 
            k[1+d::self.D+1] = 2.0 * self.gamma * diff[0, :, d] * phi[0]
        val = (k @ self.W) * self.y_std + self.y_mean
        return max(val, 1e-6)
        
    def predict_gradient(self, x):
        diff = self._normalize(np.atleast_2d(x))[0] - self.X_norm
        phi = np.exp(-self.gamma * np.sum(diff**2, axis=-1))
        g2 = 2.0 * self.gamma
        W_vals = self.W[0::self.D+1]
        grad_norm = np.zeros(self.D)
        for d1 in range(self.D):
            grad_norm[d1] += np.sum(W_vals * (-g2 * diff[:, d1] * phi))
            for d2 in range(self.D):
                grad_norm[d1] += np.sum(self.W[1+d2::self.D+1] * (g2 * (1.0*(d1==d2) - g2 * diff[:, d1] * diff[:, d2]) * phi))
        return grad_norm * (self.y_std / self.range)

def multistart_optimize(f, bounds, jac=None, n_starts=3, prev_best=None):
    lb, ub = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
    starts = lb + qmc.LatinHypercube(d=len(bounds), seed=42).random(n=n_starts) * (ub - lb)
    starts = np.vstack([np.zeros(len(bounds)), starts])
    
    if prev_best is not None:
        starts = np.vstack([starts, prev_best])
        
    best_x, best_val = None, np.inf
    for x0 in starts:
        res = minimize(f, x0, method='L-BFGS-B', jac=jac, bounds=bounds, options={'maxiter': 100})
        if res.fun < best_val: 
            best_val, best_x = res.fun, res.x
    return best_x

def precompute_climate(climate_df, oblast_order, n_regions):
    climate_fast = np.zeros((367, n_regions, 3)) 
    for day in range(1, 367):
        df_day = climate_df[climate_df['day_of_year'] == day]
        for i, name in enumerate(oblast_order):
            row = df_day[df_day['oblast'] == name]
            if not row.empty:
                climate_fast[day, i, 0] = row['temperature_avg'].values[0]
                climate_fast[day, i, 1] = row['humidity_avg'].values[0]
                climate_fast[day, i, 2] = row['precipitation_sum'].values[0]
            elif not df_day.empty:
                climate_fast[day, i, 0] = df_day.iloc[0]['temperature_avg']
                climate_fast[day, i, 1] = df_day.iloc[0]['humidity_avg']
                climate_fast[day, i, 2] = df_day.iloc[0]['precipitation_sum']
    return climate_fast

def compute_dp(t, climate_fast, reg_i, params, start_day):
    day = int((t) % 365) + 1
    day = max(1, min(366, day))
    temp, humidity, precip = climate_fast[day, reg_i, 0], climate_fast[day, reg_i, 1], climate_fast[day, reg_i, 2]
    
    return {
        'Lambda_R': params['base_Lambda_R'] * max(0.5, 1 + (temp - 10) / 30),
        'mu_R': params['base_mu_R'] * max(0.8, 1 - (temp - 10) / 40),
        'Lambda_A': params['base_Lambda_A'] * max(0.5, 1 + (temp - 10) / 30),
        'mu_A': params['base_mu_A'] * max(0.8, 1 - (temp - 10) / 40),
        'delta_W': params['base_delta_W'] * max(0.1, 1 + (temp - 15) / 20),
        'beta_L': params['base_beta_L'] * (1 + precip / 10),
        'beta_A': params['base_beta_A'] * (1 + precip / 10),
        'beta_H': params['base_beta_H'] * (1 + humidity / 100),
        'beta_R': params['base_beta_R'] * (1 + humidity / 100),
        'beta_AL': params['base_beta_AL'] * (1 + humidity / 100)
    }

def global_forward_model(t, y, climate_fast, params, neighbors_indexed, n_regions, u_schedule, start_day):
    state = y.reshape((n_regions, N_VARS))
    dydt = np.zeros_like(state)
    
    m_idx = min(int((t - start_day) // DAYS_PER_MONTH), N_MONTHS - 1)
    if m_idx < 0: m_idx = 0

    for i in range(n_regions):
        S_H, I_H, R_H, I_L, R_L, S_R, I_R, S_A, I_A, W = state[i]
        u1, u2 = u_schedule[i, m_idx, 0], u_schedule[i, m_idx, 1]
        
        dp = compute_dp(t, climate_fast, i, params, start_day)
        
        I_R_s, I_A_s, W_s = max(0.0, I_R), max(0.0, I_A), max(0.0, W)
        S_R_s, S_A_s = max(0.0, S_R), max(0.0, S_A)
        S_H_used = max(0.0, 1.0 - max(0.0, I_H) - max(0.0, R_H) - max(0.0, I_L) - max(0.0, R_L))

        l_H = dp['beta_H'] * I_R_s
        l_L = dp['beta_L'] * W_s + dp['beta_AL'] * I_A_s
        eff_mu_R, eff_delta_W = dp['mu_R'] + u1, dp['delta_W'] + u2

        mig_S_R, mig_I_R, mig_S_A, mig_I_A = 0, 0, 0, 0
        for j in neighbors_indexed[i]:
            mig_S_R += params['m_R'] * state[j, i_S_R]
            mig_I_R += params['m_R'] * state[j, i_I_R]
            mig_S_A += params['m_A'] * state[j, i_S_A]
            mig_I_A += params['m_A'] * state[j, i_I_A]
            
        num_neighbors = len(neighbors_indexed[i])
        mig_S_R -= num_neighbors * params['m_R'] * S_R
        mig_I_R -= num_neighbors * params['m_R'] * I_R
        mig_S_A -= num_neighbors * params['m_A'] * S_A
        mig_I_A -= num_neighbors * params['m_A'] * I_A

        dS_H = -l_H * S_H_used - l_L * S_H_used + params['omega_H']*R_H + params['omega_L']*R_L
        dI_H = l_H * S_H_used - params['gamma_H'] * I_H
        dR_H = params['gamma_H'] * I_H - params['omega_H'] * R_H
        dI_L = l_L * S_H_used - params['gamma_L'] * I_L
        dR_L = params['gamma_L'] * I_L - params['omega_L'] * R_L
        dS_R = dp['Lambda_R'] - dp['beta_R'] * S_R_s * I_R_s - eff_mu_R * S_R_s + mig_S_R
        dI_R = dp['beta_R'] * S_R_s * I_R_s - eff_mu_R * I_R_s + mig_I_R
        dS_A = dp['Lambda_A'] - dp['beta_A'] * S_A_s * W_s - dp['mu_A'] * S_A_s + mig_S_A
        dI_A = dp['beta_A'] * S_A_s * W_s - (dp['mu_A'] + params['gamma_A']) * I_A_s + mig_I_A
        dW = params['alpha_A'] * I_A_s - eff_delta_W * W_s

        dydt[i] = [dS_H, dI_H, dR_H, dI_L, dR_L, dS_R, dI_R, dS_A, dI_A, dW]

    return dydt.flatten()

def run_global_simulation(y0_1d, climate_fast, params, neighbors_indexed, n_regions, u_schedule, start_day, duration):
    T = start_day + duration
    t_eval = np.arange(start_day, T + 1, 1)
    
    sol = solve_ivp(fun=global_forward_model, t_span=(start_day, T), y0=y0_1d, t_eval=t_eval,
                    args=(climate_fast, params, neighbors_indexed, n_regions, u_schedule, start_day), 
                    method='RK45', rtol=1e-3, atol=1e-3)
    
    y_history = sol.y.reshape((n_regions, N_VARS, len(t_eval)))
    return y_history, t_eval

def local_forward_model(t, y, reg_i, u_schedule_local, climate_fast, params, start_day, num_neighbors, mig_in_func):
    dp = compute_dp(t, climate_fast, reg_i, params, start_day)
    mig_in_S_R, mig_in_I_R, mig_in_S_A, mig_in_I_A = mig_in_func(t)
    
    S_H, I_H, R_H, I_L, R_L, S_R, I_R, S_A, I_A, W = y
    m_idx = min(int((t - start_day) // DAYS_PER_MONTH), N_MONTHS - 1)
    if m_idx < 0: m_idx = 0
    u1, u2 = u_schedule_local[m_idx, 0], u_schedule_local[m_idx, 1]

    I_R_s, I_A_s, W_s = max(0.0, I_R), max(0.0, I_A), max(0.0, W)
    S_R_s, S_A_s = max(0.0, S_R), max(0.0, S_A)
    S_H_used = max(0.0, 1.0 - max(0.0, I_H) - max(0.0, R_H) - max(0.0, I_L) - max(0.0, R_L))

    l_H = dp['beta_H'] * I_R_s
    l_L = dp['beta_L'] * W_s + dp['beta_AL'] * I_A_s

    mig_S_R = mig_in_S_R - num_neighbors * params['m_R'] * S_R
    mig_I_R = mig_in_I_R - num_neighbors * params['m_R'] * I_R
    mig_S_A = mig_in_S_A - num_neighbors * params['m_A'] * S_A
    mig_I_A = mig_in_I_A - num_neighbors * params['m_A'] * I_A

    dS_H = -l_H * S_H_used - l_L * S_H_used + params['omega_H']*R_H + params['omega_L']*R_L
    dI_H = l_H * S_H_used - params['gamma_H'] * I_H
    dR_H = params['gamma_H'] * I_H - params['omega_H'] * R_H
    dI_L = l_L * S_H_used - params['gamma_L'] * I_L
    dR_L = params['gamma_L'] * I_L - params['omega_L'] * R_L
    dS_R = dp['Lambda_R'] - dp['beta_R'] * S_R_s * I_R_s - (dp['mu_R'] + u1) * S_R_s + mig_S_R
    dI_R = dp['beta_R'] * S_R_s * I_R_s - (dp['mu_R'] + u1) * I_R_s + mig_I_R
    dS_A = dp['Lambda_A'] - dp['beta_A'] * S_A_s * W_s - dp['mu_A'] * S_A_s + mig_S_A
    dI_A = dp['beta_A'] * S_A_s * W_s - (dp['mu_A'] + params['gamma_A']) * I_A_s + mig_I_A
    dW = params['alpha_A'] * I_A_s - (dp['delta_W'] + u2) * W_s

    return [dS_H, dI_H, dR_H, dI_L, dR_L, dS_R, dI_R, dS_A, dI_A, dW]

def local_adjoint_model(t, lam, y_fwd_func, reg_i, u_schedule_local, climate_fast, params, start_day, num_neighbors):
    y = y_fwd_func(t)
    m_idx = min(int((t - start_day) // DAYS_PER_MONTH), N_MONTHS - 1)
    if m_idx < 0: m_idx = 0
    u1, u2 = u_schedule_local[m_idx, 0], u_schedule_local[m_idx, 1]
    dp = compute_dp(t, climate_fast, reg_i, params, start_day)

    S_H, I_H, R_H, I_L, R_L, S_R, I_R, S_A, I_A, W = y
    lam_S_H, lam_I_H, lam_R_H, lam_I_L, lam_R_L, lam_S_R, lam_I_R, lam_S_A, lam_I_A, lam_W = lam

    S_H_s, S_R_s, S_A_s = max(0.0, S_H), max(0.0, S_R), max(0.0, S_A)
    I_R_s, I_A_s, W_s = max(0.0, I_R), max(0.0, I_A), max(0.0, W)

    dlam_S_H = (-dp['beta_H'] * I_R_s * lam_I_H - (dp['beta_AL'] * I_A_s + dp['beta_L'] * W_s) * lam_I_L + (dp['beta_AL'] * I_A_s + dp['beta_H'] * I_R_s + dp['beta_L'] * W_s) * lam_S_H)
    dlam_I_H = -A1 + params['gamma_H'] * lam_I_H - params['gamma_H'] * lam_R_H
    dlam_R_H = params['omega_H'] * (lam_R_H - lam_S_H)
    dlam_I_L = -A2 + params['gamma_L'] * lam_I_L - params['gamma_L'] * lam_R_L
    dlam_R_L = params['omega_L'] * (lam_R_L - lam_S_H)
    
    dlam_S_R = (-dp['beta_R'] * I_R_s * lam_I_R + (dp['beta_R'] * I_R_s + dp['mu_R'] + u1) * lam_S_R) + num_neighbors * params['m_R'] * lam_S_R
    dlam_I_R = (-dp['beta_H'] * S_H_s * lam_I_H + dp['beta_H'] * S_H_s * lam_S_H + dp['beta_R'] * S_R_s * lam_S_R + (-dp['beta_R'] * S_R_s + dp['mu_R'] + u1) * lam_I_R) + num_neighbors * params['m_R'] * lam_I_R
    dlam_S_A = (-dp['beta_A'] * W_s * lam_I_A + (dp['beta_A'] * W_s + dp['mu_A']) * lam_S_A) + num_neighbors * params['m_A'] * lam_S_A
    dlam_I_A = (-dp['beta_AL'] * S_H_s * lam_I_L + dp['beta_AL'] * S_H_s * lam_S_H - params['alpha_A'] * lam_W + (params['gamma_A'] + dp['mu_A']) * lam_I_A) + num_neighbors * params['m_A'] * lam_I_A
    dlam_W = (-dp['beta_A'] * S_A_s * lam_I_A + dp['beta_A'] * S_A_s * lam_S_A - dp['beta_L'] * S_H_s * lam_I_L + dp['beta_L'] * S_H_s * lam_S_H + (dp['delta_W'] + u2) * lam_W)

    return [dlam_S_H, dlam_I_H, dlam_R_H, dlam_I_L, dlam_R_L, dlam_S_R, dlam_I_R, dlam_S_A, dlam_I_A, dlam_W]

def evaluate_local_gradient(u_flat, y0_local, reg_i, climate_fast, params, start_day, duration, num_neighbors, mig_in_func, t_eval):
    u_schedule_local = u_flat.reshape((N_MONTHS, N_CONTROLS))
    T = start_day + duration
    
    sol_fwd = solve_ivp(local_forward_model, (start_day, T), y0_local, t_eval=t_eval, 
                        args=(reg_i, u_schedule_local, climate_fast, params, start_day, num_neighbors, mig_in_func), 
                        method='RK45', rtol=1e-4, atol=1e-4)
    y_fwd = sol_fwd.y

    def y_fwd_func(t):
        idx = int(max(0, min(t - start_day, duration)))
        if idx >= len(t_eval) - 1: return y_fwd[:, -1]
        frac = (t - start_day) - idx
        return y_fwd[:, idx] * (1 - frac) + y_fwd[:, idx + 1] * frac

    lam0 = np.zeros(10)
    sol_adj = solve_ivp(lambda t, lam: local_adjoint_model(t, lam, y_fwd_func, reg_i, u_schedule_local, climate_fast, params, start_day, num_neighbors), 
                        (T, start_day), lam0, t_eval=t_eval[::-1], method='RK45', rtol=1e-4, atol=1e-4)
    lam_hist = sol_adj.y[:, ::-1]

    J = 0.0
    grad = np.zeros(TOTAL_PARAMS)
    dt = np.diff(t_eval)
    
    for k in range(len(t_eval) - 1):
        m = min(int((t_eval[k] - start_day) // DAYS_PER_MONTH), N_MONTHS - 1)
        u1, u2 = u_schedule_local[m, 0], u_schedule_local[m, 1]
        
        J += (A1 * max(0.0, y_fwd[i_I_H, k]) + A2 * max(0.0, y_fwd[i_I_L, k]) + 0.5*B1*u1**2 + 0.5*B2*u2**2) * dt[k]
        
        S_R_k, I_R_k, W_k = max(0.0, y_fwd[i_S_R, k]), max(0.0, y_fwd[i_I_R, k]), max(0.0, y_fwd[i_W, k])
        
        grad[2*m] += (B1 * u1 - I_R_k * lam_hist[i_I_R, k] - S_R_k * lam_hist[i_S_R, k]) * dt[k]
        grad[2*m + 1] += (B2 * u2 - W_k * lam_hist[i_W, k]) * dt[k]

    return J, grad

def get_optimal_controls(sim_data, y0_2d, start_day, duration, M=80, params=None):
    """
    ГОЛОВНА ФУНКЦІЯ: Інтегрована з GUI.
    Використовує метод просторової декомпозиції (BCD) для 27 регіонів.
    """
    if params is None:
        params = DEFAULT_PARAMS

    y0_1d = y0_2d.flatten()
    n_regions = sim_data['n_regions']
    neighbors_indexed = sim_data['neighbors_indexed']
    
    hash_str = hashlib.md5(np.concatenate([y0_1d, [start_day, duration]]).tobytes()).hexdigest()
    cache_file = f"{CACHE_FILENAME}_{hash_str}.pkl"
    
    if os.path.exists(cache_file):
        print(f"\n[ОПТИМІЗАЦІЯ] Знайдено збережене оптимальне керування (BCD)! Миттєве завантаження...")
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
            return cache_data['u1'], cache_data['u2']
            
    print(f"\n[ОПТИМІЗАЦІЯ] Кеш не знайдено. Запуск адаптивного алгоритму BCD + GE-RBF...")
    climate_fast = precompute_climate(sim_data['climate_df'], sim_data['oblast_order'], n_regions)
    
    u_schedule = np.zeros((n_regions, N_MONTHS, N_CONTROLS))
    
    y_global_hist, t_eval = run_global_simulation(y0_1d, climate_fast, params, neighbors_indexed, n_regions, u_schedule, start_day, duration)
    
    eps = 1e-5
    active_regs = []
    for i in range(n_regions):
        max_inf = np.max(y_global_hist[i, i_I_H, :] + y_global_hist[i, i_I_L, :] + 
                         y_global_hist[i, i_I_R, :] + y_global_hist[i, i_I_A, :] + y_global_hist[i, i_W, :])
        if max_inf > eps:
            active_regs.append(i)
            
    if not active_regs:
        print("Інфекція відсутня на всій території. Керування нульове.")
        return np.zeros((n_regions, N_MONTHS)), np.zeros((n_regions, N_MONTHS))
        
    print(f"Виявлено активних регіонів для оптимізації: {len(active_regs)}")

    K_ITERS = 3 
    n_samples = max(20, M * 4) 
    sampler = qmc.LatinHypercube(d=TOTAL_PARAMS, seed=42)
    lb, ub = np.array([b[0] for b in BOUNDS]), np.array([b[1] for b in BOUNDS])
    base_lhs_samples = lb + sampler.random(n=n_samples) * (ub - lb)
    base_lhs_samples = np.vstack([np.zeros(TOTAL_PARAMS), base_lhs_samples]) # Додаємо нульове керування
    
    for iteration in range(1, K_ITERS + 1):
        print(f"\n--- Глобальна ітерація BCD {iteration}/{K_ITERS} ---")
        
        for reg_i in active_regs:
            print(f"  > Оптимізація регіону: {sim_data['oblast_order'][reg_i]} ...")
            y0_local = y0_2d[reg_i]
            num_neighbors = len(neighbors_indexed[reg_i])
            
            mig_in_history = np.zeros((4, len(t_eval)))
            for j in neighbors_indexed[reg_i]:
                mig_in_history[0, :] += params['m_R'] * y_global_hist[j, i_S_R, :]
                mig_in_history[1, :] += params['m_R'] * y_global_hist[j, i_I_R, :]
                mig_in_history[2, :] += params['m_A'] * y_global_hist[j, i_S_A, :]
                mig_in_history[3, :] += params['m_A'] * y_global_hist[j, i_I_A, :]
                
            mig_interp = interp1d(t_eval, mig_in_history, axis=1, bounds_error=False, fill_value="extrapolate")
            def mig_in_func(t): return mig_interp(t)

            current_u_flat = u_schedule[reg_i].flatten()
            X_eval = np.vstack([base_lhs_samples, current_u_flat]) if iteration > 1 else base_lhs_samples
            
            X, Y, dY = [], [], []
            for u_flat in X_eval:
                J, grad = evaluate_local_gradient(u_flat, y0_local, reg_i, climate_fast, params, start_day, duration, num_neighbors, mig_in_func, t_eval)
                X.append(u_flat)
                Y.append(J)
                dY.append(grad)
                
            gerbf = GERBF(bounds=BOUNDS)
            gerbf.train(np.array(X), np.array(Y), np.array(dY))
            
            u_opt_flat = multistart_optimize(gerbf.predict, BOUNDS, jac=gerbf.predict_gradient, n_starts=2, prev_best=current_u_flat)
            
            u_schedule[reg_i] = u_opt_flat.reshape((N_MONTHS, N_CONTROLS))
            
            y_global_hist, _ = run_global_simulation(y0_1d, climate_fast, params, neighbors_indexed, n_regions, u_schedule, start_day, duration)

    print("\n[ОК] BCD оптимізацію завершено.")
    
    u1_opt = u_schedule[:, :, 0]
    u2_opt = u_schedule[:, :, 1]

    with open(cache_file, 'wb') as f:
        pickle.dump({'u1': u1_opt, 'u2': u2_opt}, f)
        
    return u1_opt, u2_opt

def run_simulation_with_controls(start_day, simulation_days, y0_2d, sim_data, params, u1, u2):
    """
    Приймає u1, u2 так, як їх видає головна програма, і пакує в потрібний формат.
    """
    print(f"[SimEngine] Запуск фінальної симуляції (Графіки)...")
    n_regions = sim_data['n_regions']
    climate_fast = precompute_climate(sim_data['climate_df'], sim_data['oblast_order'], n_regions)
    
    u_schedule = np.zeros((n_regions, N_MONTHS, 2))
    if u1.ndim == 1:
        for m in range(N_MONTHS):
            u_schedule[:, m, 0] = u1
            u_schedule[:, m, 1] = u2
    else:
        u_schedule[:, :, 0] = u1
        u_schedule[:, :, 1] = u2
    
    T = start_day + simulation_days
    t_eval_points = np.arange(start_day, T + 1, 1)
    args_fwd = (climate_fast, params, sim_data['neighbors_indexed'], n_regions, u_schedule, start_day)
    
    sol = solve_ivp(fun=global_forward_model, t_span=(start_day, T), y0=y0_2d.flatten(), 
                    t_eval=t_eval_points, args=args_fwd, method='LSODA')
    
    if not sol.success:
        print(f"[ПОМИЛКА] Симуляція не вдалася: {sol.message}")
        return None, None
        
    return sol.y.reshape((n_regions, N_VARS, len(t_eval_points))), t_eval_points