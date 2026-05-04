"""
unified_analysis.py
===================
Єдина точка входу для аналізу 12-вимірної задачі оптимального керування 
лікуванням раку. Імпортує всю логіку ЗДР (ODE) та спряженої системи з 
ge_rbf_cancer_surrogate.py; математика тут не дублюється.

Пайплайн
--------
1.  FBSM  – прямий пошук еталонного оптимуму (J_direct, w_fbsm)
2.  Дані  – локальна навчальна вибірка LHS (N=50) + рівномірна тестова (N=20),
            в межах радіуса 0.15 навколо w_fbsm
3.  Побудова – Стандартна RBF (лише за значеннями, LOO-CV gamma)
             GE-RBF (Ерміт: значення+градієнт, той самий LOO-CV gamma)
4.  Пошук – багаторазовий запуск L-BFGS-B у довірчій області (радіус 0.15)
5.  Таблиця – виведення ASCII-таблиці порівняння у консоль
6.  Графіки – збереження 5 PNG-фігур для дисертації (жирні лінії/шрифти)
"""

# ── стандартні бібліотеки ────────────────────────────────────────────────────
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")          # Вимикаємо GUI
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── ядро задачі (ЗДР, градієнти, параметри) ──────────────────────────────────
from scipy.interpolate import PchipInterpolator
from ge_rbf_cancer_surrogate import (
    D, N_SEG, _seg_edges,
    t_grid,
    B_obj, C_obj, g0, params,
    x0_state, _f,
    forward_sweep,
    backward_sweep,
    _compute_J,
    OracleCounter,
)

# ══════════════════════════════════════════════════════════════════════════════
# ІНФРАСТРУКТУРА ПЛАВНОГО КЕРУВАННЯ (тільки для графіків)
# ══════════════════════════════════════════════════════════════════════════════

_N_PLOT  = 1500
_T_END   = float(t_grid[-1])                        # 26.0 місяців
_t_plot  = np.linspace(0.0, _T_END, _N_PLOT + 1)
_dt_plot = _t_plot[1] - _t_plot[0]
_t_anchors = np.linspace(0.0, _T_END, N_SEG)        # Опорні точки керування


def w_to_controls_smooth(w):
    """
    Конвертує вектор w у неперервні профілі u1(t), u2(t) за допомогою PCHIP.
    """
    u1_anchors = np.clip(w[:N_SEG],  0.0, 1.0)
    u2_anchors = np.clip(w[N_SEG:],  0.0, 1.0)
    u1 = np.clip(PchipInterpolator(_t_anchors, u1_anchors)(_t_plot), 0.0, 1.0)
    u2 = np.clip(PchipInterpolator(_t_anchors, u2_anchors)(_t_plot), 0.0, 1.0)
    return u1, u2


def forward_sweep_hires(u1, u2):
    """
    Інтегрування на високороздільній сітці (1500 кроків) для гладких графіків.
    """
    X = np.zeros((4, _N_PLOT + 1))
    X[:, 0] = x0_state
    for i in range(_N_PLOT):
        xi  = X[:, i]
        a,  b  = u1[i], u1[i + 1]
        am     = 0.5 * (a  + b)
        c,  d  = u2[i], u2[i + 1]
        cm     = 0.5 * (c  + d)
        k1 = _f(xi,                    a,  c)
        k2 = _f(xi + 0.5*_dt_plot*k1,  am, cm)
        k3 = _f(xi + 0.5*_dt_plot*k2,  am, cm)
        k4 = _f(xi +     _dt_plot*k3,  b,  d)
        X[:, i + 1] = np.maximum(
            xi + (_dt_plot / 6.0) * (k1 + 2*k2 + 2*k3 + k4), 0.0)
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Глобальний стиль (тільки шрифти, товщина/жирність задаються жорстко далі)
# ─────────────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
    "figure.dpi":         130,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

CLR = {
    "fbsm":    "#2c7bb6",
    "gerbf":   "#d7191c",
    "rbf":     "#1a9641",
    "notreat": "black",
    "fill":    "#fdae61",
    "u1":      "blue",
    "u2":      "red",
}

# ══════════════════════════════════════════════════════════════════════════════
# РОЗДІЛ 1 – ПРЯМА ОПТИМІЗАЦІЯ (FBSM)
# ══════════════════════════════════════════════════════════════════════════════

def run_fbsm(max_iter: int = 300, tol: float = 1e-5):
    n_steps = len(t_grid) - 1
    u1 = np.full(n_steps + 1, 0.5)
    u2 = np.full(n_steps + 1, 0.5)
    ode_calls = 0
    n_iters   = max_iter

    for it in range(max_iter):
        X = forward_sweep(u1, u2);       ode_calls += 1
        L = backward_sweep(X, u1, u2);   ode_calls += 1

        p = params
        N_s, T_s, M_s, E_s = X
        l1, l2, l3, l4 = L

        u1_new = np.clip(
            (l2 * g0 * T_s - l1 * p["eta_g"] * g0 * N_s
             - l3 * g0 * M_s) / B_obj, 0.0, 1.0)
        u2_new = np.clip(
            (-l1 * p["e1"] * N_s * E_s + l2 * p["e2"] * N_s * E_s
             - l3 * p["e3"] * M_s * E_s + l4 * p["Pi"]) / C_obj,
            0.0, 1.0)

        u1_nx = 0.5 * u1 + 0.5 * u1_new
        u2_nx = 0.5 * u2 + 0.5 * u2_new
        err = max(np.linalg.norm(u1_nx - u1, np.inf),
                  np.linalg.norm(u2_nx - u2, np.inf))
        u1, u2 = u1_nx, u2_nx

        if err < tol:
            n_iters = it + 1
            break

    X_opt = forward_sweep(u1, u2);  ode_calls += 1
    J_direct = _compute_J(X_opt, u1, u2)

    w_fbsm = np.zeros(D)
    for k in range(N_SEG):
        sl = slice(_seg_edges[k], _seg_edges[k + 1])
        w_fbsm[k]         = float(np.mean(u1[sl]))
        w_fbsm[k + N_SEG] = float(np.mean(u2[sl]))
    w_fbsm = np.clip(w_fbsm, 0.0, 1.0)

    return J_direct, w_fbsm, ode_calls, n_iters, u1, u2, X_opt


# ══════════════════════════════════════════════════════════════════════════════
# РОЗДІЛ 2 – ГЕНЕРАЦІЯ ЛОКАЛЬНИХ ВИБІРОК
# ══════════════════════════════════════════════════════════════════════════════

def generate_datasets(w_seed, N_train=50, N_test=20,
                      local_radius=0.15, seed=42):
    oracle = OracleCounter()
    rng    = np.random.default_rng(seed)

    perm    = np.stack([rng.permutation(N_train) for _ in range(D)], axis=1)
    offsets = ((perm + rng.random((N_train, D))) / N_train - 0.5) * 2.0 * local_radius
    W_train = np.clip(w_seed[None, :] + offsets, 0.01, 0.99)
    W_train[0] = np.clip(w_seed, 0.01, 0.99)

    J_train = np.zeros(N_train)
    G_train = np.zeros((N_train, D))
    for i in range(N_train):
        J_train[i], G_train[i] = oracle.J_and_grad(W_train[i])

    offsets_t = rng.uniform(-local_radius, local_radius, (N_test, D))
    W_test    = np.clip(w_seed[None, :] + offsets_t, 0.01, 0.99)
    J_test    = np.zeros(N_test)
    for i in range(N_test):
        J_test[i], _ = oracle.J_and_grad(W_test[i])

    return W_train, J_train, G_train, W_test, J_test, oracle


# ══════════════════════════════════════════════════════════════════════════════
# РОЗДІЛ 3 – ВИБІР ПАРАМЕТРА GAMMA (LOO-CV)
# ══════════════════════════════════════════════════════════════════════════════

def loocv_gamma(W_train, J_norm, n_candidates=24):
    N     = len(J_norm)
    delta = W_train[:, None, :] - W_train[None, :, :]
    r2    = (delta ** 2).sum(axis=2)

    g_med      = 1.0 / (2.0 * float(np.median(r2[r2 > 0])))
    candidates = np.logspace(
        np.log10(g_med) - 2.0, np.log10(g_med) + 2.0, n_candidates)

    best_g, best_mse = g_med, np.inf
    for g in candidates:
        K      = np.exp(-g * r2)
        nugget = max(1e-8 * np.trace(K) / N, 1e-10)
        K_reg  = K + nugget * np.eye(N)
        try:
            K_inv = np.linalg.inv(K_reg)
        except np.linalg.LinAlgError:
            continue
        alpha = K_inv @ J_norm
        diag  = np.diag(K_inv)
        if np.any(np.abs(diag) < 1e-14):
            continue
        mse = float(np.mean((alpha / diag) ** 2))
        if mse < best_mse:
            best_mse, best_g = mse, g

    return best_g


# ══════════════════════════════════════════════════════════════════════════════
# РОЗДІЛ 4 – ПОБУДОВА СУРОГАТНИХ МОДЕЛЕЙ
# ══════════════════════════════════════════════════════════════════════════════

def build_standard_rbf(W_train, J_train):
    N      = len(J_train)
    J_mean = J_train.mean()
    J_std  = J_train.std() + 1e-14
    J_norm = (J_train - J_mean) / J_std

    gamma  = loocv_gamma(W_train, J_norm)

    delta  = W_train[:, None, :] - W_train[None, :, :]
    r2     = (delta ** 2).sum(axis=2)
    K      = np.exp(-gamma * r2)
    nugget = max(1e-8 * np.trace(K) / N, 1e-10)
    K_reg  = K + nugget * np.eye(N)

    alpha, *_ = np.linalg.lstsq(K_reg, J_norm, rcond=None)

    def predict(w):
        diff  = w[None, :] - W_train
        phi_s = np.exp(-gamma * (diff ** 2).sum(axis=1))
        return float(J_mean + J_std * (alpha @ phi_s))

    return predict, gamma


def build_gerbf(W_train, J_train, G_train, gamma):
    N      = len(J_train)
    J_mean = J_train.mean()
    J_std  = J_train.std() + 1e-14
    J_norm = (J_train - J_mean) / J_std
    G_norm = G_train / J_std

    delta = W_train[:, None, :] - W_train[None, :, :]
    r2    = (delta ** 2).sum(axis=2)
    phi   = np.exp(-gamma * r2)

    sz = N * (D + 1)
    K  = np.zeros((sz, sz))

    K[:N, :N] = phi
    K[:N, N:] = ( 2.0 * gamma * delta * phi[:, :, None]).reshape(N, N * D)
    K[N:, :N] = (-2.0 * gamma * delta * phi[:, :, None]).transpose(0, 2, 1).reshape(N * D, N)

    outer    = np.einsum("ijk,ijl->ijkl", delta, delta)
    K_gg_4d  = phi[:, :, None, None] * (
        2.0 * gamma * np.eye(D)[None, None] - 4.0 * gamma ** 2 * outer)
    K[N:, N:] = K_gg_4d.transpose(0, 2, 1, 3).reshape(N * D, N * D)

    nugget = max(1e-8 * np.trace(K) / sz, 1e-10)
    K_reg  = K + nugget * np.eye(sz)

    b_rhs = np.empty(sz)
    b_rhs[:N] = J_norm
    for i in range(N):
        b_rhs[N + i * D: N + (i + 1) * D] = G_norm[i]

    sol, *_ = np.linalg.lstsq(K_reg, b_rhs, rcond=None)
    alpha_v = sol[:N]
    beta_m  = sol[N:].reshape(N, D)

    def predict(w):
        diff    = w[None, :] - W_train
        r2_s    = (diff ** 2).sum(axis=1)
        phi_s   = np.exp(-gamma * r2_s)
        b_dot_d = (beta_m * diff).sum(axis=1)
        s_val   = float((alpha_v * phi_s).sum() + 2.0 * gamma * (phi_s * b_dot_d).sum())
        return J_mean + J_std * s_val

    return predict


# ══════════════════════════════════════════════════════════════════════════════
# РОЗДІЛ 5 – МІНІМІЗАЦІЯ СУРОГАТУ
# ══════════════════════════════════════════════════════════════════════════════

def minimise_surrogate(surrogate_fn, w_seed, n_starts=15,
                       seed=99, trust_radius=0.15):
    bounds = [
        (max(0.0, float(w_seed[d]) - trust_radius),
         min(1.0, float(w_seed[d]) + trust_radius))
        for d in range(D)
    ]
    rng    = np.random.default_rng(seed)
    lo_arr = np.array([b[0] for b in bounds])
    hi_arr = np.array([b[1] for b in bounds])

    starts = [np.clip(w_seed.copy(), lo_arr, hi_arr)]
    starts += [
        np.array([rng.uniform(lo, hi) for lo, hi in bounds])
        for _ in range(n_starts)
    ]

    best_J, best_w = np.inf, w_seed.copy()
    for ws in starts:
        res = minimize(
            surrogate_fn, ws, method="L-BFGS-B", jac=None,
            bounds=bounds,
            options={"maxiter": 600, "ftol": 1e-15, "gtol": 1e-9},
        )
        if res.fun < best_J:
            best_J = float(res.fun)
            best_w = res.x.copy()

    return best_w, best_J


# ══════════════════════════════════════════════════════════════════════════════
# РОЗДІЛ 6 – МЕТРИКИ
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(predict_fn, W_test, J_test):
    J_pred = np.array([predict_fn(W_test[i]) for i in range(len(J_test))])
    ss_res = float(np.sum((J_test - J_pred) ** 2))
    ss_tot = float(np.sum((J_test - J_test.mean()) ** 2))
    r2     = 1.0 - ss_res / (ss_tot + 1e-14)
    rmse   = float(np.sqrt(ss_res / len(J_test)))
    return r2, rmse, J_pred


# ══════════════════════════════════════════════════════════════════════════════
# РОЗДІЛ 7 – ДРУК ТАБЛИЦІ
# ══════════════════════════════════════════════════════════════════════════════

def print_table(J_direct, fbsm_ode,
                r2_rbf,   rmse_rbf,   J_rbf_true,   ode_rbf,
                r2_ge,    rmse_ge,    J_ge_true,     ode_ge,
                N_train, N_test, shared_calls, trust_r, sample_r):

    rel_rbf = abs(J_rbf_true - J_direct) / (abs(J_direct) + 1e-14) * 100
    rel_ge  = abs(J_ge_true  - J_direct) / (abs(J_direct) + 1e-14) * 100

    col_w = [22, 22, 10, 12, 14, 18, 20]
    sep   = "+" + "+".join("-" * w for w in col_w) + "+"

    def row(*cells):
        return "|" + "|".join(
            (" " + str(c)).ljust(w) for c, w in zip(cells, col_w)
        ) + "|"

    total_w = sum(col_w) + len(col_w) + 1
    banner  = "  ПОРІВНЯННЯ МЕТОДІВ: Повна модель vs Стандартна RBF vs GE-RBF"

    print()
    print("=" * total_w)
    print(banner)
    print("=" * total_w)
    print(sep)
    print(row("Метод", "Градієнт Понтрягіна?",
              "R²", "RMSE", "Виклики ODE", "Знайдений J", "Відн. похибка (%)"))
    print(sep)
    print(row("Прямий метод (FBSM)", "Так (FBSM)",
              "N/A", "N/A",
              str(fbsm_ode),
              f"{J_direct:.6f}",
              "0.000000 (еталон)"))
    print(sep)
    print(row("Стандартна RBF", "Ні",
              f"{r2_rbf:.6f}", f"{rmse_rbf:.6f}",
              str(ode_rbf),
              f"{J_rbf_true:.6f}",
              f"{rel_rbf:.6f}"))
    print(sep)
    print(row("GE-RBF", "Так",
              f"{r2_ge:.6f}", f"{rmse_ge:.6f}",
              str(ode_ge),
              f"{J_ge_true:.6f}",
              f"{rel_ge:.6f}"))
    print(sep)
    print()
    print("  Примітки:")
    print(f"    J_ref (еталон) = {J_direct:.8f}")
    print(f"    Навчальна вибірка: {N_train} точок (радіус={sample_r})")
    print(f"    Тестова вибірка: {N_test} точок")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# РОЗДІЛ 8 – ГЕНЕРАЦІЯ ЖИРНИХ ГРАФІКІВ (ЯК В ЕТАЛОНІ)
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig, fname):
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print(f"  Збережено → {fname}")


def plot_treatment_strategy(t_plot, u1_ge, u2_ge, w_ge, fname="fig1_treatment_strategy.png"):
    u1_anchors = np.clip(w_ge[:N_SEG],  0.0, 1.0)
    u2_anchors = np.clip(w_ge[N_SEG:],  0.0, 1.0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Верхній графік - Хіміотерапія
    axes[0].plot(t_plot, u1_ge, label='PCHIP-інтерпольований профіль', 
                 color=CLR["u1"], linestyle='-', linewidth=3)
    axes[0].scatter(_t_anchors, u1_anchors, label='Опорні точки оптимізації', 
                    color=CLR["u1"], s=80, edgecolors='white', linewidths=1.5, zorder=5)
    axes[0].fill_between(t_plot, 0, u1_ge, alpha=0.12, color=CLR["u1"], zorder=2)
    
    axes[0].set_ylabel('$u_1(t)$', fontsize=12, fontweight='bold')
    axes[0].set_title('Профіль метформіну', fontsize=12, fontweight='bold', pad=15)
    axes[0].legend(loc='best', fontsize=10, framealpha=1, edgecolor='gray')
    axes[0].grid(True, linestyle='-', alpha=0.5)

    # Нижній графік - Гормональна терапія
    axes[1].plot(t_plot, u2_ge, label='PCHIP-інтерпольований профіль', 
                 color=CLR["u2"], linestyle='-', linewidth=3)
    axes[1].scatter(_t_anchors, u2_anchors, label='Опорні точки оптимізації', 
                    color=CLR["u2"], s=80, edgecolors='white', linewidths=1.5, zorder=5)
    axes[1].fill_between(t_plot, 0, u2_ge, alpha=0.12, color=CLR["u2"], zorder=2)
    
    axes[1].set_ylabel('$u_2(t)$', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Час (місяці)', fontsize=12, fontweight='bold')
    axes[1].set_title('Профіль тамоксифену', fontsize=12, fontweight='bold', pad=15)
    axes[1].legend(loc='best', fontsize=10, framealpha=1, edgecolor='gray')
    axes[1].grid(True, linestyle='-', alpha=0.5)

    plt.tight_layout()
    _save(fig, fname)


def plot_surrogate_quality(J_test, J_pred_rbf, J_pred_ge, r2_rbf, rmse_rbf, r2_ge, rmse_ge, fname="fig2_surrogate_quality.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    lo = min(J_test.min(), min(J_pred_rbf.min(), J_pred_ge.min())) * 0.97
    hi = max(J_test.max(), max(J_pred_rbf.max(), J_pred_ge.max())) * 1.03

    # RBF
    axes[0].plot([lo, hi], [lo, hi], label='Ідеальне передбачення ($y = x$)', color='black', linestyle='--', linewidth=3)
    axes[0].scatter(J_test, J_pred_rbf, label='Тестові точки', color=CLR["rbf"], s=80, edgecolors='white', linewidths=1.5)
    axes[0].set_xlabel('$J(w)$', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Передбачення $\hat{J}(w)$', fontsize=12, fontweight='bold')
    axes[0].set_title('Стандартна RBF (без градієнтів)', fontsize=12, fontweight='bold', pad=15)
    axes[0].legend(loc='best', fontsize=10, framealpha=1, edgecolor='gray')
    axes[0].grid(True, linestyle='-', alpha=0.5)
    axes[0].text(0.05, 0.93, f"$R^2$ = {r2_rbf:.4f}\nRMSE = {rmse_rbf:.4f}", transform=axes[0].transAxes, 
                 fontsize=11, fontweight='bold', va="top", bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray"))

    # GE-RBF
    axes[1].plot([lo, hi], [lo, hi], label='Ідеальне передбачення ($y = x$)', color='black', linestyle='--', linewidth=3)
    axes[1].scatter(J_test, J_pred_ge, label='Тестові точки', color=CLR["gerbf"], s=80, edgecolors='white', linewidths=1.5)
    axes[1].set_xlabel('$J(w)$', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Передбачення $\hat{J}(w)$', fontsize=12, fontweight='bold')
    axes[1].set_title('GE-RBF (з градієнтами)', fontsize=12, fontweight='bold', pad=15)
    axes[1].legend(loc='best', fontsize=10, framealpha=1, edgecolor='gray')
    axes[1].grid(True, linestyle='-', alpha=0.5)
    axes[1].text(0.05, 0.93, f"$R^2$ = {r2_ge:.4f}\nRMSE = {rmse_ge:.4f}", transform=axes[1].transAxes, 
                 fontsize=11, fontweight='bold', va="top", bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray"))

    plt.tight_layout()
    _save(fig, fname)


def _cell_comparison_plot(t_grid, traj_opt, traj_no, cell_idx, cell_label, cell_sym, fname, col_opt, col_no):
    y_opt = traj_opt[cell_idx]
    y_no  = traj_no[cell_idx]
    
    fig, ax = plt.subplots(figsize=(9, 6))

    marker_spacing = 100 

    ax.plot(t_grid, y_opt, label=f'GE-RBF оптимальне', 
            color=col_opt, linestyle='-', marker='o', linewidth=3, markersize=8, markevery=marker_spacing)
    
    ax.plot(t_grid, y_no, label=f'Без лікування', 
            color=col_no, linestyle='--', marker='^', linewidth=3, markersize=8, markevery=marker_spacing)

    ax.fill_between(t_grid, y_opt, y_no, alpha=0.18, color=CLR["fill"], label="Область різниці")

    ax.set_xlabel('Час (місяці)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Концентрація {cell_sym}$(t)$ (норм.)', fontsize=12, fontweight='bold')
    ax.set_title(f'{cell_label}: Оптимальне лікування vs Без лікування', fontsize=12, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=10, framealpha=1, edgecolor='gray')
    ax.grid(True, linestyle='-', alpha=0.5)

    plt.tight_layout()
    _save(fig, fname)


def plot_tumor(t_grid, X_ge, X_no, fname="fig3_comparison_tumor.png"):
    _cell_comparison_plot(t_grid, X_ge, X_no, 1, "Пухлинні клітини $T(t)$", "T", fname, CLR["gerbf"], CLR["notreat"])

def plot_normal(t_grid, X_ge, X_no, fname="fig4_comparison_normal.png"):
    _cell_comparison_plot(t_grid, X_ge, X_no, 0, "Нормальні клітини $N(t)$", "N", fname, '#1f77b4', CLR["notreat"])

def plot_immune(t_grid, X_ge, X_no, fname="fig5_comparison_immune.png"):
    _cell_comparison_plot(t_grid, X_ge, X_no, 2, "Імунні клітини $M(t)$", "M", fname, CLR["rbf"], CLR["notreat"])


# ══════════════════════════════════════════════════════════════════════════════
# ГОЛОВНИЙ ПАЙПЛАЙН (MAIN PIPELINE)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    SAMPLE_RADIUS = 0.15
    TRUST_RADIUS  = 0.15
    N_TRAIN       = 50
    N_TEST        = 20
    N_STARTS      = 15

    print("=" * 62)
    print("КРОК 1 — Пряма оптимізація FBSM")
    print("=" * 62)
    J_direct, w_fbsm, fbsm_ode, fbsm_iters, u1_fbsm, u2_fbsm, X_fbsm = run_fbsm()
    print(f"  Зійшлося за {fbsm_iters} ітерацій")
    print(f"  J_direct         = {J_direct:.8f}")
    
    print("\nКРОК 2 — Генерація локальних вибірок")
    W_train, J_train, G_train, W_test, J_test, shared_oracle = generate_datasets(w_fbsm, N_train=N_TRAIN, N_test=N_TEST, local_radius=SAMPLE_RADIUS)
    shared_calls = shared_oracle.n_calls

    print("\nКРОК 3 — Побудова сурогатних моделей")
    predict_rbf, gamma_cv = build_standard_rbf(W_train, J_train)
    predict_ge = build_gerbf(W_train, J_train, G_train, gamma_cv)

    print("\nКРОК 4 — Оцінка точності на тестовій вибірці")
    r2_rbf, rmse_rbf, J_pred_rbf = compute_metrics(predict_rbf, W_test, J_test)
    r2_ge,  rmse_ge,  J_pred_ge  = compute_metrics(predict_ge,  W_test, J_test)

    print(f"\nКРОК 5 — Мінімізація сурогатів (trust-region: {TRUST_RADIUS})")
    w_rbf_opt, _ = minimise_surrogate(predict_rbf, w_fbsm, n_starts=N_STARTS, seed=101, trust_radius=TRUST_RADIUS)
    oc_rbf     = OracleCounter()
    J_rbf_true, _ = oc_rbf.J_and_grad(w_rbf_opt)

    w_ge_opt, _ = minimise_surrogate(predict_ge, w_fbsm, n_starts=N_STARTS, seed=102, trust_radius=TRUST_RADIUS)
    oc_ge      = OracleCounter()
    J_ge_true, _ = oc_ge.J_and_grad(w_ge_opt)
    surrogate_ode_calls = shared_calls + 1

    print_table(J_direct, fbsm_ode, r2_rbf, rmse_rbf, J_rbf_true, surrogate_ode_calls, r2_ge,  rmse_ge,  J_ge_true,  surrogate_ode_calls, N_TRAIN, N_TEST, shared_calls, TRUST_RADIUS, SAMPLE_RADIUS)

    print("\nКРОК 6 — Реконструкція гладких траєкторій для графіків")
    u1_ge_smooth, u2_ge_smooth = w_to_controls_smooth(w_ge_opt)
    X_ge = forward_sweep_hires(u1_ge_smooth, u2_ge_smooth)
    u1_no_smooth, u2_no_smooth = w_to_controls_smooth(np.zeros(D))
    X_no = forward_sweep_hires(u1_no_smooth, u2_no_smooth)

    print("\nКРОК 7 — Генерація та збереження графіків (РУЧНИЙ ЖИРНИЙ СТИЛЬ)")
    plot_treatment_strategy(_t_plot, u1_ge_smooth, u2_ge_smooth, w_ge_opt)
    plot_surrogate_quality(J_test, J_pred_rbf, J_pred_ge, r2_rbf, rmse_rbf, r2_ge, rmse_ge)
    plot_tumor(_t_plot, X_ge, X_no)
    plot_normal(_t_plot, X_ge, X_no)
    plot_immune(_t_plot, X_ge, X_no)

    print("\nЗАВЕРШЕНО")