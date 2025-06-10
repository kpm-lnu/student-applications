import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from kriging_module import KrigingSurrogate

def corrected_model(y, t, beta, mu, alpha, gamma):
    S, E, I, R = y
    return [
        -beta * S * I,
        beta * S * I - mu * E,
        alpha * mu * E - gamma * I,
        gamma * I + (1 - alpha) * mu * E
    ]

def simulate_I(params, t, y0=[0.7, 0.25, 0.05, 0.0]):
    sol = odeint(corrected_model, y0, t, args=tuple(params))
    return sol[:, 2]

def run_experiment(sample_method: str, n_train: int = 50, random_seed: int = 42):
    label = "Рівномірна" if sample_method == "uniform" else "LHS"
    print(f"\n=== {label} вибірка ===")

    domain = [[0.1, 0.3], [0.005, 0.01], [0.2, 0.5], [0.005, 0.01]]
    n_samples = int(n_train / 0.7)
    sampling_calls = {"calls": 0}
    direct_calls = {"calls": 0}

    def objective_rumor(params):
        sampling_calls["calls"] += (1 if plt.fignum_exists(0) else 0)
        beta, mu, alpha, gamma = params
        y0 = [0.7, 0.25, 0.05, 0.0]
        t = np.linspace(0, 500, 500)
        sol = odeint(corrected_model, y0, t, args=(beta, mu, alpha, gamma))
        return np.sum(sol[:, 2])

    def obj_sampling(x):
        sampling_calls["calls"] += 1
        return objective_rumor(x)

    def obj_direct(x):
        direct_calls["calls"] += 1
        return objective_rumor(x)

    surrogate = KrigingSurrogate(
        objective_func=obj_sampling,
        domain=domain,
        sample_method=sample_method,
        n_samples=n_samples,
        random_seed=random_seed
    )
    surrogate.sample_data()
    surrogate.fit()

    res_sur = surrogate.optimize_surrogate(method='Powell')
    opt_params_sur = res_sur.x * (surrogate.X_max - surrogate.X_min) + surrogate.X_min
    f_sur = obj_direct(opt_params_sur)

    direct_calls["calls"] = 0
    init_guess = np.array([(b[0]+b[1]) / 2 for b in domain])
    res_dir = minimize(obj_direct, init_guess, bounds=[tuple(b) for b in domain], method='Powell')
    f_dir = res_dir.fun

    rmse, mae, rrmse_pct, r2 = surrogate.evaluate_performance()

    print("----- Сурогатна модель -----")
    print(f"RMSE (абс)      : {rmse:.6f}")
    print(f"MAE  (абс)      : {mae:.6f}")
    print(f"RMSE (відносно) : {rrmse_pct:.2f}%")
    print(f"R²              : {r2:.6f}\n")
    print("----- Порівняння оптимізації -----")
    print(f"Викликів (семпл / пряма): {n_train} / {direct_calls['calls']}")
    print(f"Сурогат x*  : {np.round(opt_params_sur,4)} → J = {f_sur:.4f}")
    print(f"Прямий x*   : {np.round(res_dir.x,4)} → J = {f_dir:.4f}")

    Y_true = np.array([
        obj_direct(pt * (surrogate.X_max - surrogate.X_min) + surrogate.X_min)
        for pt in surrogate.X_test_norm
    ])
    Y_pred = np.array([surrogate.surrogate_predict(pt) for pt in surrogate.X_test_norm])

    plt.figure(figsize=(6,6))
    plt.scatter(Y_true, Y_pred, c='green', edgecolor='k', s=40, label='Тестові зразки')
    mn, mx = Y_true.min(), Y_true.max()
    plt.plot([mn, mx], [mn, mx], 'k--', label='Ідеал')
    plt.xlabel("Справжнє J")
    plt.ylabel("Прогноз J")
    plt.title(f"Парний графік — {label} вибірка")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

    t = np.linspace(0, 500, 500)
    I_sur = simulate_I(opt_params_sur, t)
    I_dir = simulate_I(res_dir.x, t)

    plt.figure(figsize=(8,4))
    plt.plot(t, I_sur, lw=2, label='Сурогат-оптимум')
    plt.plot(t, I_dir, '--', lw=2, label='Прямий-оптимум')
    plt.xlabel("Час, t")
    plt.ylabel("I(t) — активні поширювачі")
    plt.title(f"Криві I(t) — {label} вибірка")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    for method in ("uniform", "lhs"):
        run_experiment(method)
