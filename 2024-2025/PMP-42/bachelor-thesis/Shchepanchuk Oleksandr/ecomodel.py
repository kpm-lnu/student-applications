import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from kriging_module import KrigingSurrogate


def eco_model(y, t, r, K, a, b, d):
    """Рівняння Rosenzweig–MacArthur (X – жертви, Y – хижаки)."""
    X, Y = y
    dXdt = r * X * (1 - X / K) - a * X * Y
    dYdt = b * a * X * Y - d * Y
    return [dXdt, dYdt]


def simulate_Y(params, t, y0=[0.8, 0.2]):
    sol = odeint(eco_model, y0, t, args=tuple(params))
    return sol[:, 1]  

def run_experiment(sample_method: str, n_train: int = 60, random_seed: int = 42):
    label = "Рівномірна" if sample_method == "uniform" else "LHS"
    print(f"\n=== {label} вибірка ===")

    domain = [
        [1.0, 5],   # r  – intrinsic ріст жертви
        [1.0, 2.0],   # K  – carrying capacity
        [0.02, 0.2],  # a  – швидкість атаки
        [0.1, 0.6],   # b  – коеф. конверсії
        [0.5, 1.5]    # d  – смертність хижаків
    ]

    
    n_samples = int(n_train / 0.7)
    sampling_calls, direct_calls = {"calls": 0}, {"calls": 0}

    
    
    
    def objective_ecosystem(params, 
                     t: np.ndarray = np.linspace(0, 200, 1000)):
        sol = odeint(eco_model, [0.8, 0.2], t, args=tuple(params))
        X, Y = sol.T
        P = (sum(Y) - sum(X)) 
        return -P      
        
    

    def obj_sampling(x):
        sampling_calls["calls"] += 1
        return objective_ecosystem(x)

    def obj_direct(x):
        direct_calls["calls"] += 1
        return objective_ecosystem(x)

    surrogate = KrigingSurrogate(
        objective_func=obj_sampling,
        domain=domain,
        sample_method=sample_method,
        n_samples=n_samples,
        random_seed=random_seed,
    )
    surrogate.sample_data()
    surrogate.fit()

    res_sur = surrogate.optimize_surrogate(method="Powell")
    opt_params_sur = res_sur.x * (surrogate.X_max - surrogate.X_min) + surrogate.X_min
    f_sur = obj_direct(opt_params_sur)

    direct_calls["calls"] = 0
    init_guess = np.array([(b[0] + b[1]) / 2 for b in domain])
    res_dir = minimize(
        obj_direct,
        init_guess,
        bounds=[tuple(b) for b in domain],
        method="Powell",
    )
    f_dir = res_dir.fun

    rmse, mae, rrmse_pct, r2 = surrogate.evaluate_performance()

    print("----- Сурогатна модель -----")
    print(f"RMSE (абс)      : {rmse:.6f}")
    print(f"MAE  (абс)      : {mae:.6f}")
    print(f"RMSE (відносно) : {rrmse_pct:.2f}%")
    print(f"R²              : {r2:.6f}\n")
    print("----- Порівняння оптимізації -----")
    print(f"Викликів (семпл / пряма): {n_train} / {direct_calls['calls']}")
    print(f"Сурогат x*  : {np.round(opt_params_sur, 4)} → J = {f_sur:.4f}")
    print(f"Прямий x*   : {np.round(res_dir.x, 4)} → J = {f_dir:.4f}")

    Y_true = np.array(
        [
            obj_direct(pt * (surrogate.X_max - surrogate.X_min) + surrogate.X_min)
            for pt in surrogate.X_test_norm
        ]
    )
    Y_pred = np.array([surrogate.surrogate_predict(pt) for pt in surrogate.X_test_norm])

    plt.figure(figsize=(6, 6))
    plt.scatter(Y_true, Y_pred, c="green", edgecolor="k", s=40, label="Тестові зразки")
    mn, mx = Y_true.min(), Y_true.max()
    plt.plot([mn, mx], [mn, mx], "k--", label="Ідеал")
    plt.xlabel("Справжнє J")
    plt.ylabel("Прогноз J")
    plt.title(f"Парний графік — {label} вибірка")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    for method in ("uniform", "lhs"):
        run_experiment(method)
