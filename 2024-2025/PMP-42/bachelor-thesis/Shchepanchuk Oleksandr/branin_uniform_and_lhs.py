import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from kriging_module import KrigingSurrogate


def branin(x):
    b = 5.1 / (4 * np.pi ** 2)
    c = 5 / np.pi
    r, s, t = 6.0, 10.0, 1.0 / (8 * np.pi)
    x1, x2 = x
    return (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s



def run_experiment(sample_method: str, n_train: int = 25, random_seed: int = 42):
    
    uk_map = {"uniform": "рівномірна", "lhs": "LHS"}
    print(f"\n=== {uk_map[sample_method]} вибірка ===")

    domain = [[-5, 10], [0, 15]]
    n_samples = int(n_train / 0.7)      

    direct_calls = {"calls": 0}        

    def obj_sampling(x):
        return branin(x)

    def obj_direct(x):
        direct_calls["calls"] += 1
        return branin(x)

    surrogate = KrigingSurrogate(
        objective_func=obj_sampling,
        domain=domain,
        sample_method=sample_method,
        n_samples=n_samples,
        random_seed=random_seed,
    )
    surrogate.sample_data()
    surrogate.fit()
    try:
        # if KrigingSurrogate stores them already un‐normalized
        X_train = surrogate.X  
    except AttributeError:
        # otherwise un‐normalize the stored normed points
        X_train = surrogate.X_train_norm * (surrogate.X_max - surrogate.X_min) + surrogate.X_min

    # now plot them
    plt.figure(figsize=(6,6))
    plt.scatter(
        X_train[:,0],
        X_train[:,1],
        marker='s',
        facecolors='none',
        edgecolors='red',
        s=80,
        linewidths=1.5
    )
    # set grid and ticks every 1 unit
    plt.xlim(domain[0])
    plt.ylim(domain[1])
    plt.xticks(np.arange(domain[0][0], domain[0][1]+1, 1))
    plt.yticks(np.arange(domain[1][0], domain[1][1]+1, 1))
    plt.grid(True, which='both', linestyle=':', alpha=0.7)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(f'План вибірки — {uk_map[sample_method].capitalize()} вибірка')
    plt.tight_layout()
    plt.show()

    res_sur = surrogate.optimize_surrogate(method="Powell")
    x_sur = res_sur.x * (surrogate.X_max - surrogate.X_min) + surrogate.X_min
    f_sur = obj_direct(x_sur)

    init_guess = np.mean(domain, axis=1)
    res_dir = minimize(
        obj_direct, init_guess, bounds=[tuple(b) for b in domain], method="Powell"
    )

    rmse, mae, rrmse_pct, r2 = surrogate.evaluate_performance()
    print(
        f"RMSE={rmse:.6f}  MAE={mae:.6f}  RMSE%={rrmse_pct:.2f}%  R²={r2:.6f}"
    )
    print(
        f"Виклики (вибірка / пряма оптимізація): "
        f"{n_train} / {direct_calls['calls']}"
    )
    print(
        f"x* (сур / пряма)  : {np.round(x_sur,4)}  /  {np.round(res_dir.x,4)}"
    )
    print(
        f"f* (сур / пряма)  : {f_sur:.6f} / {res_dir.fun:.6f}"
    )

    Y_true = np.array(
        [
            branin(pt * (surrogate.X_max - surrogate.X_min) + surrogate.X_min)
            for pt in surrogate.X_test_norm
        ]
    )
    Y_pred = np.array(
        [surrogate.surrogate_predict(pt) for pt in surrogate.X_test_norm]
    )

    plt.figure(figsize=(5, 5))
    plt.scatter(Y_true, Y_pred, c="crimson", edgecolor="k", s=40)
    mn, mx = Y_true.min(), Y_true.max()
    plt.plot([mn, mx], [mn, mx], "k--")
    plt.xlabel("Справжнє $f$")
    plt.ylabel("Передбачене $f$")
    plt.title(f"Парний графік — {uk_map[sample_method].capitalize()} вибірка")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    for method in ("uniform", "lhs"):
        run_experiment(n_train=30, sample_method=method)
