import numpy as np
from scipy.integrate import odeint
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pyDOE2 import lhs
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from bachelors.RBF_MODULE import (
    generate_plan,
    normalize,
    get_rbf_func,
    rbf_interpolation,
    rbf_predict,
    cross_val_score_rbf,
)

np.random.seed(42)


#  ---------SEIR ADAPTIVE-------------
# def beta(a):
#     """
#     SMOOTH exposure rate (transmission).
#     Uses a sigmoid function to model how the transmission rate grows with virality.
#     It ranges from a low base rate to a high rate for very viral news.
#     """
#     base_rate = 0.1
#     max_rate = 0.9
#     return base_rate + (max_rate - base_rate) * (1 / (1 + np.exp(-10 * (a - 0.5))))

# def alpha(a):
#     """
#     SMOOTH fraction of exposed who become spreaders.
#     More viral news (higher 'a') makes people more likely to share.
#     """
#     min_share_fraction = 0.1
#     max_share_fraction = 0.8
#     return min_share_fraction + (max_share_fraction - min_share_fraction) * a**2

# zeta = 0.2
# gamma = 0.1

# def seir_model(y, t, a):
#     """
#     SEIR-like ODE system for fake news spread with SMOOTH parameters.
#     y = [S, E, I, R],  a = 'virality_index' in [0,1]
#     """
#     S, E, I, R = y
#     b = beta(a)
#     al = alpha(a)
#     dSdt = -b * S * I
#     dEdt = b * S * I - zeta * E
#     dIdt = zeta * al * E - gamma * I
#     dRdt = zeta * (1 - al) * E + gamma * I
#     return [dSdt, dEdt, dIdt, dRdt]

# def simulate_model(a, t_max=200, N=1.0, E0=0.05, I0=0.01, R0=0.0):
#     """
#     Simulates the SEIR system over [0, t_max] for a given virality_index 'a'.
#     Returns the cumulative infected measure: int_0^T I(t) dt.
#     """
#     a = float(a)
#     S0 = N - E0 - I0 - R0
#     y0 = [S0, E0, I0, R0]
#     t = np.linspace(0, t_max, t_max + 1)
#     sol = odeint(seir_model, y0, t, args=(a,))
#     return np.trapz(sol[:, 2], t)

# param_bounds = np.array([[0.0, 1.0]])




#  ---------SEIR CLASSIC-------------

def model_func(y, t, beta, mu, gamma, alpha):
        S, E, I, R = y
        dSdt = -beta * S * I
        dEdt = beta * S * I - mu * E
        dIdt = -gamma * I + alpha * mu * E
        dRdt = gamma * I + (1 - alpha) * mu * E
        return [dSdt, dEdt, dIdt, dRdt]

def simulate_model(params):
    beta, mu, gamma, alpha = params
    N = 1
    E0, I0, R0 = 0.25, 0.05, 0
    S0 = N - I0 - R0 - E0
    y0 = [S0, E0, I0, R0]
    t = np.linspace(0, 500, 500)
    sol = odeint(model_func, y0, t, args=(beta, mu,gamma, alpha))
    return np.sum(sol[:, 2])


param_bounds = np.array([
    [0.1, 0.3], 
    [0.005, 0.01], 
    [0.005, 0.01], 
    [0.2, 0.5]
])



#  ---------BRANIN FUNCTION-------------
# def simulate_model(params):
#     b = 5.1 / (4 * np.pi**2)
#     c = 5 / np.pi
#     r = 6.0
#     s = 10.0
#     t = 1 / (8 * np.pi)
#     x1 = params[0]
#     x2 = params[1]
#     return (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s

# param_bounds = [[-5, 10], [0, 15]]

# def model_to_plot(params):
#     b = 5.1 / (4 * np.pi**2)
#     c = 5 / np.pi
#     r = 6.0
#     s = 10.0
#     t = 1 / (8 * np.pi)
#     x1 = params[0]
#     x2 = params[1]
#     return (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s



def expand_bounds(bounds, expansion_ratio=0.05):
    lower, upper = bounds[:, 0], bounds[:, 1]
    delta = (upper - lower) * expansion_ratio
    return np.stack([lower - delta, upper + delta], axis=1)

def generate_plan(n, bounds, method='lhs'):
    dim = bounds.shape[0]
    if method == 'lhs':
        samples = lhs(dim, samples=n)
    elif method == 'uniform':
        samples = np.random.rand(n, dim)
    else:
        raise ValueError("Unsupported sampling method")
    return bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * samples


def find_best_epsilon(X_train, Y_train, rbf_type, epsilons, k=5):
    """
    Знаходить найкращий епсилон за допомогою крос-валідації на тренувальних даних.
    """
    best_eps = None
    best_cv_mse = float('inf')

    for eps in epsilons:
        current_mse = cross_val_score_rbf(X_train, Y_train, rbf_type, eps, k=k)
        if not np.isnan(current_mse) and current_mse < best_cv_mse:
            best_cv_mse = current_mse
            best_eps = eps
            
    return best_eps, best_cv_mse


def main():
    #SEIR
    n_train, n_test = 50, 15
    #BRANIN
    n_train, n_test = 60, 20

    plan_methods = ['lhs', 'uniform']
    rbf_types    = ['cubic', 'gaussian', 'multiquadric', 'inverse_multiquadric', 'linear']
    epsilons     = [0.3, 1.0, 1.5, 3.0, 5.0]

    expanded_bounds = expand_bounds(param_bounds, expansion_ratio=0.05)

    header = f"{'Plan':<8} {'RBF Type':<20} {'Epsilon':<8} {'MAE':>10} {'RMSE':>10} {'R2':>10}{'CV_MSE':>10}"
    print(header)
    print("-" * len(header))

    results = []

    for plan_method in plan_methods:
        train_bounds = expanded_bounds if plan_method == 'lhs' else param_bounds

        X_train_real = generate_plan(n_train, train_bounds, plan_method)
        Y_train = np.array([simulate_model(p) for p in X_train_real])

        X_test_real  = generate_plan(n_test, param_bounds, plan_method)
        Y_test  = np.array([simulate_model(p) for p in X_test_real])

        X_train_norm = normalize(X_train_real, train_bounds)
        X_test_norm = normalize(X_test_real, param_bounds)

        for rbf_type in rbf_types:
            if rbf_type in ['gaussian', 'multiquadric', 'inverse_multiquadric']:
                for eps in epsilons:
                    try:
                        lambdas = rbf_interpolation(X_train_norm, Y_train, rbf_type, eps)
                        Y_pred  = rbf_predict(X_train_norm, lambdas, X_test_norm, rbf_type, eps)
                        mae  = mean_absolute_error(Y_test, Y_pred)
                        rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
                        r2   = r2_score(Y_test, Y_pred)
                        cv_mse = cross_val_score_rbf(X_train_norm, Y_train, rbf_type, eps)

                    except Exception:
                        mae = rmse = r2 = float('nan')

                    print(f"{plan_method:<8} {rbf_type:<20} {eps:<8.2f} {mae:10.4f} {rmse:10.4f} {r2:10.4f} {cv_mse:10.4f}")
                    results.append({ "Plan": plan_method, "RBF Type": rbf_type, "Epsilon": eps, "MAE": mae, "RMSE": rmse, "R2": r2 ,"CV_MSE": cv_mse})
            else: 
                try:
                    lambdas = rbf_interpolation(X_train_norm, Y_train, rbf_type)
                    Y_pred  = rbf_predict(X_train_norm, lambdas, X_test_norm, rbf_type)
                    mae  = mean_absolute_error(Y_test, Y_pred)
                    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
                    r2   = r2_score(Y_test, Y_pred)
                    cv_mse = cross_val_score_rbf(X_train_norm, Y_train, rbf_type)
                except Exception:
                    mae = rmse = r2 = float('nan')
                print(f"{plan_method:<8} {rbf_type:<20} {'-':<8} {mae:10.4f} {rmse:10.4f} {r2:10.4f} ")
                results.append({ "Plan": plan_method, "RBF Type": rbf_type, "Epsilon": None, "MAE": mae, "RMSE": rmse, "R2": r2 })

    valid_results = [r for r in results if not np.isnan(r["R2"])]
    if valid_results:
        best_result = max(valid_results, key=lambda r: r["R2"])
        print("\n✅ Best configuration (with normalization):")
        print(f"Plan     : {best_result['Plan']}")
        print(f"RBF Type : {best_result['RBF Type']}")
        print(f"Epsilon  : {best_result['Epsilon'] if best_result['Epsilon'] is not None else '-'}")
        print(f"MAE      : {best_result['MAE']:.4f}")
        print(f"RMSE     : {best_result['RMSE']:.4f}")
        print(f"R²       : {best_result['R2']:.4f}")
    else:
        print("\nNo valid results found.")

if __name__ == "__main__":
    main()

