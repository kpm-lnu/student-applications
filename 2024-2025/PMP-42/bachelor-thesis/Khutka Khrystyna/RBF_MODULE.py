
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from pyDOE2 import lhs
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

np.random.seed(42)
call_counter = {"corrected_model_calls": 0}

def get_rbf_func(rbf_type, epsilon):
    rbf_type = rbf_type.lower()
    if rbf_type == 'cubic': return lambda r: r**3
    elif rbf_type == 'linear': return lambda r: r
    elif rbf_type == 'gaussian': return lambda r: np.exp(-(epsilon * r)**2)
    elif rbf_type == 'multiquadric': return lambda r: np.sqrt(1 + (epsilon * r)**2)
    elif rbf_type == 'inverse_multiquadric': return lambda r: 1.0 / (1 + (epsilon * r)**2)
    else: raise ValueError("Unsupported RBF type")

def rbf_interpolation(X_train, Y_train, rbf_type, epsilon):
    phi = get_rbf_func(rbf_type, epsilon)
    A = phi(cdist(X_train, X_train))
    nugget = 1e-3
    A += np.eye(A.shape[0]) * nugget
    try:
        return np.linalg.solve(A, Y_train)
    except np.linalg.LinAlgError:
        return None

def rbf_predict(X_train, lambdas, X_test, rbf_type, epsilon):
    if lambdas is None: return np.full(X_test.shape[0], np.nan)
    phi = get_rbf_func(rbf_type, epsilon)
    B = phi(cdist(X_test, X_train))
    return B @ lambdas

def generate_plan(n, dim, method='lhs'):
    if method == 'lhs': return lhs(dim, samples=n)
    elif method == 'uniform': return np.random.rand(n, dim)
    else: raise ValueError("Unsupported sampling method")

def normalize(X, bounds):
    return (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

def denormalize(X_norm, bounds):
    return bounds[:, 0] + X_norm * (bounds[:, 1] - bounds[:, 0])

def optimize_true_model_hybrid(sim_func, param_bounds):
    bounds_array = np.array(param_bounds)
    bounds_norm = [(0, 1)] * bounds_array.shape[0]
    call_counter["corrected_model_calls"] = 0
    def sim_func_norm(x_norm):
        call_counter["corrected_model_calls"] += 1
        return sim_func(denormalize(x_norm, bounds_array))
    result = minimize(sim_func_norm, x0=np.full(bounds_array.shape[0], 0.5), bounds=bounds_norm, method='Powell')
    return denormalize(result.x, bounds_array), result.fun

def cross_val_score_rbf(X, Y, rbf_type, epsilon, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    errors = []
    for train_idx, test_idx in kf.split(X):
        if len(train_idx) < X.shape[1] + 1: continue
        lambdas = rbf_interpolation(X[train_idx], Y[train_idx], rbf_type, epsilon)
        if lambdas is None: continue
        y_pred = rbf_predict(X[train_idx], lambdas, X[test_idx], rbf_type, epsilon)
        errors.append(mean_squared_error(Y[test_idx], y_pred))
    return np.nanmean(errors) if errors else float('inf')

def tune_rbf_hyperparameters(X_train, Y_train):
    rbf_options = ['gaussian', 'multiquadric', 'inverse_multiquadric', 'cubic', 'linear']
    epsilon_options = [0.1, 0.5, 1.0, 1.5, 3.0]
    best_config = {'rbf_type': 'linear', 'epsilon': None, 'cv_mse': float('inf')}
    for rbf_type in rbf_options:
        if rbf_type in ['cubic', 'linear']:
            cv_mse = cross_val_score_rbf(X_train, Y_train, rbf_type, None)
            if cv_mse < 1e-8: continue
            if cv_mse < best_config['cv_mse']:
                best_config.update({'rbf_type': rbf_type, 'epsilon': None, 'cv_mse': cv_mse})
        else:
            for eps in epsilon_options:
                cv_mse = cross_val_score_rbf(X_train, Y_train, rbf_type, eps)
                if cv_mse < 1e-8: continue
                if cv_mse < best_config['cv_mse']:
                    best_config.update({'rbf_type': rbf_type, 'epsilon': eps, 'cv_mse': cv_mse})
    return best_config

def find_optimal_clusters(X_norm, Y_norm, max_k=5, silhouette_threshold=0.55):
    print(f"\n🤖 Searching for optimal number of clusters (max_k={max_k})...")
    combined_data = np.hstack((X_norm, Y_norm.reshape(-1, 1)))
    best_k, best_score = 1, -1.0
    for k in range(2, max_k + 1):
        if len(combined_data) <= k: break
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(combined_data)
        score = silhouette_score(combined_data, kmeans.labels_)
        print(f"  ➤ For k={k}, silhouette score = {score:.4f}")
        if score > best_score:
            best_score, best_k = score, k
    if best_score < silhouette_threshold:
        print(f"  ➤ Best score {best_score:.4f} is below threshold {silhouette_threshold}. No division needed.")
        return 1, None
    print(f"  ➤ Optimal k = {best_k} found with score {best_score:.4f}.")
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(combined_data)
    return best_k, kmeans.labels_

class HybridRBFSurrogate:
    def __init__(self, X_train_norm, Y_train):
        self.X_train_norm = X_train_norm
        self.Y_train = Y_train
        self.models = []
        self.centroids = []
        self._build_model()

    def _build_model(self):
        y_min, y_max = self.Y_train.min(), self.Y_train.max()
        Y_norm = (self.Y_train - y_min) / (y_max - y_min) if (y_max - y_min) > 0 else np.zeros_like(self.Y_train)
        num_clusters, labels = find_optimal_clusters(self.X_train_norm, Y_norm)
        if num_clusters == 1:
            print("\nBUILDING A SINGLE GLOBAL MODEL")
            config = tune_rbf_hyperparameters(self.X_train_norm, self.Y_train)
            print(f"  ➤ Best config: {config['rbf_type']}, epsilon={config['epsilon']}, CV_MSE={config['cv_mse']:.4f}")
            lambdas = rbf_interpolation(self.X_train_norm, self.Y_train, config['rbf_type'], config['epsilon'])
            config.update({'lambdas': lambdas, 'X_train': self.X_train_norm})
            self.models.append(config)
            self.centroids.append(np.mean(self.X_train_norm, axis=0))
        else:
            print(f"\nBUILDING {num_clusters} SEPARATE MODELS")
            for i in range(num_clusters):
                mask = (labels == i)
                X_cluster, Y_cluster = self.X_train_norm[mask], self.Y_train[mask]
                if len(X_cluster) < 5: continue
                print(f"\nTuning hyperparameters for cluster {i} ({len(X_cluster)} points)...")
                config = tune_rbf_hyperparameters(X_cluster, Y_cluster)
                print(f"  ➤ Best config for cluster {i}: {config['rbf_type']}, epsilon={config['epsilon']}, CV_MSE={config['cv_mse']:.4f}")
                lambdas = rbf_interpolation(X_cluster, Y_cluster, config['rbf_type'], config['epsilon'])
                config.update({'lambdas': lambdas, 'X_train': X_cluster})
                self.models.append(config)
                self.centroids.append(np.mean(X_cluster, axis=0))

    def predict(self, X_test_norm):
        predictions = np.zeros(len(X_test_norm))
        for i, x_test in enumerate(X_test_norm):
            if not self.centroids: return np.full(len(X_test_norm), np.nan)
            distances = [np.linalg.norm(x_test - centroid) for centroid in self.centroids]
            idx = np.argmin(distances)
            model = self.models[idx]
            predictions[i] = rbf_predict(model['X_train'], model['lambdas'], [x_test], model['rbf_type'], model['epsilon'])
        return predictions

def optimize_hybrid_surrogate(surrogate_model, param_bounds):
    bounds_norm = [(0, 1)] * param_bounds.shape[0]
    def objective(x_norm):
        return surrogate_model.predict([x_norm])[0]
    result = minimize(objective, x0=np.full(param_bounds.shape[0], 0.5), bounds=bounds_norm, method='Powell')
    return denormalize(result.x, param_bounds), result.fun



def optimize_surrogate_model(X_train_norm, Y_train, param_bounds, rbf_type='multiquadric', epsilon=1.0):
    lambdas = rbf_interpolation(X_train_norm, Y_train, rbf_type, epsilon)

    x0_norm = np.full(X_train_norm.shape[1], 0.5)

    def surrogate(x_norm):
        return rbf_predict(X_train_norm, lambdas, [x_norm], rbf_type, epsilon)[0]

    result = minimize(
        surrogate,
        x0_norm,
        bounds=[(0, 1)] * X_train_norm.shape[1],
        method='Powell'
    )

    x_opt_real = denormalize(result.x, param_bounds)
    return x_opt_real, result.fun

def benchmark_selected_surrogate(simulate_model, param_bounds,
                                 sampling_method='lhs', rbf_type='gaussian',
                                 n_train=35, n_test=15, epsilon=1.0):
    """
    Benchmarks the surrogate model for a given RBF type and sampling method.

    Parameters:
        simulate_model (function): The true simulation model.
        param_bounds (list of [min, max]): Parameter bounds.
        sampling_method (str): 'lhs' or 'uniform'.
        rbf_type (str): RBF kernel type (e.g., 'gaussian', 'cubic').
        n_train (int): Number of training samples.
        n_test (int): Number of test samples.
        epsilon (float): RBF kernel parameter.
    """
    import matplotlib.pyplot as plt

    print(f"\n📌 Benchmarking: RBF = {rbf_type.upper()} | Sampling = {sampling_method.upper()}")

    X_train = generate_plan(n_train, len(param_bounds), sampling_method)
    X_test = generate_plan(n_test, len(param_bounds), sampling_method)

    X_train_real = denormalize(X_train, param_bounds)
    X_test_real = denormalize(X_test, param_bounds)

    Y_train = np.array([simulate_model(x) for x in X_train_real])
    Y_test = np.array([simulate_model(x) for x in X_test_real])

    try:
        lambdas = rbf_interpolation(X_train, Y_train, rbf_type, epsilon)
        Y_pred = rbf_predict(X_train, lambdas, X_test, rbf_type, epsilon)

        evaluate_model(Y_test, Y_pred, f'RBF ({rbf_type}) on {sampling_method}')
        plt.figure()
        plt.scatter(Y_test, Y_pred, color='crimson', label=f'RBF: {rbf_type}')
        plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', label='Ideal')
        plt.xlabel('True values')
        plt.ylabel('Predicted values')
        plt.title(f'Benchmark: RBF = {rbf_type} | Sampling = {sampling_method}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"⚠️ Benchmark failed for RBF ({rbf_type}) with {sampling_method}: {e}")





def evaluate_model(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n📊 {name}")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  R²   = {r2:.4f}")
    return {'mae': mae, 'rmse': rmse, 'r2': r2}