"""
kriging_module.py

This module implements ordinary kriging for surrogate modeling.
It provides a KrigingSurrogate class that can be used with any 
objective function defined over an arbitrary number of parameters.
It also supports training the surrogate from a pre-existing dataset.

"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

try:
    from smt.sampling_methods import LHS
    HAS_SMT = True
except ImportError:
    HAS_SMT = False


def correlation(x1, x2, theta):
    """
    Compute the correlation between two points.
    Uses an exponential kernel: exp(-sum(theta * (x1 - x2)^2)).
    """
    return np.exp(-np.sum(theta * (x1 - x2) ** 2))


def build_correlation_matrix(X, theta, nugget=1e-6):
    """
    Build the correlation matrix for a set of samples.
    A nugget is added on the diagonal for stability.
    """
    n = X.shape[0]
    R = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            R[i, j] = correlation(X[i], X[j], theta)
    R += np.eye(n) * nugget
    return R


def ordinary_kriging_predict(x_new, X_train, Y_train, R, theta):
    """
    Predicts the output at a new normalized point x_new.
    """
    n = X_train.shape[0]
    r = np.zeros(n)
    for i in range(n):
        r[i] = correlation(x_new, X_train[i], theta)
    # Build the augmented system for ordinary kriging.
    A = np.block([[R, np.ones((n, 1))],
                  [np.ones((1, n)), np.zeros((1, 1))]])
    b = np.hstack([r, 1])
    sol = np.linalg.solve(A, b)
    lambdas = sol[:n]
    f_pred = np.dot(lambdas, Y_train.flatten())
    return f_pred


def log_likelihood(theta_array, X_train, Y_train, nugget=1e-6):
    """
    Computes the negative log-likelihood based on the current theta.
    This function is minimized to calibrate the correlation parameters.
    """
    theta = theta_array
    R = build_correlation_matrix(X_train, theta, nugget)
    n = X_train.shape[0]
    ones = np.ones((n, 1))
    R_inv = np.linalg.inv(R)
    
    # Solve for the trend parameter mu.
    mu = (ones.T @ R_inv @ Y_train) / (ones.T @ R_inv @ ones)
    residual = Y_train - mu * ones

    L = np.linalg.cholesky(R)
    logdetR = 2.0 * np.sum(np.log(np.diag(L)))
    quad = (residual.T @ R_inv @ residual)[0, 0]
    return logdetR + n * np.log(quad / n)


class KrigingSurrogate:
    """
    A class for building and optimizing a Kriging surrogate model.
    Supports training based on function sampling or using a dataset.
    """

    def __init__(self, objective_func=None, domain=None, sample_method="uniform", n_samples=100,
                 train_ratio=0.7, random_seed=0, nugget=1e-6,
                 dataset=None, dataset_filepath=None, x_columns=None, y_column="f"):
        """
        Initialize the surrogate model.
        
        Parameters:
            objective_func : callable, optional
                The function to be approximated. It must accept a numpy array x of shape (d,)
                and return a scalar. Not required if dataset is provided.
            domain : list, optional
                A list of [lower, upper] pairs for each parameter. Required if no dataset is provided,
                or if you want to use sampling-based methods.
            sample_method : str, optional
                'uniform' or 'lhs' (Latin Hypercube Sampling). Default is 'uniform'.
            n_samples : int, optional
                Number of samples to generate in the domain. Default is 100.
            train_ratio : float, optional
                Ratio of samples used for training. The rest are used for testing.
            random_seed : int, optional
                Seed for random number generator.
            nugget : float, optional
                Nugget parameter added to the diagonal of the correlation matrix for stability.
            dataset : pandas.DataFrame, optional
                A pre-existing dataset containing training samples. Should include predictor columns 
                and one target column.
            dataset_filepath : str, optional
                Path to a CSV file containing data. Used if dataset is not provided directly.
            x_columns : list, optional
                List of column names for input variables in the dataset.
                If not provided and dataset is given, all columns except y_column will be used.
            y_column : str, optional
                Name of the column containing the target function values. Default is "f".
        """
        self.objective_func = objective_func
        self.domain = np.array(domain) if domain is not None else None
        self.sample_method = sample_method
        self.n_samples = n_samples
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.nugget = nugget
        
        self.dataset = dataset
        self.dataset_filepath = dataset_filepath
        self.x_columns = x_columns
        self.y_column = y_column
        
        self.X_all = None
        self.Y_all = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        
        self.X_min = None
        self.X_max = None
        self.X_train_norm = None
        self.X_test_norm = None
        
        self.theta_opt = None
        self.R_train = None
        self.res_surrogate = None

    def sample_data(self):
        """
        Loads/samples the data and computes objective values.
        Splits the data into training and test sets and normalizes the inputs.

        If a dataset (or dataset_filepath) is provided, it uses that data.
        Otherwise, it samples points from the given domain using the objective function.
        """
        np.random.seed(self.random_seed)
        
        # Use dataset if available.
        if self.dataset is not None or (self.dataset_filepath is not None and os.path.exists(self.dataset_filepath)):
            if self.dataset is None:
                self.dataset = pd.read_csv(self.dataset_filepath)
            
            if self.x_columns is None:
                self.x_columns = [col for col in self.dataset.columns if col != self.y_column]
            
            self.X_all = self.dataset[self.x_columns].values
            self.Y_all = self.dataset[self.y_column].values.reshape(-1, 1)
            
            if self.domain is None:
                lower = self.X_all.min(axis=0)
                upper = self.X_all.max(axis=0)
                self.domain = np.array([[l, u] for l, u in zip(lower, upper)])
        else:
            # When dataset is not provided, use function sampling.
            if self.objective_func is None or self.domain is None:
                raise ValueError("When no dataset is provided, both objective_func and domain must be given.")
            d = self.domain.shape[0]
            if self.sample_method == "lhs":
                if not HAS_SMT:
                    raise ImportError("smt library is required for LHS sampling. Please install it.")
                from smt.sampling_methods import LHS  # Import locally to avoid issues.
                sampling = LHS(xlimits=self.domain)
                self.X_all = sampling(self.n_samples)
            else:
                lower_bounds = self.domain[:, 0]
                upper_bounds = self.domain[:, 1]
                self.X_all = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(self.n_samples, d))
            
            # Evaluate the function at all sample points.
            self.Y_all = np.array([self.objective_func(x) for x in self.X_all]).reshape(-1, 1)
        
        # Randomly split the data into training and testing sets.
        indices = np.arange(self.X_all.shape[0])
        np.random.shuffle(indices)
        train_size = int(self.train_ratio * len(indices))
        train_idx = indices[:train_size]
        test_idx = indices[train_size:]
        self.X_train = self.X_all[train_idx]
        self.Y_train = self.Y_all[train_idx]
        self.X_test = self.X_all[test_idx]
        self.Y_test = self.Y_all[test_idx]
        
        # Normalize inputs using training set min and max.
        self.X_min = self.X_train.min(axis=0)
        self.X_max = self.X_train.max(axis=0)
        self.X_train_norm = (self.X_train - self.X_min) / (self.X_max - self.X_min)
        self.X_test_norm = (self.X_test - self.X_min) / (self.X_max - self.X_min)

    def fit(self):
        """
        Fit the surrogate model by optimizing the theta parameters.
        The optimization is performed in the normalized input space.
        """
        if self.X_train_norm is None:
            self.sample_data()
        
        d = self.domain.shape[0]
        # Optimize theta (correlation parameters) using L-BFGS-B.
        res_theta = minimize(log_likelihood, x0=np.ones(d),
                             args=(self.X_train_norm, self.Y_train, self.nugget),
                             bounds=[(1e-3, 100)] * d,
                             method='L-BFGS-B')
        self.theta_opt = res_theta.x
        self.R_train = build_correlation_matrix(self.X_train_norm, self.theta_opt, self.nugget)

    def surrogate_predict(self, x):
        """
        Predict the function value at a new point x (in normalized space).
        
        Parameters:
            x : numpy array of shape (d,)
                New point in normalized space.
                
        Returns:
            float : Predicted function value.
        """
        return ordinary_kriging_predict(x, self.X_train_norm, self.Y_train, self.R_train, self.theta_opt)

    def optimize_surrogate(self, initial_guess=None, bounds_norm=None, method='Powell'):
        """
        Optimize the surrogate model over the normalized space.
        
        Parameters:
            initial_guess : array-like, optional
                Initial guess in normalized space. If None, the center (0.5,...,0.5) is used.
            bounds_norm : list of tuples, optional
                Bounds for the optimization in normalized space. Default is [(0, 1), ...].
            method : str, optional
                Optimization method (default 'Powell').
                
        Returns:
            res: The result of the optimization.
        """
        d = self.domain.shape[0]
        if initial_guess is None:
            initial_guess = np.full(d, 0.5)
        if bounds_norm is None:
            bounds_norm = [(0, 1)] * d

        def surrogate_obj(x):
            return self.surrogate_predict(x)

        self.res_surrogate = minimize(surrogate_obj, initial_guess, bounds=bounds_norm, method=method)
        return self.res_surrogate

    def evaluate_performance(self, return_relative=True):
        """
        Evaluate surrogate accuracy on the test set.

        Returns
        -------
        rmse : float
            Absolute RMSE.
        mae : float
            Absolute MAE.
        rrmse_pct : float
            Relative RMSE in %, if `return_relative=True`.
            Defined as RMSE / (max − min) × 100 of Y_test.
        r2 : float
            R-squared score.
        """
        # Predictions on the normalized test inputs
        Y_pred_test = np.array([self.surrogate_predict(x) for x in self.X_test_norm])

        # Absolute errors
        rmse = np.sqrt(mean_squared_error(self.Y_test, Y_pred_test))
        mae  = mean_absolute_error(self.Y_test, Y_pred_test)
        r2   = r2_score(self.Y_test, Y_pred_test)

        if not return_relative:
            return rmse, mae, r2

        # Relative RMSE (% of the data span)
        y_span = float(self.Y_test.max() - self.Y_test.min())
        rrmse_pct = np.nan if y_span == 0 else 100.0 * rmse / y_span

        return rmse, mae, rrmse_pct, r2


    def plot_results(self, grid_points=100, show_train=True, show_test=True, show_optimum=True, optimum_point=None):
        """
        Visualize the surrogate model predictions over a grid (only for 2D problems).
        
        Parameters:
            grid_points : int, optional
                Number of grid points per dimension.
            show_train : bool, optional
                If True, plot the training data points.
            show_test : bool, optional
                If True, plot the testing data points.
            show_optimum : bool, optional
                If True, mark the optimum point.
            optimum_point : array-like, optional
                Optimum point in the original scale to be plotted.
                
        Raises:
            ValueError: if the problem dimension is not 2.
        """
        d = self.domain.shape[0]
        if d != 2:
            raise ValueError("Plotting is only supported for 2D problems.")
        
        nx, ny = grid_points, grid_points
        x1_lin_norm = np.linspace(0, 1, nx)
        x2_lin_norm = np.linspace(0, 1, ny)
        X1_norm, X2_norm = np.meshgrid(x1_lin_norm, x2_lin_norm)
        X_grid_norm = np.vstack([X1_norm.ravel(), X2_norm.ravel()]).T

        # Compute surrogate predictions on the normalized grid.
        Y_pred_grid = np.array([self.surrogate_predict(x) for x in X_grid_norm])
        Y_pred_grid = Y_pred_grid.reshape(nx, ny)

        # Map grid to original input scale.
        X_grid_orig = X_grid_norm * (self.X_max - self.X_min) + self.X_min
        X1_orig = X_grid_orig[:, 0].reshape(nx, ny)
        X2_orig = X_grid_orig[:, 1].reshape(nx, ny)

        plt.figure(figsize=(12, 5))
        cp = plt.contourf(X1_orig, X2_orig, Y_pred_grid, levels=20, cmap='viridis')
        if show_train:
            plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c='red', edgecolor='k', s=10, label='Train Samples')
        if show_test:
            plt.scatter(self.X_test[:, 0], self.X_test[:, 1], c='white', edgecolor='k', s=10, label='Test Samples')
        if show_optimum and optimum_point is not None:
            plt.plot(optimum_point[0], optimum_point[1], 'kx', markersize=12, label='Optimum')
        plt.title("Ordinary Kriging Surrogate Prediction")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.colorbar(cp)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    pass