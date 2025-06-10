
import numpy as np
from scipy.integrate import odeint
import importlib.util
import sys
from RBF_MODULE import (
    normalize, denormalize, generate_plan,
    optimize_surrogate_model,
    call_counter,optimize_true_model_hybrid, benchmark_selected_surrogate 
)



def load_input_from_file(filepath):
    spec = importlib.util.spec_from_file_location("user_input", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def expand_bounds(bounds, expansion_ratio=0.05):
    lower, upper = bounds[:, 0], bounds[:, 1]
    delta = (upper - lower) * expansion_ratio
    return np.stack([lower - delta, upper + delta], axis=1)

def main(input_file_path):
    data = load_input_from_file(input_file_path)

    sim_func = data.simulate_model
    

    bounds = np.array(data.param_bounds)
    n_train = data.n_train

    #USE ONLY IF LHS!!!!!!!!
    #---------
    expanded_bounds = expand_bounds(bounds, expansion_ratio=0.05)
    bounds= expanded_bounds
    #---------

    result_true_x, result_true_fun = optimize_true_model_hybrid(sim_func, bounds)
    print("🔹 True optimization:")
    print("  ➤ Optimal params:", result_true_x)
    print("  ➤ Objective value:", result_true_fun)
    print("  ➤ Function calls:", call_counter["corrected_model_calls"])

    X_train_norm = generate_plan(n_train, bounds.shape[0], 'lhs')  
    X_train_real = denormalize(X_train_norm, bounds)
    Y_train = np.array([sim_func(x) for x in X_train_real])

    x_opt_sur, f_opt_sur = optimize_surrogate_model(X_train_norm, Y_train, bounds, 'gaussian',1)
    print("🔹 Surrogate optimization:")
    print("  ➤ Optimal surrogate params:", x_opt_sur)
    print("  ➤ Surrogate value:", f_opt_sur)
    true_value_at_sur_opt = sim_func(x_opt_sur)
    print("  ➤ True value at surrogate optimum:", true_value_at_sur_opt)
    
    benchmark_selected_surrogate(sim_func, bounds,
                             sampling_method='lhs',     
                             rbf_type='gaussian',
                             n_train=n_train,    
                             n_test=15,
                             epsilon= 1.0)
  
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py path_to_input_file.py")
    else:
        main(sys.argv[1])

