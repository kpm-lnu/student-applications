import numpy as np
import importlib.util
import sys
import matplotlib.pyplot as plt
from RBF_MODULE import (
    denormalize, generate_plan,
    optimize_true_model_hybrid,
    HybridRBFSurrogate, optimize_hybrid_surrogate, evaluate_model,call_counter
)

def load_input_from_file(filepath):
    spec = importlib.util.spec_from_file_location("user_input", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def run_experiment(sampling_method, sim_func, bounds):
    """
    Runs a full experiment for a given sampling method.
    """
    print(f"\n{'='*20} RUNNING EXPERIMENT FOR SAMPLING: {sampling_method.upper()} {'='*20}")
    n_train, n_test = 40, 15

    X_train_norm = generate_plan(n_train, bounds.shape[0], sampling_method)
    X_train_real = denormalize(X_train_norm, bounds)
    Y_train = np.array([sim_func(x) for x in X_train_real])

    hybrid_surrogate = HybridRBFSurrogate(X_train_norm, Y_train)
    x_opt_sur, f_opt_sur = optimize_hybrid_surrogate(hybrid_surrogate, bounds)
    true_value_at_sur_opt = sim_func(x_opt_sur)
    
    X_test_norm = generate_plan(n_test, bounds.shape[0], 'lhs') # Use LHS for consistent test sets
    X_test_real = denormalize(X_test_norm, bounds)
    Y_test_true = np.array([sim_func(x) for x in X_test_real])
    Y_test_pred = hybrid_surrogate.predict(X_test_norm)
    metrics = evaluate_model(Y_test_true, Y_test_pred, f"Hybrid Model with {sampling_method.upper()} Sampling")

    plt.figure(figsize=(8, 6))
    plt.scatter(Y_test_true, Y_test_pred, c='royalblue', alpha=0.8, label=f'{sampling_method.upper()} Test Points')
    plt.plot([min(Y_test_true), max(Y_test_true)], [min(Y_test_true), max(Y_test_true)], 'r--', label='Ideal')
    plt.title(f'Hybrid Surrogate: True vs. Predicted ({sampling_method.upper()})')
    plt.xlabel('True Values'); plt.ylabel('Predicted Values')
    plt.legend(); plt.grid(True); plt.show()

    return {
        "sampling_method": sampling_method,
        "metrics": metrics,
        "surrogate_opt_x": x_opt_sur,
        "surrogate_opt_val": f_opt_sur,
        "true_value_at_sur_opt": true_value_at_sur_opt,
        "model_structure": hybrid_surrogate.models
    }


def main(input_file_path):
    data = load_input_from_file(input_file_path)
    sim_func, bounds = data.simulate_model, np.array(data.param_bounds)

    print("--- Finding True Optimum (Baseline) ---")
    result_true_x, result_true_fun = optimize_true_model_hybrid(sim_func, bounds)
    print(f"🔹 True Optimum: x={result_true_x}, f(x)={result_true_fun:.4f}\n")
    print(f"🔹 Function Calls: {call_counter['corrected_model_calls']}\n")


    plans_to_test = ['lhs', 'uniform']
    results = []
    for plan in plans_to_test:
        try:
            exp_result = run_experiment(plan, sim_func, bounds)
            results.append(exp_result)
        except Exception as e:
            print(f"💥 Experiment failed for {plan.upper()} sampling: {e}")

    if not results:
        print("\nNo experiments succeeded.")
        return

    best_experiment = max(results, key=lambda r: r['metrics']['r2'])
    
    print(f"\n{'='*25} FINAL RESULTS {'='*25}")
    print(f"🏆 Best performing sampling method: {best_experiment['sampling_method'].upper()} (based on R² score)")
    
    print("\n--- Summary of the Winning Model ---")
    print(f"Sampling Method: {best_experiment['sampling_method'].upper()}")
    print(f"Accuracy (R²): {best_experiment['metrics']['r2']:.4f}")
    print(f"Optimal Parameters Found: {best_experiment['surrogate_opt_x']}")
    print(f"Optimal Value: {best_experiment['surrogate_opt_val']:.4f}")
    print(f"True Value at Surrogate Optimum: {best_experiment['true_value_at_sur_opt']:.4f}")
    
    
    print("\nModel Structure:")
    if len(best_experiment['model_structure']) == 1:
        config = best_experiment['model_structure'][0]
        print("  - Single Global Model was built.")
        print(f"  - RBF Config: type={config['rbf_type']}, epsilon={config['epsilon']}")
    else:
        print(f"  - Hybrid model with {len(best_experiment['model_structure'])} clusters was built.")
        for i, config in enumerate(best_experiment['model_structure']):
            print(f"    - Cluster {i}: RBF type={config['rbf_type']}, epsilon={config['epsilon']}")
    
    print(f"\n--- For Reference: True Optimum ---")
    print(f"Optimal Parameters: {result_true_x}")
    print(f"Value at Optimum: {result_true_fun:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main-2_division.py path/to/your/input_file.py")
        sys.exit(1)
    main(sys.argv[1])