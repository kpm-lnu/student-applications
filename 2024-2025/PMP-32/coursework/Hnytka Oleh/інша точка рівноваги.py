from scipy.optimize import fsolve


s = 0.8
L = 50
alpha_1 = 0.0031
lambda_2 = 0.0007
pi_1 = 0.03
phi_0 = 0.4
r = 0.5
K = 100
pi = 0.004
lambda_ = 0.007
lambda_0 = 0.4
pi_2 = 0.09
gamma_1 = 0.0002
phi = 0.006


B = 20
N = 50

def equations_PE(vars):
    P, E = vars
    eq1 = E - (phi * (L - B)) / (phi_0 + gamma_1 * P)
    eq2 = P - (lambda_ * N) / (lambda_0 + pi_2 * gamma_1 * E)
    return [eq1, eq2]

P, E = fsolve(equations_PE, (1, 1))

eq_B = s * B * (1 - B / L) - alpha_1 * B * N - lambda_2 * B**2 * P + pi_1 * phi_0 * E
eq_N = r * N * (1 - N / K) + pi * alpha_1 * B * N

print(f"Перше рівняння (dB/dt=0): {eq_B:.6f}")
print(f"Друге рівняння (dN/dt=0): {eq_N:.6f}")
print(f"P = {P:.6f}")
print(f"E = {E:.6f}")
