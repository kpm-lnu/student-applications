from sympy import symbols, Eq, solve, simplify, Matrix, diff

I_h, R_h, I_r, E = symbols('I_h R_h I_r E', real=True, nonnegative=True)

mu_h, lambda_hS, gamma_h, gamma_e, r2, lambda_hI, r1, mu_rI, lambda_r, gamma_r, alpha, beta = symbols(
    'mu_h lambda_hS gamma_h gamma_e r2 lambda_hI r1 mu_rI lambda_r gamma_r alpha beta',
    real=True, positive=True
)

f1 = (gamma_h * I_r + gamma_e * E) * (1 - I_h - R_h) - (lambda_hI + r1) * I_h
f2 = r1 * I_h - (lambda_hS + r2) * R_h
f3 = (gamma_r * (1 - I_r) + mu_rI - lambda_r) * I_r
f4 = alpha * I_r - beta * E

eqs = [Eq(f1, 0), Eq(f2, 0), Eq(f3, 0), Eq(f4, 0)]

solutions = solve(eqs, [I_h, R_h, I_r, E], dict=True)

vars_ = [I_h, R_h, I_r, E]
f = [f1, f2, f3, f4]
J = Matrix([[diff(fi, vj) for vj in vars_] for fi in f])

for i, sol in enumerate(solutions, 1):
    print(f"\nСтаціонарна точка E_{i}:")
    simplified_sol = {k: simplify(v) for k, v in sol.items()}
    
    for var, expr in simplified_sol.items():
        print(f"{var} = {expr}")

    J_subs = J.subs(simplified_sol)

    eigenvals = J_subs.eigenvals()
    
    print("\nВласні значення:")
    for ev, mult in eigenvals.items():
        print(f"{ev} (кратність: {mult})")

    print("\n" + "-"*60)
