from sympy import symbols, Function, Eq, solve, Matrix

t = symbols('t')
U = Function('U')(t)
I = Function('I')(t)
V = Function('V')(t)
λ, ρ, β, δ, p, c = symbols('λ ρ β δ p c')
U0, I0, V0 = 998, 1, 1 

λ, ρ, β, δ, p, c = 0.1, 0.01, 2.5, 0.15, 15, 5

dU_dt = λ - ρ*U - β*U*V
dI_dt = β*U*V - δ*I
dV_dt = p*I - c*V

R0 = (λ*β*p)/(c*ρ*δ)

equations = [dU_dt, dI_dt, dV_dt]

equilibrium_points = solve(equations, (U, I, V), dict=True)

variables = [U, I, V]
jacobian_matrix = Matrix(equations).jacobian(variables)

eigenvalues = jacobian_matrix.subs(equilibrium_points[0]).eigenvals()

print(equilibrium_points, eigenvalues)
print(R0)