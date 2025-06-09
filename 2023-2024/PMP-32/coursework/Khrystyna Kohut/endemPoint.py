from sympy import symbols, Matrix

λ, ρ, β, δ, p, c = symbols('λ ρ β δ p c')

matrix = Matrix([
    [-(λ * p * β - 2 * ρ * δ * c) / (δ * c), 0, -δ * c / p],
    [λ * p * β - ρ * δ * c, -δ, δ * c / p],
    [0, p, -c]
])

eigenvalues = matrix.eigenvals()

print("Eigenvalues (symbolic):")
for eigenvalue, multiplicity in eigenvalues.items():
    print(f"{eigenvalue}: multiplicity {multiplicity}")

params = {λ: 0.1, ρ: 0.01, β: 2.5, δ: 0.15, p: 15, c: 5}

numeric_eigenvalues = [eigenvalue.evalf(subs=params) for eigenvalue in eigenvalues]

print("\nEigenvalues (numeric):")
for eigenvalue in numeric_eigenvalues:
    print(eigenvalue)
