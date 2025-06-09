from sympy import symbols, Matrix

λ, ρ, β, δ, p, c  = symbols('λ, ρ, β, δ, p, c')

# A = Matrix([
#     [-r, 0, -b*L/r],
#     [0, -δ, b*L/r],
#     [0, -p, -c]
# ])

matrix = Matrix([
    [-(λ * p * β - 2 * ρ * δ * c) / (δ * c), 0, -δ * c / p],
    [λ * p * β - ρ * δ * c, -δ, δ * c / p],
    [0, p, -c]
])

# B = Matrix([
#     [0, 0, 0],
#     [0, 0, 0],
#     [0, p, 0]
# ])

# A_inv = A.inv()

# print(A)
# print("Inverse of the matrix A:")
# A_inv
# print(A_inv)
# print("FV-1")
# FV1 = (B.multiply(A_inv))
# print(-FV1)
# print("Eigenvalues")
# print((-FV1).eigenvals())
print("Eigenvalues")
print((matrix).eigenvals())