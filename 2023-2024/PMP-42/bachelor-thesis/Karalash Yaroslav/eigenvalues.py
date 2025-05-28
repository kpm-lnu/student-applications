from sympy import symbols, Matrix, solve

# Задаємо символьні змінні
I_h, R_h, I_r = symbols('I_h R_h I_r')

# Задаємо параметри
gamma_h, lambda_hI, r1, lambda_hS, r2, mu_rI, lambda_rI, gamma_r = symbols('gamma_h lambda_hI r1 lambda_hS r2 mu_rI lambda_rI gamma_r')

# Задаємо точки рівноваги
point_1 = (0, 0, 0)
point_2 = (gamma_h*(lambda_hS + r2)*(gamma_r - lambda_rI + mu_rI)/(gamma_h*gamma_r*lambda_hS + gamma_h*gamma_r*r1 + gamma_h*gamma_r*r2 - gamma_h*lambda_hS*lambda_rI + gamma_h*lambda_hS*mu_rI - gamma_h*lambda_rI*r1 - gamma_h*lambda_rI*r2 + gamma_h*mu_rI*r1 + gamma_h*mu_rI*r2 + gamma_r*lambda_hI*lambda_hS + gamma_r*lambda_hI*r2 + gamma_r*lambda_hS*r1 + gamma_r*r1*r2), 
           gamma_h*r1*(gamma_r - lambda_rI + mu_rI)/(gamma_h*gamma_r*lambda_hS + gamma_h*gamma_r*r1 + gamma_h*gamma_r*r2 - gamma_h*lambda_hS*lambda_rI + gamma_h*lambda_hS*mu_rI - gamma_h*lambda_rI*r1 - gamma_h*lambda_rI*r2 + gamma_h*mu_rI*r1 + gamma_h*mu_rI*r2 + gamma_r*lambda_hI*lambda_hS + gamma_r*lambda_hI*r2 + gamma_r*lambda_hS*r1 + gamma_r*r1*r2), 
           (gamma_r - lambda_rI + mu_rI)/gamma_r)

# Задаємо систему рівнянь
equations = [
    gamma_h * I_r * (1 - I_h - R_h) - I_h * (lambda_hI + r1),
    r1 * I_h - R_h * (lambda_hS + r2),
    mu_rI * I_r - lambda_rI * I_r + gamma_r * I_r * (1 - I_r)
]

# Побудова матриці Якобі
J = Matrix([[equ.diff(var) for var in [I_h, R_h, I_r]] for equ in equations])
print(J)
# Підстановка точок рівноваги у матрицю Якобі
J_subs_1 = J.subs({I_h: point_1[0], R_h: point_1[1], I_r: point_1[2]})
J_subs_2 = J.subs({I_h: point_2[0], R_h: point_2[1], I_r: point_2[2]})

# Обчислення власних значень
eigenvalues_1 = solve(J_subs_1.det())
eigenvalues_2 = solve(J_subs_2.det())

print("Власні значення у точці рівноваги (0, 0, 0):", eigenvalues_1)
print("Власні значення у точці рівноваги:", eigenvalues_2)
