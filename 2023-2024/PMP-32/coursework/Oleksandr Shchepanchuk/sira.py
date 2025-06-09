from sympy import symbols, Function, Matrix, solve, Eq

beta_1, beta_2, gamma, delta, k, t, omega = symbols('beta_1 beta_2 gamma delta k t omega')
S = Function('S')(t)
I = Function('I')(t)
A = Function('A')(t)
R = Function('R')(t)

def find_disease_free_equilibrium(equilibrium_points, infected_var):

    for eq_point in equilibrium_points:
        if eq_point[infected_var] == 0:
            return dict(zip([S, I, A, R], eq_point))
    return None 

def find_endemic_equilibrium(equilibrium_points, infected_var):
    
    for eq_point in equilibrium_points:
        if eq_point[infected_var] != 0:
            return dict(zip([S, I, A, R], eq_point))
    return None 

def get_Next_Generation_Matrix():
    S = Function('S')(t)
    I = Function('I')(t)
    A = Function('A')(t)
    R = 1-S-I-A
    dS_dt = -beta_1 * S * I - beta_2 * S * A
    dI_dt = -gamma * I + beta_1 *  S * I
    dA_dt = delta * R
    
    system_equations,variables = [dA_dt,dI_dt,dS_dt],[A,I,S]
    
    F = Matrix([[0,
                 beta_1*k*S*I,
                 0]])
    V= Matrix([[
        delta*(1-S-A-I),
        -gamma*I,
        -beta_1*S*I*k-beta_2*S*A*k
    ]])
    
    Vx = V.jacobian(variables).subs({S:1,I:0,A:0})
    Fx = F.jacobian(variables).subs({S:1,I:0,A:0})
    Vx1= Vx.inv()
    FV1 = Fx.multiply(Vx1)
    return FV1



def define_system():

    dS_dt = -beta_1 * S * I*k - beta_2  * S * A*k
    dI_dt = -gamma * I + beta_1 *  S * I *k
    dA_dt = delta * R 
    dR_dt = -delta * R + gamma * I + beta_2 *  S * A *k

    return [dS_dt, dA_dt, dI_dt, dR_dt], [S, A, I, R]

def find_equilibrium(system_equations, variables, additional_eq):
    system_equations_with_constraint = system_equations + [additional_eq]
    return solve(system_equations_with_constraint, variables)

def compute_jacobian(system_equations, variables):
    Jacobian = Matrix(system_equations).jacobian(variables)
    return Jacobian

def jacobian_eigenvalues(Jacobian, equilibrium):
    eigenvals = Jacobian.subs(equilibrium).eigenvals()
    return eigenvals

def main():
    system_equations, variables = define_system()
    additional_eq = Eq(sum(variables), 1)  

    equilibrium_points = find_equilibrium(system_equations, variables, additional_eq)
    print("Equilibrium points:")
    print(equilibrium_points)
    
    print("FV-1")
    print(get_Next_Generation_Matrix())
    
    print("eigenvals of NGM")
    print(((-get_Next_Generation_Matrix()).eigenvals()))    
    

    disease_free_eq = [eq for eq in equilibrium_points if tuple(eq)[2] == 0]
    endemic_eq = [eq for eq in equilibrium_points if eq[2] != 0]

    Jacobian = compute_jacobian(system_equations, variables)
    print("\n\nJacobian")
    print(Jacobian)
    if disease_free_eq:
        print("\nJacobian Matrix at Disease-Free Equilibrium:")
        print(Jacobian.subs(dict(zip(variables, disease_free_eq[0]))))

        print("\nEigenvalues at Disease-Free Equilibrium:")
        print(jacobian_eigenvalues(Jacobian, dict(zip(variables, disease_free_eq[0]))))
        
        print("\nJacobian Matrix at second Disease-Free Equilibrium:")
        print(Jacobian.subs(dict(zip(variables, disease_free_eq[1]))))

        print("\nEigenvalues at second Disease-Free Equilibrium:")
        print(jacobian_eigenvalues(Jacobian, dict(zip(variables, disease_free_eq[1]))))


    
    
    
if __name__ == "__main__":
    main()
