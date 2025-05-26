import system_of_elements as s1d
import system_2d as s2d
import numpy as np
import matplotlib.pyplot as plt

def get_statistics_at_x(b1, d1, b2, d2, degree, f, ug, at_x, K = [1 ,1], element_type = 'D2QU4N', number = 3, u_analytical=None):
    n_mesh = 1
    tuples = []
    errors = []
    for i in range(number):
        n_mesh *= 2
        print(n_mesh)

        u_values, nodes, elements = s2d.get_solution(b1, d1, b2, d2, n_mesh, n_mesh, degree, f, ug, K, element_type)
        indices = extract_indices_at_x(at_x, nodes)
        y_values = extract_nodes_at_indices(nodes, indices)
        u_approx = extract_u_approx_indices(u_values, indices)    #extract u values where x==at_x
        u_exact = extract_u_exact_at_fixed(u_analytical, nodes, indices)
        error = calculate_error(u_exact, u_approx)
        # print('error '+str(n_mesh) + ' = ', error)
        tuples.append([y_values, u_approx])
        errors.append(error)
    return tuples, errors

def extract_indices_at_x(at_x, nodes):
    indices = []
    for i in range(len(nodes)):
        if nodes[i][0]==at_x:
            indices.append(i)
    return indices

def extract_indices_at_y(at_y, nodes):
    indices = []
    for i in range(len(nodes)):
        if nodes[i][1]==at_y:
            indices.append(i)
    return indices

def extract_nodes_at_indices(nodes, indices):
    absc_at_i = []
    for index in indices:
        absc_at_i.append(nodes[index][1])
    return absc_at_i

def extract_u_approx_indices(u_values, indices):
    u_approx = []
    for index in indices:
        u_approx.append(u_values[index])
    return u_approx

def extract_u_exact_at_fixed(u, nodes, indices):
    u_exact = []
    for index in indices:
        u_exact.append(u(nodes[index][0], nodes[index][1]))
    return u_exact


def calculate_error(u_exact, u_approx):
    n = len(u_approx)
    # print('n = ', n)
    sum = 0
    for i in range(n):
        sum+=(u_exact[i]-u_approx[i])**2
    sum /= n
    return sum

def get_statistics_at_y(b1, d1, b2, d2, degree, f, ug, at_y, K = [1, 1], element_type = 'D2QU4N', number = 3, u_analytical=None):
    n_mesh = 1
    tuples = []
    for i in range(number):
        n_mesh *= 2

        u_values, nodes, elements = s2d.get_solution(b1, d1, b2, d2, n_mesh, n_mesh, degree, f, ug, element_type)
        indices = extract_indices_at_y(at_y, nodes)
        x_values = extract_nodes_at_indices(nodes, indices)
        u_approx = extract_u_approx_indices(u_values, indices)    #extract u values where x==at_x
        u_exact = extract_u_exact_at_fixed(u_analytical, nodes, indices)

        tuples.append([x_values, u_approx])
    return tuples


'''
def plot_statistics(b2, d2, fixed_value, statistics, u_exact):
    abscissas = np.linspace(b2, d2, 100)
    u_values = [u_exact(fixed_value, absc) for absc in abscissas]
    plt.plot(abscissas, u_values, color='red', label='u exact')
    for i in range(len(statistics)):
        plt.plot(statistics[i][0], statistics[i][1], label='approx '+str(i))

    plt.xlabel('abscissas')
    plt.ylabel('u(x = at_x, y)')
    plt.title('exact and approx solutions for different number')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_errors(errors):
    plt.plot([i for i in range(len(errors))], errors, color='green')
    plt.yscale('log')
    plt.xlabel('degree of nodes')
    plt.ylabel('errors')
    plt.grid(True)
    plt.show()
'''

def plot_statistics(b2, d2, fixed_value, statistics, u_exact):
    abscissas = np.linspace(b2, d2, 100)
    u_values = [u_exact(fixed_value, absc) for absc in abscissas]
    plt.plot(abscissas, u_values, color='red', label='u exact')
    for i in range(len(statistics)):
        plt.plot(statistics[i][0], statistics[i][1], label='approx '+str(i))

    plt.xlabel('abscissas')
    plt.ylabel('u(x = at_x, y)')
    plt.title('exact and approx solutions for different number')
    plt.legend()
    plt.grid(True)
#    plt.show()

def plot_errors(errors):
    plt.plot([i for i in range(len(errors))], errors, color='green')
    plt.xlabel('degree of nodes')
    plt.ylabel('errors')
    plt.grid(True)
#    plt.show() 
