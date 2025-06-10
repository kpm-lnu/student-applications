import numpy as np
import matplotlib.pyplot as plt

def gauss_legendre(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w

def map_to_quad(xi, eta, vertices):
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]
    x4, y4 = vertices[3]
    
    N1 = 0.25 * (1 - xi) * (1 - eta)
    N2 = 0.25 * (1 + xi) * (1 - eta)
    N3 = 0.25 * (1 + xi) * (1 + eta)
    N4 = 0.25 * (1 - xi) * (1 + eta)
    
    x = N1 * x1 + N2 * x2 + N3 * x3 + N4 * x4
    y = N1 * y1 + N2 * y2 + N3 * y3 + N4 * y4
    
    return x, y

def jacobian_bilinear(xi, eta, vertices):
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]
    x4, y4 = vertices[3]
    
    dx_dxi = 0.25 * (-(1-eta)*x1 + (1-eta)*x2 + (1+eta)*x3 - (1+eta)*x4)
    dx_deta = 0.25 * (-(1-xi)*x1 - (1+xi)*x2 + (1+xi)*x3 + (1-xi)*x4)
    dy_dxi = 0.25 * (-(1-eta)*y1 + (1-eta)*y2 + (1+eta)*y3 - (1+eta)*y4)
    dy_deta = 0.25 * (-(1-xi)*y1 - (1+xi)*y2 + (1+xi)*y3 + (1-xi)*y4)
    
    det_j = dx_dxi * dy_deta - dx_deta * dy_dxi
    
    return abs(det_j)

def calculate_optimal_quad_weights(vertices, n):

    xi, w_xi = gauss_legendre(n)
    eta, w_eta = gauss_legendre(n)
    
    num_points = n * n
    quad_points = np.zeros((num_points, 2))  # Координати точок (x, y)
    quad_weights = np.zeros(num_points)      # Ваги для кожної точки
    
    point_idx = 0
    for i in range(n):
        for j in range(n):
            xi_val = xi[i]
            eta_val = eta[j]
            
            x, y = map_to_quad(xi_val, eta_val, vertices)
            
            jac = jacobian_bilinear(xi_val, eta_val, vertices)
            
            quad_points[point_idx] = [x, y]
            quad_weights[point_idx] = w_xi[i] * w_eta[j] * jac
            
            point_idx += 1
    
    return quad_points, quad_weights

def visualize_quad_points(vertices, quad_points, quad_weights):
    plt.figure(figsize=(10, 8))
    
    vertices_closed = np.vstack((vertices, vertices[0]))
    plt.plot(vertices_closed[:, 0], vertices_closed[:, 1], 'k-', linewidth=2)
    
    max_weight = max(quad_weights)
    for i, (x, y) in enumerate(quad_points):
        size = 50 * (quad_weights[i] / max_weight)  # Розмір точки пропорційний до ваги
        plt.scatter(x, y, s=size, c='r', alpha=0.7)
        plt.text(x+0.05, y+0.05, f"{quad_weights[i]:.4f}", fontsize=8)
    
    for i, (x, y) in enumerate(vertices):
        plt.text(x, y, f"P{i+1}({x:.2f}, {y:.2f})", fontsize=10, weight='bold')
    
    plt.axis('equal')
    plt.grid(True)
    plt.title('Точки інтегрування та їхні ваги на чотирикутнику')
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.tight_layout()
    plt.show()

def save_optimal_weights(vertices, n, filename):
    quad_points, quad_weights = calculate_optimal_quad_weights(vertices, n)
    
    with open(filename, 'w') as f:
        f.write(f"# Оптимальні ваги для чотирикутника (n={n})\n")
        f.write("# Вершини чотирикутника:\n")
        for i, (x, y) in enumerate(vertices):
            f.write(f"# P{i+1}: ({x:.6f}, {y:.6f})\n")
        
        f.write("\n# Формат: x y вага\n")
        for i in range(len(quad_points)):
            f.write(f"{quad_points[i,0]:.10f} {quad_points[i,1]:.10f} {quad_weights[i]:.10f}\n")
    
    print(f"Оптимальні ваги збережено у файл: {filename}")

if __name__ == "__main__":
    vertices = np.array([
        [0.00, 0.00],  # (x1, y1)
        [1.00, 0.00],  # (x2, y2)
        [2.70, 0.43],  # (x3, y3)
        [0.01, 1.04]   # (x4, y4)
    ])
    
    for n in [2, 3, 4, 5]:
        print(f"\nОптимальні ваги для n={n}:")
        quad_points, quad_weights = calculate_optimal_quad_weights(vertices, n)
        
        print("Точки інтегрування:")
        for i in range(len(quad_points)):
            print(f"Точка {i+1}: ({quad_points[i,0]:.6f}, {quad_points[i,1]:.6f}), вага: {quad_weights[i]:.6f}")
        
        print(f"Сума ваг: {np.sum(quad_weights):.6f}")
        
        if n == 3:
            visualize_quad_points(vertices, quad_points, quad_weights)
            
            save_optimal_weights(vertices, n, f"optimal_weights_n{n}.txt")
    
    def test_function(x, y):
        return x * y
    
    print("\nІнтеграл x*y з використанням оптимальних ваг:")
    for n in [2, 3, 4, 5]:
        quad_points, quad_weights = calculate_optimal_quad_weights(vertices, n)
        integral = 0
        for i in range(len(quad_points)):
            x, y = quad_points[i]
            integral += test_function(x, y) * quad_weights[i]
        print(f"n={n}: {integral:.6f}")