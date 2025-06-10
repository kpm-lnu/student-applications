import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import os
import time
import concurrent.futures
import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  # для відображення прогресу

class Node:
    def __init__(self, node_id: int, r: float, z: float):
        self.node_id = node_id
        # self.node_id = node_idf
        self.r = r
        self.z = z
        
        self.dof_indices = []
        self.displacements = [0.0, 0.0]  # ur, uz

    def __repr__(self):
        return f"Node({self.node_id}, r={self.r:.3f}, z={self.z:.3f})"

class Material:
    def __init__(self, name: str, E: float, nu: float):
        self.name = name
        self.E = E
        self.nu = nu
        self._cached_D = None  # Кеш для матриці пружності

    def __repr__(self):
        return f"Material(name={self.name}, E={self.E}, nu={self.nu})"
    
    def get_elastic_matrix(self) -> np.ndarray:
        if self._cached_D is None:
            E, nu = self.E, self.nu
            factor = E / ((1 + nu) * (1 - 2 * nu))
            self._cached_D = np.array([
                [(1 - nu) * factor, nu * factor, 0, nu * factor],
                [nu * factor, (1 - nu) * factor, 0, nu * factor],
                [0, 0, (0.5 - nu) * factor, 0],
                [nu * factor, nu * factor, 0, (1 - nu) * factor]
            ])
        return self._cached_D


class AxisymmetricQuadrature:
    _cached_points = None

    def __init__(self, n_points: int):
        self.n_points = n_points
        if AxisymmetricQuadrature._cached_points is None:
            gp_val = 1.0 / np.sqrt(3)
            AxisymmetricQuadrature._cached_points = [
                {"xi": -gp_val, "eta": -gp_val, "weight": 1.0},
                {"xi":  gp_val, "eta": -gp_val, "weight": 1.0},
                {"xi":  gp_val, "eta":  gp_val, "weight": 1.0},
                {"xi": -gp_val, "eta":  gp_val, "weight": 1.0},
            ]

    def gauss_points(self):
        return AxisymmetricQuadrature._cached_points
        
def gauss_legendre(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w

class ShapeFunction:
    def __init__(self, nodes_count=4):
        self.nodes_count = nodes_count
        
    def evaluate(self, xi, eta):
        one_minus_xi = 1 - xi
        one_plus_xi = 1 + xi
        one_minus_eta = 1 - eta
        one_plus_eta = 1 + eta

        N = 0.25 * np.array([
            one_minus_xi * one_minus_eta,
            one_plus_xi * one_minus_eta,
            one_plus_xi * one_plus_eta,
            one_minus_xi * one_plus_eta
        ])

        dN_dxi = 0.25 * np.array([
            -one_minus_eta,
            one_minus_eta,
            one_plus_eta,
            -one_plus_eta
        ])

        dN_deta = 0.25 * np.array([
            -one_minus_xi,
            -one_plus_xi,
            one_plus_xi,
            one_minus_xi
        ])

        return N, dN_dxi, dN_deta


class Mesh:
    def __init__(self):
        self.nodes = {}  # Словник для зберігання вузлів
        self.elements = []  # Список для зберігання елементів
        self.node_dof = 2  # Кількість степенів свободи на вузол (r, z)
        
    def add_node(self, node_id: int, r: float, z: float) -> Node:
        node = Node(node_id, r, z)
        self.nodes[node_id] = node
        return node
    
    def add_element(self, element):
        self.elements.append(element)
        return element

class Element:
    def __init__(self, 
                 element_id: int, 
                 node_ids: list, 
                 material: Material,
                 weights: np.ndarray = None):
        self.element_id = element_id
        self.node_ids = node_ids
        self.material = material
        self.shape_func = ShapeFunction()
        self.quadrature = AxisymmetricQuadrature(2)  
        self.weights = weights if weights is not None else np.ones(4) 
        
    def compute_element_stiffness(self, mesh: Mesh) -> np.ndarray:
        D = self.material.get_elastic_matrix()
        element_nodes = [mesh.nodes[nid] for nid in self.node_ids]
        coords = np.array([[node.r, node.z] for node in element_nodes])
        Ke = np.zeros((8, 8))

        for gp in self.quadrature.gauss_points():
            xi, eta, weight = gp["xi"], gp["eta"], gp["weight"]
            N, dN_dxi, dN_deta = self.shape_func.evaluate(xi, eta)
            
            J = np.vstack([dN_dxi, dN_deta]) @ coords
            detJ = np.linalg.det(J)
            if np.abs(detJ) < 1e-10:
                return np.zeros((8, 8))  # Обробка сингулярності
            
            invJ = np.linalg.inv(J)
            dN_real = invJ @ np.vstack([dN_dxi, dN_deta])
            r_current = N @ coords[:, 0]
            
            B = np.zeros((4, 8))
            B[0, 0:4] = dN_real[0, :]  # dN/dr
            B[1, 4:8] = dN_real[1, :]  # dN/dz
            B[2, 0:4] = dN_real[1, :]   # dN/dz (для деформації зсуву)
            B[2, 4:8] = dN_real[0, :]   # dN/dr (для деформації зсуву)
            B[3, 0:4] = N / r_current   # N/r для осьової симетрії
            
            Ke += (B.T @ D @ B) * (weight * detJ)
        
        return Ke

    def compute_reference_stiffness(self, mesh: Mesh, n_points: int) -> np.ndarray:
        class HighOrderQuadrature:
            def gauss_points(self):
                points = []
                xi, w_xi = gauss_legendre(n_points)
                eta, w_eta = gauss_legendre(n_points)
                for i in range(n_points):
                    for j in range(n_points):
                        points.append({
                            "xi": xi[i],
                            "eta": eta[j],
                            "weight": w_xi[i] * w_eta[j]
                        })
                return points

        original_quadrature = self.quadrature
        self.quadrature = HighOrderQuadrature()
        try:
            Ke_ref = self.compute_element_stiffness(mesh)
        except np.linalg.LinAlgError:
            Ke_ref = np.zeros((8, 8))
        finally:
            self.quadrature = original_quadrature
        return Ke_ref

    def alternative_fitness(self, weights: np.ndarray) -> float:
        total_similarity = 0.0
        valid_quads = 0
        
        for quad_idx in range(self.num_quads):
            quad_weights = weights[quad_idx * self.quad_size: (quad_idx + 1) * self.quad_size]
            
            element = self.mesh.elements[quad_idx]
            
            try:
                original_weights = element.weights.copy()
                element.weights = np.ones(4)
                reference_K = self.compute_reference_stiffness(element, 30)
                
                element.weights = quad_weights
                
                Ke = element.compute_element_stiffness(self.mesh)
                
                norm_diff = np.linalg.norm(Ke - reference_K, 'fro')
                norm_ref = np.linalg.norm(reference_K, 'fro')
                
                if norm_ref > 1e-10:
                    relative_diff = norm_diff / norm_ref
                    similarity = 1.0 / (1.0 + relative_diff)  # Ближче до 1 - кращий результат
                else:
                    similarity = 0.0
                
                element.weights = original_weights
                
                total_similarity += similarity
                valid_quads += 1
            except np.linalg.LinAlgError:
                total_similarity += 0.01
                valid_quads += 1
        
        return total_similarity / valid_quads if valid_quads > 0 else 0.01
def map_to_quad(xi, eta, vertices):
    N = np.zeros(4)
    N[0] = 0.25 * (1 - xi) * (1 - eta)
    N[1] = 0.25 * (1 + xi) * (1 - eta)
    N[2] = 0.25 * (1 + xi) * (1 + eta)
    N[3] = 0.25 * (1 - xi) * (1 + eta)
    
    x = sum(N[i] * vertices[i, 0] for i in range(4))
    y = sum(N[i] * vertices[i, 1] for i in range(4))
    
    return x, y

def jacobian_bilinear(xi, eta, vertices):
    dN_dxi = np.zeros(4)
    dN_dxi[0] = -0.25 * (1 - eta)
    dN_dxi[1] = 0.25 * (1 - eta)
    dN_dxi[2] = 0.25 * (1 + eta)
    dN_dxi[3] = -0.25 * (1 + eta)
    
    dN_deta = np.zeros(4)
    dN_deta[0] = -0.25 * (1 - xi)
    dN_deta[1] = -0.25 * (1 + xi)
    dN_deta[2] = 0.25 * (1 + xi)
    dN_deta[3] = 0.25 * (1 - xi)
    
    dx_dxi = sum(dN_dxi[i] * vertices[i, 0] for i in range(4))
    dy_dxi = sum(dN_dxi[i] * vertices[i, 1] for i in range(4))
    dx_deta = sum(dN_deta[i] * vertices[i, 0] for i in range(4))
    dy_deta = sum(dN_deta[i] * vertices[i, 1] for i in range(4))
    
    jacobian = dx_dxi * dy_deta - dx_deta * dy_dxi
    return abs(jacobian)

def calculate_optimal_quad_weights(vertices, n):
    xi, w_xi = gauss_legendre(n)
    eta, w_eta = gauss_legendre(n)
    
    num_points = n * n
    quad_points = np.zeros((num_points, 2))  # Координати точок (x, y)
    quad_weights = np.zeros(num_points)  # Ваги для кожної точки
    
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

# цей клас повинен зараз використовуватися
class SequentialFitnessEvaluator:
    def __init__(self, material, num_quads: int, quad_size: int, coords_per_quad: int):
        self.material = material
        self.num_quads = num_quads
        self.quad_size = quad_size
        self.coords_per_quad = coords_per_quad

    def _unflatten_vertices(self, flat_array: np.ndarray) -> List[np.ndarray]:
        vertices = []
        idx = 0
        for _ in range(self.num_quads):
            quad_coords = flat_array[idx: idx + self.coords_per_quad]
            vertices.append(quad_coords.reshape(self.quad_size, 2))
            idx += self.coords_per_quad
        return vertices

    def evaluate_quad_fitness(self, quad_idx: int, vertices: np.ndarray, weights: np.ndarray) -> float:
        temp_mesh = Mesh()
        for i, v in enumerate(vertices):
            temp_mesh.add_node(i, v[0], v[1])
        element = Element(quad_idx, list(range(len(vertices))), self.material)
        element.weights = weights
        temp_mesh.add_element(element)

        try:
            Ke_ref = element.compute_reference_stiffness(temp_mesh, n_points=30)

            Ke_2 = element.compute_element_stiffness(temp_mesh)

            Ke_current = element.compute_element_stiffness(temp_mesh)

            err_init = np.max(np.abs(Ke_2      - Ke_ref))
            err_curr = np.max(np.abs(Ke_current - Ke_ref))

            fitness = err_init / err_curr if err_curr > 0 else np.inf

            return fitness

        except np.linalg.LinAlgError:
            return np.inf


    def evaluate_batch(self, particles: np.ndarray) -> np.ndarray:
        def fitness_for(particle):
            coords = particle[:self.num_quads * self.coords_per_quad]
            weights = particle[self.num_quads * self.coords_per_quad:]
            vertices_list = self._unflatten_vertices(coords)
            errors = [self.evaluate_quad_fitness(i, v, 
                    weights[i*4:(i+1)*4]) for i, v in enumerate(vertices_list)]
            return np.mean(errors)

        return np.apply_along_axis(fitness_for, 1, particles)


# повинен використовуватися
class PSOQuadWeightOptimizerWithThreshold:
    def __init__(
        self,
        vertices_list: List[np.ndarray],
        material: Material,
        initial_weights: np.ndarray,
        num_particles: int = 300,
        max_iterations: int = 5000,
        alpha: float = 0.5,
        beta: float = 1.5,
        gamma: float = 1.5,
        checkpoint_interval: int = 100,
        fitness_threshold: float = None
    ):
        self.vertices_list = vertices_list
        self.material = material
        self.initial_weights = initial_weights
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.checkpoint_interval = checkpoint_interval
        self.fitness_threshold = fitness_threshold

        self.num_quads = len(vertices_list)
        self.quad_size = 4
        self.coords_per_quad = self.quad_size * 2   # 8
        self.weights_per_quad = 4
        self.particle_dim = self.num_quads * (self.coords_per_quad + self.weights_per_quad)

        flat_coords = np.concatenate([v.flatten() for v in vertices_list])
        coord_size = self.num_quads * self.coords_per_quad

        self.particles = np.empty((num_particles, self.particle_dim))

        coord_mask = identify_fixed_nodes(vertices_list)
        weight_mask = np.zeros(self.num_quads * self.weights_per_quad, dtype=bool)
        self.fixed_mask = np.concatenate([coord_mask, weight_mask])
        self.fixed_coords = flat_coords[coord_mask]
        
        noise = np.random.uniform(-0.1, 0.1, (num_particles, coord_size))
        noise[:, self.fixed_mask[:coord_size]] = 0
        self.particles[:, :coord_size] = flat_coords + noise

        # ваги
        self.particles[:, coord_size:] = (
            np.tile(initial_weights, (num_particles, 1))
            * np.random.uniform(0.9, 1.1, (num_particles, self.num_quads * self.weights_per_quad))
        )
        self.velocities = np.zeros_like(self.particles)

        self.best_positions = self.particles.copy()
        self.best_fitness = np.full(self.num_particles, -np.inf)
        self.global_best_position = None
        self.global_best_fitness = -np.inf

        self.fitness_evaluator = SequentialFitnessEvaluator(
            material, self.num_quads, self.quad_size, self.coords_per_quad
        )

        
    def optimize(self):
        history = []
        print(">>> Початок оптимізації")
        for it in range(self.max_iterations):
            if it % 100 == 0:
                print(f"Ітерація {it}/{self.max_iterations}, поточний глобальний фітнес = {self.global_best_fitness:.6e}")

            fitness_vals = self.fitness_evaluator.evaluate_batch(self.particles)

            improved = fitness_vals > self.best_fitness
            self.best_fitness[improved] = fitness_vals[improved]
            self.best_positions[improved] = self.particles[improved].copy()

            idx = np.argmax(fitness_vals)
            if fitness_vals[idx] > self.global_best_fitness:
                self.global_best_fitness = fitness_vals[idx]
                self.global_best_position = self.particles[idx].copy()

            if self.fitness_threshold is not None and self.global_best_fitness <= self.fitness_threshold:
                print(f"Досягнуто порогове значення {self.fitness_threshold:.6e}× на ітерації {it}")
                break

            #PSO-оновлення
            r1 = np.random.rand(self.num_particles, self.particle_dim)
            r2 = np.random.rand(self.num_particles, self.particle_dim)
            cog = self.beta * r1 * (self.best_positions - self.particles)
            soc = self.gamma * r2 * (self.global_best_position - self.particles)
            self.velocities = self.alpha * self.velocities + cog + soc
            self.particles += self.velocities

            self.particles[:, self.fixed_mask] = self.fixed_coords

            history.append(self.global_best_fitness)

            if it % self.checkpoint_interval == 0:
                with open(f"pso_checkpoint_{it}.pkl", "wb") as f:
                    pickle.dump({
                        "particles": self.particles,
                        "velocities": self.velocities,
                        "best_positions": self.best_positions,
                        "best_fitness": self.best_fitness,
                        "global_best_position": self.global_best_position,
                        "global_best_fitness": self.global_best_fitness,
                        "iteration": it
                    }, f)

        return self.global_best_position, self.global_best_fitness, history
    
class ParallelFitnessEvaluator:
    def __init__(self, material, num_quads, quad_size, coords_per_quad):
        self.material = material
        self.num_quads = num_quads
        self.quad_size = quad_size
        self.coords_per_quad = coords_per_quad

    def evaluate_batch(self, particles_batch):
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._evaluate_particle, particle) for particle in particles_batch]
            return [f.result() for f in futures]

    def _evaluate_particle(self, particle):
        # Логіка обчислення фітнесу для однієї частинки
        return np.mean([self._evaluate_quad(quad_idx, particle) for quad_idx in range(self.num_quads)])

def read_quads_from_file(filename: str) -> List[np.ndarray]:
    vertices_list = []
    
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            coords = []
            parts = line.strip().split(')')
            
            for part in parts:
                part = part.strip()
                if part:
                    part = part.replace('(', '').strip()
                    if part:
                        try:
                            x, y = map(float, part.split(','))
                            coords.append([x, y])
                        except ValueError:
                            print(f"Помилка парсингу координат у рядку: {line}")
            
            if len(coords) == 4:  # Переконуємося, що маємо чотирикутник
                vertices_list.append(np.array(coords))
    
    return vertices_list

def save_weights_to_file(filename: str, weights: np.ndarray, vertices_list: List[np.ndarray]):
    with open(filename, 'w') as file:
        file.write("# Формат: Індекс чотирикутника, Координати вершин, Оптимальні ваги\n")
        
        for quad_idx in range(len(vertices_list)):
            vertices = vertices_list[quad_idx]
            quad_weights = weights[quad_idx * 4: (quad_idx + 1) * 4]
            
            file.write(f"Quad {quad_idx}: ")
            
            for i in range(4):
                file.write(f"({vertices[i][0]:.6f}, {vertices[i][1]:.6f}) ")
            
            file.write("Weights: ")
            
            for i in range(4):
                file.write(f"{quad_weights[i]:.6f} ")
            
            file.write("\n")

def save_optimized_data(save_dir: str, original_vertices: List[np.ndarray], 
                       optimized_coords: np.ndarray, optimized_weights: np.ndarray):
    os.makedirs(save_dir, exist_ok=True) 
    
    with open(os.path.join(save_dir, "optimized_coords.txt"), 'w') as f:
        for quad_idx, vertices in enumerate(original_vertices):
            new_vertices = optimized_coords[quad_idx*8 : (quad_idx+1)*8].reshape(4, 2)
            for v in new_vertices:
                f.write(f"({v[0]:.6f}, {v[1]:.6f}) ")
            f.write("\n")
    
    with open(os.path.join(save_dir, "optimized_weights.txt"), 'w') as f:
        for quad_idx in range(len(original_vertices)):
            weights = optimized_weights[quad_idx*4 : (quad_idx+1)*4]
            f.write(f"Quad {quad_idx}: {weights}\n")

def plot_fitness_history(history: List[float], save_dir: str):
    plt.figure(figsize=(10, 6))
    plt.plot(history)
    plt.title("Еволюція пристосованості")
    plt.xlabel("Ітерація")
    plt.ylabel("Пристосованість")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "fitness_history.png"))
    plt.close()

def visualize_optimized_quads(original_vertices: List[np.ndarray], 
                             optimized_coords: np.ndarray, 
                             optimized_weights: np.ndarray, 
                             save_dir: str):
    plt.figure(figsize=(12, 10))
    cmap = plt.cm.viridis
    
    for quad_idx in range(len(original_vertices)):
        orig_quad = original_vertices[quad_idx]
        plt.plot(orig_quad[:, 0], orig_quad[:, 1], 'r--', alpha=0.3)
        
        opt_quad = optimized_coords[quad_idx*8 : (quad_idx+1)*8].reshape(4, 2)
        opt_quad_closed = np.vstack([opt_quad, opt_quad[0]])
        plt.plot(opt_quad_closed[:, 0], opt_quad_closed[:, 1], 'b-', alpha=0.7)
        
        weights = optimized_weights[quad_idx*4 : (quad_idx+1)*4]
        for i in range(4):
            plt.scatter(opt_quad[i, 0], opt_quad[i, 1], 
                       c=[weights[i]], cmap=cmap, 
                       vmin=0.9, vmax=1.1, s=100, edgecolors='k')
    
    plt.colorbar(label='Вага вершини')
    plt.title('Початкові (червоні) та оптимізовані (сині) чотирикутники')
    plt.xlabel('r координата')
    plt.ylabel('z координата')
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "optimized_quads.png"))
    plt.close()

def identify_fixed_nodes(vertices_list):
    fixed_mask = np.zeros(len(vertices_list) * 8, dtype=bool)
    for quad_idx, vertices in enumerate(vertices_list):
        for vertex_idx, vertex in enumerate(vertices):
            print(f"Quad {quad_idx}, Vertex {vertex_idx}: {vertex}")
            if (
                np.isclose(vertex[0], 1.0, atol=1e-6) and 
                np.isclose(vertex[1], 0.0, atol=1e-6)
            ) or (
                np.isclose(vertex[0], 2.0, atol=1e-6) and 
                np.isclose(vertex[1], 0.0, atol=1e-6)
            ):
                pos_r = quad_idx * 8 + vertex_idx * 2
                pos_z = quad_idx * 8 + vertex_idx * 2 + 1
                fixed_mask[pos_r] = True
                fixed_mask[pos_z] = True
    print(">>> Total fixed positions:", np.where(fixed_mask)[0])
    return fixed_mask

def main():
    import argparse

    parser = argparse.ArgumentParser(description="PSO optimization for all quads in one result set")
    parser.add_argument('--input', type=str, default="combined_shapes_q4.txt")
    parser.add_argument('--particles', type=int, default=100)
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--checkpoint-interval', type=int, default=100)
    parser.add_argument('--fitness-threshold', type=float, default=4)
    parser.add_argument('--processes', type=int, default=11)
    args = parser.parse_args()

    quads = read_quads_from_file(args.input)
    material = Material("Steel", E=210e9, nu=0.3)
    results_dir = "resultsPSO"
    os.makedirs(results_dir, exist_ok=True)

    coord_path = os.path.join(results_dir, "all_optimized_coords.txt")
    weight_path = os.path.join(results_dir, "all_optimized_weights.txt")
    combined_path = os.path.join(results_dir, "all_optimized_combined.txt")

    for path in [coord_path, weight_path, combined_path]:
        with open(path, 'w') as f:
            f.write("# Результати оптимізації чотирикутників\n")

    params = {
        'material': material,
        'particles': args.particles,
        'iterations': args.iterations,
        'checkpoint_interval': args.checkpoint_interval,
        'fitness_threshold': args.fitness_threshold,
        'results_dir': results_dir
    }

    tasks = [(idx, quad, params) for idx, quad in enumerate(quads)]

    chunk_size = max(1, len(tasks) // (args.processes * 2))
    chunks = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]

    with ProcessPoolExecutor(max_workers=args.processes) as executor:
        futures = [executor.submit(process_chunk, chunk, params) for chunk in chunks]
        for future in concurrent.futures.as_completed(futures):
            try:
                for quad_idx, best_fit in future.result():
                    print(f"[✓] Quad {quad_idx} — Best fitness: {best_fit:.6f}")
            except Exception as e:
                print(f"[!] Помилка обробки чотирикутника: {e}")


def optimize_single_quad(args):
    idx, quad_vertices, params = args
    _, gauss_weights = calculate_optimal_quad_weights(quad_vertices, n=2)
    initial_weights = gauss_weights / np.max(gauss_weights)

    optimizer = PSOQuadWeightOptimizerWithThreshold(
        vertices_list=[quad_vertices],
        material=params['material'],
        initial_weights=initial_weights,
        num_particles=params['particles'],
        max_iterations=params['iterations'],
        checkpoint_interval=params['checkpoint_interval'],
        fitness_threshold=params['fitness_threshold']
    )

    best_pos, best_fit, history = optimizer.optimize()

    optimized_coords = best_pos[:8].reshape(4, 2)
    optimized_weights = best_pos[8:12]

    coords_file = os.path.join(params['results_dir'], "all_optimized_coords.txt")
    weights_file = os.path.join(params['results_dir'], "all_optimized_weights.txt")
    combined_file = os.path.join(params['results_dir'], "all_optimized_combined.txt")

    with open(coords_file, 'a') as f:
        f.write(f"Quad {idx}: ")
        for x, y in optimized_coords:
            f.write(f"({x:.6f}, {y:.6f}) ")
        f.write("\n")

    with open(weights_file, 'a') as f:
        f.write(f"Quad {idx}: ")
        for w in optimized_weights:
            f.write(f"{w:.6f} ")
        f.write("\n")

    with open(combined_file, 'a') as f:
        f.write(f"Quad {idx}: ")
        for (x, y), w in zip(optimized_coords, optimized_weights):
            f.write(f"({x:.6f}, {y:.6f}, w={w:.6f}) ")
        f.write("\n")

    return idx, best_fit

def process_chunk(chunk: List[Tuple[int, np.ndarray, dict]], params: dict) -> List[Tuple[int, float]]:
    chunk_results = []
    for task in chunk:
        quad_idx, quad_vertices, task_params = task
        try:
            _, best_fit = optimize_single_quad((quad_idx, quad_vertices, task_params))
            chunk_results.append((quad_idx, best_fit))
        except Exception as e:
            print(f"Помилка в чотирикутнику {quad_idx}: {e}")
    return chunk_results

if __name__ == "__main__":
    main()