from material import Material
from shapeFunction import Quadratic8ShapeFunction
from axisymmetricQuadrature import AxisymmetricQuadrature
from mesh import Mesh
from boundaryConditions import DirichletBC, NeumannBC
from axisymmetricFEMSolver import AxisymmetricFEMSolver
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
from scipy.interpolate import griddata
import torch
import torch.nn as nn
from quad_predictor import QuadPredictor
import copy
import pickle


def to_local_coords(p0, p1, p):
    dx, dy = p1 - p0
    L = np.hypot(dx, dy)
    if L < 1e-8:
        raise ValueError("Нульова довжина сторони для нормалізації")

    cos_theta = dx / L
    sin_theta = dy / L

    R = np.array([[cos_theta, sin_theta],
                  [-sin_theta, cos_theta]])
    return R @ (p - p0) / L

def main():
    # Параметри задачі
    r_min, r_max = 1.0, 2.0
    z_min, z_max = 0.0, 32.0
    nu = 0.3
    mu = 0.7
    p = 1
    E = 2 * mu * (1 + nu)
    rN, zN = 10, 10
    quadrature_points = 3
    node_dof = 2
    compare_eps = 1e-6
    fixed_z = 32.0

    # Матеріал і сітка
    mat = Material("Steel", E=E, nu=nu)
    shape_func = Quadratic8ShapeFunction()
    quad_rule = AxisymmetricQuadrature(n_points=quadrature_points)
    
    mesh = Mesh(mat, shape_func, quad_rule, node_dof)
    mesh.generate_skewed_cylinder(r_min, r_max, z_min, z_max, 40, rN, zN)
    mesh.visualize_mesh()

    # Стандартна копія сітки
    mesh_standard = copy.deepcopy(mesh)

    # Завантаження моделі
    model = QuadPredictor()
    model.load_state_dict(torch.load("quad_predictor_minmax.pth"))
    model.eval()

    with open("scalers_minmax.pkl", "rb") as f: 
        scalers = pickle.load(f)
        X_scaler = scalers['X_scaler']

    # Перебір елементів
    for elem in mesh.elements.values():
        node_coords = np.array([[mesh.nodes[nid].r, mesh.nodes[nid].z] for nid in elem.node_ids])

        # Локальне перетворення координат 3-ї та 4-ї точки
        p0 = node_coords[0]
        p1 = node_coords[1]
        p2 = node_coords[2]
        p3 = node_coords[3]

        p2_local = to_local_coords(p0, p1, p2)
        p3_local = to_local_coords(p0, p1, p3)

        input_nn = torch.tensor([
            p2_local[0], p2_local[1],
            p3_local[0], p3_local[1]
        ], dtype=torch.float32).reshape(1, -1)  # reshape до (1, 4)

        input_scaled = X_scaler.transform(input_nn)
        input_nn = torch.tensor(input_scaled, dtype=torch.float32)

        with torch.no_grad():
            output = model(input_nn.unsqueeze(0)).squeeze().numpy()

        custom_points = np.array([
            [output[0], output[1]],
            [output[2], output[3]],
            [output[4], output[5]],
            [output[6], output[7]]
        ])
        custom_weights = output[8:12]

        elem.quadrature = AxisymmetricQuadrature(
            custom_points=custom_points,
            custom_weights=custom_weights
        )


    # Граничні умови (однакові для обох сіток)
    left_nodes = [nid for nid, node in mesh.nodes.items() if abs(node.r - r_min) < compare_eps]
    right_nodes = [nid for nid, node in mesh.nodes.items() if abs(node.r - r_max) < compare_eps]
    bottom_nodes = [nid for nid, node in mesh.nodes.items() if abs(node.z - z_min) < compare_eps]
    top_nodes = [nid for nid, node in mesh.nodes.items() if abs(node.z - z_max) < compare_eps]

    bcs = [
        NeumannBC(left_nodes, 0, p),
        NeumannBC(right_nodes, 0, 0),
        *[DirichletBC(nid, 1, 0.0) for nid in bottom_nodes],
        *[DirichletBC(nid, 1, 0.0) for nid in top_nodes]
    ]

    # Розв'язання для обох сіток
    solver_standard = AxisymmetricFEMSolver(mesh_standard, bcs)
    solver_standard.run()
    
    solver_custom = AxisymmetricFEMSolver(mesh, bcs)
    solver_custom.run()
    
    print("================Standart================")
    for node_id, node in mesh_standard.nodes.items():
        print(f"Node {node_id}: ur = {node.displacements[0]:.6f}, "
              f"uz = {node.displacements[1]:.6f}")  
    
    plot_u_sigma(mesh_standard, r_min, r_max, p, mu, nu, fixed_z, mat, compare_eps, shape_func)
    print("===================NN===================")
    for node_id, node in mesh.nodes.items():
        print(f"Node {node_id}: ur = {node.displacements[0]:.6f}, "
              f"uz = {node.displacements[1]:.6f}")  
    plot_u_sigma(mesh, r_min, r_max, p, mu, nu, fixed_z, mat, compare_eps, shape_func)
    
    # Візуалізація результатів
    plot_comparison(
        mesh_standard, mesh, 
        r_min, r_max, p, mu, nu, 
        fixed_z, mat, compare_eps, shape_func
    )


def analytical_solution(r, a, b, p, mu, nu):
    ur = (1 / (2 * mu * (b**2 - a**2))) * ((1 - 2 * nu) * a**2 * p * r + p * a**2 * b**2 / r)
    uz = 0
    sigma_rr = (1 / (b**2 - a**2)) * (a**2 * p - a**2 * b**2 / r**2)
    sigma_zz = (2 * nu * a**2 * p) / (b**2 - a**2)
    sigma_rz = 0
    sigma_phi_phi = (1 / (b**2 - a**2)) * (a**2 * p + a**2 * b**2 / r**2)

    return ur, uz, sigma_rr, sigma_zz, sigma_phi_phi, sigma_rz

def plot_comparison(mesh_standard, mesh_custom, r_min, r_max, p, mu, nu, fixed_z, material, compare_eps, shape_func):
    # Аналітичні розв'язки
    ur_analytical, uz_analytical = [], []
    sigma_rr_analytical, sigma_zz_analytical = [], []
    sigma_rz_analytical, sigma_phi_phi_analytical = [], []
    
    r_values = []
    for node in mesh_standard.nodes.values():
        r = node.r
        z = node.z
        ur, uz, sigma_rr, sigma_zz, sigma_phi_phi, sigma_rz = analytical_solution(r, r_min, r_max, p, mu, nu)
        if abs(z - fixed_z) < compare_eps:
            ur_analytical.append(ur)
            uz_analytical.append(uz)
            sigma_rr_analytical.append(sigma_rr)
            sigma_zz_analytical.append(sigma_zz)
            sigma_rz_analytical.append(sigma_rz)
            sigma_phi_phi_analytical.append(sigma_phi_phi)
            r_values.append(r)

    # Дані для стандартних ваг
    nodal_stresses_standard = recover_nodal_stresses(mesh_standard, material, shape_func, fixed_z)
    nodal_stresses_custom = recover_nodal_stresses(mesh_custom, material, shape_func, fixed_z)

    def collect_stresses(nodal_stresses, mesh):
        r, sigma_rr, sigma_zz, sigma_rz, sigma_phi_phi = [], [], [], [], []  # Змінено sigma_phi -> sigma_phi_phi
        for nid, node in mesh.nodes.items():
            if abs(node.z - fixed_z) < compare_eps and nid in nodal_stresses:
                r.append(node.r)
                sigma_rr.append(nodal_stresses[nid]['sigma_rr'])
                sigma_zz.append(nodal_stresses[nid]['sigma_zz'])
                sigma_rz.append(nodal_stresses[nid]['sigma_rz'])
                sigma_phi_phi.append(nodal_stresses[nid]['sigma_tt'])  # Змінено sigma_phi -> sigma_phi_phi
        return r, sigma_rr, sigma_zz, sigma_rz, sigma_phi_phi  # Змінено sigma_phi -> sigma_phi_phi


    r_std, sigma_rr_std, sigma_zz_std, sigma_rz_std, sigma_phi_phi_std = collect_stresses(nodal_stresses_standard, mesh_standard)
    r_cus, sigma_rr_cus, sigma_zz_cus, sigma_rz_cus, sigma_phi_phi_cus = collect_stresses(nodal_stresses_custom, mesh_custom)

    plt.figure(figsize=(20, 12))

    # Оновлення списку components для відображення sigma_phi_phi
    components = [
        ('sigma_rr', r'$\sigma_{rr}$'), 
        ('sigma_zz', r'$\sigma_{zz}$'), 
        ('sigma_rz', r'$\sigma_{rz}$'), 
        ('sigma_phi_phi', r'$\sigma_{\phi\phi}$')  # Змінено з sigma_phi_phi
    ]
    
    for i, (comp, label) in enumerate(components):
        plt.subplot(2, 2, i+1)
        plt.plot(r_std, locals()[f"{comp}_std"], 'go--', label='FEM', markersize=5)
        plt.plot(r_cus, locals()[f"{comp}_cus"], 'bo--', label='NN', markersize=5)
        plt.plot(r_values, locals()[f"{comp}_analytical"], 'r-', label='Analytical')
        plt.xlabel('Radius (r)')
        plt.ylabel(label)
        plt.title(label)
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    plt.figure(figsize=(20, 12))
    
    def compute_abs_error(numerical, analytical):
        return np.abs(np.array(numerical) - np.array(analytical))
    
    for i, (comp, label) in enumerate(components):
        plt.subplot(2, 2, i+1)
        err_std = compute_abs_error(locals()[f"{comp}_std"], locals()[f"{comp}_analytical"])
        err_cus = compute_abs_error(locals()[f"{comp}_cus"], locals()[f"{comp}_analytical"])
        plt.plot(r_std, err_std, 'g--', label='FEM Error')
        plt.plot(r_cus, err_cus, 'b--', label='NN Error')
        plt.xlabel('Radius (r)')
        plt.ylabel(f'Absolute Error in {label}')
        plt.title(f'Error in {label}')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def get_sigma(mesh, material, shape_func_unused=None):
    num_nodes = len(mesh.nodes)
    nodes_coords = np.zeros((num_nodes, 2))
    u_array = np.zeros(2 * num_nodes)
    for node_id, node in mesh.nodes.items():
        nodes_coords[node_id, 0] = node.r
        nodes_coords[node_id, 1] = node.z
        u_array[node_id] = node.displacements[0]
        u_array[node_id + num_nodes] = node.displacements[1]

    D_mat = material.get_elastic_matrix()

    stress_components = {'sigma_rr': [], 'sigma_zz': [], 'sigma_rz': [], 'sigma_tt': []}
    gp_coords = {'r': [], 'z': []}

    for elem in mesh.elements.values():
        node_ids = elem.node_ids
        n_el_nodes = len(node_ids)
        el_coords = nodes_coords[node_ids, :]

        u_e = np.zeros(2 * n_el_nodes)
        for a, node_id in enumerate(node_ids):
            u_e[a] = u_array[node_id]
            u_e[n_el_nodes + a] = u_array[node_id + num_nodes]

        quad_points = elem.quadrature.gauss_points()
        shape_func = elem.shape_func

        for gp in quad_points:
            # xi = gp["xi"]
            # eta = gp["eta"]
            xi = gp.xi
            eta = gp.eta
            N, dN_dxi, dN_deta = shape_func.evaluate(xi, eta)
            
            J = np.zeros((2, 2))
            for i in range(len(el_coords)):
                J[0, 0] += dN_dxi[i] * el_coords[i, 0]
                J[0, 1] += dN_dxi[i] * el_coords[i, 1]
                J[1, 0] += dN_deta[i] * el_coords[i, 0]
                J[1, 1] += dN_deta[i] * el_coords[i, 1]
            detJ = np.linalg.det(J)
            invJ = np.linalg.inv(J)

            dN_dr = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
            dN_dz = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta

            r_gp = np.dot(N, el_coords[:, 0])

            B = np.zeros((4, 2 * n_el_nodes))
            for a in range(n_el_nodes):
                B[0, a] = dN_dr[a]
                B[1, n_el_nodes + a] = dN_dz[a]
                B[2, a] = dN_dz[a]
                B[2, n_el_nodes + a] = dN_dr[a]
                B[3, a] = N[a] / r_gp

            epsilon_gp = B @ u_e

            sigma_gp = D_mat @ epsilon_gp

            stress_components['sigma_rr'].append(sigma_gp[0])
            stress_components['sigma_zz'].append(sigma_gp[1])
            stress_components['sigma_rz'].append(sigma_gp[2])
            stress_components['sigma_tt'].append(sigma_gp[3])
 
            gp_r = np.dot(N, el_coords[:, 0])
            gp_z = np.dot(N, el_coords[:, 1])
            gp_coords['r'].append(gp_r)
            gp_coords['z'].append(gp_z)

    return stress_components, gp_coords

def recover_nodal_stresses(mesh, material, shape_func, fixed_z, tol=1e-6):
    stress_components, gp_coords = get_sigma(mesh, material, shape_func)

    elem_gp_stresses = {}
    for elem in mesh.elements.values():
        elem_gp_stresses[elem.elem_id] = []

        node_ids = elem.node_ids
        n_el_nodes = len(node_ids)

        nodes_coords = np.zeros((n_el_nodes, 2))
        u_e = np.zeros(2 * n_el_nodes)
        num_nodes_total = len(mesh.nodes)
        for a, nid in enumerate(node_ids):
            nodes_coords[a, 0] = mesh.nodes[nid].r
            nodes_coords[a, 1] = mesh.nodes[nid].z
            u_e[a] = mesh.nodes[nid].displacements[0]
            u_e[n_el_nodes + a] = mesh.nodes[nid].displacements[1]
        for gp in elem.quadrature.gauss_points():
            # xi = gp["xi"]
            # eta = gp["eta"]
            xi = gp.xi
            eta = gp.eta
            N, dN_dxi, dN_deta = elem.shape_func.evaluate(xi, eta)
            J = np.zeros((2, 2))
            for i in range(len(nodes_coords)):
                J[0, 0] += dN_dxi[i] * nodes_coords[i, 0]
                J[0, 1] += dN_dxi[i] * nodes_coords[i, 1]
                J[1, 0] += dN_deta[i] * nodes_coords[i, 0]
                J[1, 1] += dN_deta[i] * nodes_coords[i, 1]
            invJ = np.linalg.inv(J)
            dN_dr = invJ[0, 0]*dN_dxi + invJ[0, 1]*dN_deta
            dN_dz = invJ[1, 0]*dN_dxi + invJ[1, 1]*dN_deta
            r_gp = np.dot(N, nodes_coords[:, 0])
            B = np.zeros((4, 2*n_el_nodes))
            for a in range(n_el_nodes):
                B[0, a] = dN_dr[a]
                B[1, n_el_nodes + a] = dN_dz[a]
                B[2, a] = dN_dz[a]
                B[2, n_el_nodes + a] = dN_dr[a]
                B[3, a] = N[a] / r_gp
            epsilon_gp = B @ u_e
            sigma_gp = material.get_elastic_matrix() @ epsilon_gp

            elem_gp_stresses[elem.elem_id].append({
                'sigma_rr': sigma_gp[0],
                'sigma_zz': sigma_gp[1],
                'sigma_rz': sigma_gp[2],
                'sigma_tt': sigma_gp[3],
                'r_gp': np.dot(N, nodes_coords[:, 0]),
                'z_gp': np.dot(N, nodes_coords[:, 1])
            })
    
    nodal_stresses = {}
    for node_id, node in mesh.nodes.items():
        if abs(node.z - fixed_z) < tol:
            collected = {'sigma_rr': [], 'sigma_zz': [], 'sigma_rz': [], 'sigma_tt': []}
            for elem in mesh.elements.values():
                if node_id in elem.node_ids:
                    for gp_stress in elem_gp_stresses[elem.elem_id]:
                        collected['sigma_rr'].append(gp_stress['sigma_rr'])
                        collected['sigma_zz'].append(gp_stress['sigma_zz'])
                        collected['sigma_rz'].append(gp_stress['sigma_rz'])
                        collected['sigma_tt'].append(gp_stress['sigma_tt'])

            if collected['sigma_rr']:
                nodal_stresses[node_id] = {
                    'sigma_rr': np.mean(collected['sigma_rr']),
                    'sigma_zz': np.mean(collected['sigma_zz']),
                    'sigma_rz': np.mean(collected['sigma_rz']),
                    'sigma_tt': np.mean(collected['sigma_tt'])
                }
    return nodal_stresses

def recover_all_nodal_stresses(mesh, material, shape_func, tol=1e-6):
    nodal_stresses = {}
    for node_id, node in mesh.nodes.items():
        collected = {'sigma_rr': [], 'sigma_zz': [], 'sigma_rz': [], 'sigma_tt': []}
        for elem in mesh.elements.values():
            if node_id in elem.node_ids:
                n_el_nodes = len(elem.node_ids)
                nodes_coords = np.zeros((n_el_nodes, 2))
                u_e = np.zeros(2 * n_el_nodes)
                num_total = len(mesh.nodes)
                for a, nid in enumerate(elem.node_ids):
                    nodes_coords[a, 0] = mesh.nodes[nid].r
                    nodes_coords[a, 1] = mesh.nodes[nid].z
                    u_e[a] = mesh.nodes[nid].displacements[0]
                    u_e[n_el_nodes + a] = mesh.nodes[nid].displacements[1]
                for gp in elem.quadrature.gauss_points():
                    # xi = gp["xi"]
                    # eta = gp["eta"]
                    xi = gp.xi
                    eta = gp.eta
                    N, dN_dxi, dN_deta = elem.shape_func.evaluate(xi, eta)
                    J = np.zeros((mesh.node_dof, mesh.node_dof))
                    for i in range(len(nodes_coords)):
                        J[0, 0] += dN_dxi[i] * nodes_coords[i, 0]
                        J[0, 1] += dN_dxi[i] * nodes_coords[i, 1]
                        J[1, 0] += dN_deta[i] * nodes_coords[i, 0]
                        J[1, 1] += dN_deta[i] * nodes_coords[i, 1]
                    invJ = np.linalg.inv(J)
                    dN_dr = invJ[0, 0]*dN_dxi + invJ[0, 1]*dN_deta
                    dN_dz = invJ[1, 0]*dN_dxi + invJ[1, 1]*dN_deta
                    r_gp = np.dot(N, nodes_coords[:, 0])
                    B = np.zeros((4, 2*n_el_nodes))
                    for a in range(n_el_nodes):
                        B[0, a] = dN_dr[a]
                        B[1, n_el_nodes + a] = dN_dz[a]
                        B[2, a] = dN_dz[a]
                        B[2, n_el_nodes + a] = dN_dr[a]
                        B[3, a] = N[a] / r_gp
                    epsilon_gp = B @ u_e
                    sigma_gp = material.get_elastic_matrix() @ epsilon_gp
                    collected['sigma_rr'].append(sigma_gp[0])
                    collected['sigma_zz'].append(sigma_gp[1])
                    collected['sigma_rz'].append(sigma_gp[2])
                    collected['sigma_tt'].append(sigma_gp[3])
        if collected['sigma_rr']:
            nodal_stresses[node_id] = {
                'sigma_rr': np.mean(collected['sigma_rr']),
                'sigma_zz': np.mean(collected['sigma_zz']),
                'sigma_rz': np.mean(collected['sigma_rz']),
                'sigma_tt': np.mean(collected['sigma_tt'])
            }
    return nodal_stresses

def compute_errors(
    r_values,
    sigma_rr, sigma_zz, sigma_rz, sigma_phi_phi,
    sigma_rr_analytical, sigma_zz_analytical, sigma_rz_analytical, sigma_phi_phi_analytical,
    ur_values, uz_values, ur_analytical, uz_analytical,
    fixed_z
):
    r_values = np.array(r_values)
    sigma_rr = np.array(sigma_rr)
    sigma_zz = np.array(sigma_zz)
    sigma_rz = np.array(sigma_rz)
    sigma_phi_phi = np.array(sigma_phi_phi)

    sigma_rr_analytical = np.array(sigma_rr_analytical)
    sigma_zz_analytical = np.array(sigma_zz_analytical)
    sigma_rz_analytical = np.array(sigma_rz_analytical)
    sigma_phi_phi_analytical = np.array(sigma_phi_phi_analytical)

    ur_values = np.array(ur_values)
    uz_values = np.array(uz_values)
    ur_analytical = np.array(ur_analytical)
    uz_analytical = np.array(uz_analytical)

    abs_error_ur = np.abs(ur_values - ur_analytical)
    abs_error_uz = np.abs(uz_values - uz_analytical)
    abs_error_sigma_rr = np.abs(sigma_rr - sigma_rr_analytical)
    abs_error_sigma_zz = np.abs(sigma_zz - sigma_zz_analytical)
    abs_error_sigma_rz = np.abs(sigma_rz - sigma_rz_analytical)
    abs_error_sigma_phi_phi = np.abs(sigma_phi_phi - sigma_phi_phi_analytical)

    tol = 1e-12

    with np.errstate(divide='ignore', invalid='ignore'):
        rel_error_ur = np.divide(
            abs_error_ur, np.abs(ur_analytical), out=np.zeros_like(abs_error_ur),
            where=np.abs(ur_analytical) > tol
        )
        rel_error_uz = np.divide(
            abs_error_uz, np.abs(uz_analytical), out=np.zeros_like(abs_error_uz),
            where=np.abs(uz_analytical) > tol
        )
        rel_error_sigma_rr = np.divide(
            abs_error_sigma_rr, np.abs(sigma_rr_analytical), out=np.zeros_like(abs_error_sigma_rr),
            where=np.abs(sigma_rr_analytical) > tol
        )
        rel_error_sigma_zz = np.divide(
            abs_error_sigma_zz, np.abs(sigma_zz_analytical), out=np.zeros_like(abs_error_sigma_zz),
            where=np.abs(sigma_zz_analytical) > tol
        )
        rel_error_sigma_rz = np.divide(
            abs_error_sigma_rz, np.abs(sigma_rz_analytical), out=np.zeros_like(abs_error_sigma_rz),
            where=np.abs(sigma_rz_analytical) > tol
        )
        rel_error_sigma_phi_phi = np.divide(
            abs_error_sigma_phi_phi, np.abs(sigma_phi_phi_analytical), out=np.zeros_like(abs_error_sigma_phi_phi),
            where=np.abs(sigma_phi_phi_analytical) > tol
        )

    errors = {
        'absolute_errors': {
            'ur': abs_error_ur,
            'uz': abs_error_uz,
            'sigma_rr': abs_error_sigma_rr,
            'sigma_zz': abs_error_sigma_zz,
            'sigma_rz': abs_error_sigma_rz,
            'sigma_phi_phi': abs_error_sigma_phi_phi,
        },
        'relative_errors': {
            'ur': rel_error_ur,
            'uz': rel_error_uz,
            'sigma_rr': rel_error_sigma_rr,
            'sigma_zz': rel_error_sigma_zz,
            'sigma_rz': rel_error_sigma_rz,
            'sigma_phi_phi': rel_error_sigma_phi_phi,
        }
    }

    analytical = {
        'ur': ur_analytical,
        'uz': uz_analytical,
        'sigma_rr': sigma_rr_analytical,
        'sigma_zz': sigma_zz_analytical,
        'sigma_rz': sigma_rz_analytical,
        'sigma_phi_phi': sigma_phi_phi_analytical
    }

    var_names = ['ur', 'uz', 'sigma_rr', 'sigma_zz', 'sigma_rz', 'sigma_phi_phi']

    print_error_tables(errors, analytical, var_names)

    return errors

def print_error_tables(errors, analytical, var_names, tol=1e-12):
    print("\nAbsolute Errors:")
    header = "{:15} {:15} {:15} {:15}".format("Variable", "Mean", "Max", "Min")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for var in var_names:
        abs_err = errors['absolute_errors'][var]
        mean_err = np.mean(abs_err)
        max_err = np.max(abs_err)
        min_err = np.min(abs_err)
        print("{:15} {:15.5e} {:15.5e} {:15.5e}".format(var, mean_err, max_err, min_err))
    print("-" * len(header))
    
    print("\nRelative Errors (in percentage, only for variables with nonzero analytical solution):")
    header_perc = "{:15} {:15} {:15} {:15}".format("Variable", "Mean (%)", "Max (%)", "Min (%)")
    print("-" * len(header_perc))
    print(header_perc)
    print("-" * len(header_perc))
    for var in var_names:
        if np.any(np.abs(analytical[var]) > tol):
            rel_err = errors['relative_errors'][var]
            mean_rel = np.mean(rel_err) * 100.0
            max_rel = np.max(rel_err) * 100.0
            min_rel = np.min(rel_err) * 100.0
            print("{:15} {:15.5e} {:15.5e} {:15.5e}".format(var, mean_rel, max_rel, min_rel))
    print("-" * len(header_perc))

def get_deformed_coordinates(mesh, scale_factor=1.0):
    deformed_coords = []
    for node_id, node in mesh.nodes.items():
        r_deformed = node.r + scale_factor * node.displacements[0]
        z_deformed = node.z + scale_factor * node.displacements[1]
        deformed_coords.append((r_deformed, z_deformed))
    return np.array(deformed_coords)

def plot_u_sigma(mesh, r_min, r_max, p, mu, nu, fixed_z, material, compare_eps, shape_func):
    ur_analytical = []
    uz_analytical = []
    sigma_rr_analytical = []
    sigma_zz_analytical = []
    sigma_rz_analytical = []
    sigma_phi_phi_analytical = []

    for node_id, node in mesh.nodes.items():
        r = node.r
        z = node.z

        ur, uz, sigma_rr, sigma_zz, sigma_phi_phi, sigma_rz = analytical_solution(r, r_min, r_max, p, mu, nu)
        if abs(z - fixed_z) < compare_eps:
            ur_analytical.append(ur)
            uz_analytical.append(uz)
            sigma_rr_analytical.append(sigma_rr)
            sigma_zz_analytical.append(sigma_zz)
            sigma_rz_analytical.append(sigma_rz)
            sigma_phi_phi_analytical.append(sigma_phi_phi)

    r_values = []
    ur_values = []
    uz_values = []

    for node_id, node in mesh.nodes.items():
        r = node.r
        z = node.z
        if abs(z - fixed_z) < compare_eps:
            r_values.append(r)
            ur_values.append(node.displacements[0])
            uz_values.append(node.displacements[1])
    

    nodal_stresses = recover_nodal_stresses(mesh, material, shape_func, fixed_z, tol=compare_eps)
    
    r_nodal = []
    sigma_rr_nodal = []
    sigma_zz_nodal = []
    sigma_rz_nodal = []
    sigma_phi_phi_nodal = []
    
    for node_id, node in mesh.nodes.items():
        if abs(node.z - fixed_z) < compare_eps:
            if node_id in nodal_stresses:
                r_nodal.append(node.r)
                sigma_rr_nodal.append(nodal_stresses[node_id]['sigma_rr'])
                sigma_zz_nodal.append(nodal_stresses[node_id]['sigma_zz'])
                sigma_rz_nodal.append(nodal_stresses[node_id]['sigma_rz'])
                sigma_phi_phi_nodal.append(nodal_stresses[node_id]['sigma_tt'])
    

    plt.figure(figsize=(16, 8))

    plt.subplot(2, 3, 1)
    plt.plot(r_values, ur_values, 'bo-', label=f"FEM: u_r at z={fixed_z}")
    plt.plot(r_values, ur_analytical, color='orange', marker='o', label=f"Analytical: u_r at z={fixed_z}")
    plt.xlabel('Radius (r)')
    plt.ylabel('Radial Displacement ($u_r$)')
    plt.title('Radial Displacement $u_r$')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(r_nodal, sigma_rr_nodal, 'bo-', label='Recovered nodal σ_rr')
    plt.plot(r_values, sigma_rr_analytical, color='orange', label='Analytical')
    plt.xlabel('Radius (r)')
    plt.ylabel(r'$\sigma_{rr}$')
    plt.title('Radial Stress $\sigma_{rr}$')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.plot(r_nodal, sigma_zz_nodal, 'bo-', label='Recovered nodal σ_zz')
    plt.plot(r_values, sigma_zz_analytical, color='orange', label='Analytical')
    plt.xlabel('Radius (r)')
    plt.ylabel(r'$\sigma_{zz}$')
    plt.title('Axial Stress $\sigma_{zz}$')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 4)
    plt.plot(r_values, uz_values, 'bo-', label=f"FEM: u_z at z={fixed_z}")
    plt.plot(r_values, uz_analytical, color='orange', marker='o', label=f"Analytical: u_z at z={fixed_z}")
    plt.xlabel('Radius (r)')
    plt.ylabel('Axial Displacement ($u_z$)')
    plt.title('Axial Displacement $u_z$')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.plot(r_nodal, sigma_rz_nodal, 'bo-', label='Recovered nodal σ_rz')
    plt.plot(r_values, sigma_rz_analytical, color='orange', label='Analytical')
    plt.xlabel('Radius (r)')
    plt.ylabel(r'$\sigma_{rz}$')
    plt.title('Shear Stress $\sigma_{rz}$')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 6)
    plt.plot(r_nodal, sigma_phi_phi_nodal, 'bo-', label='Recovered nodal σ_φφ')
    plt.plot(r_values, sigma_phi_phi_analytical, color='orange', label='Analytical')
    plt.xlabel('Radius (r)')
    plt.ylabel(r'$\sigma_{\phi\phi}$')
    plt.title('Hoop Stress $\sigma_{\phi\phi}$')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show(block=False)

    abs_error_ur         = np.abs(np.array(ur_values) - np.array(ur_analytical))
    abs_error_uz         = np.abs(np.array(uz_values) - np.array(uz_analytical))
    abs_error_sigma_rr   = np.abs(np.array(sigma_rr_nodal) - np.array(sigma_rr_analytical))
    abs_error_sigma_zz   = np.abs(np.array(sigma_zz_nodal) - np.array(sigma_zz_analytical))
    abs_error_sigma_rz   = np.abs(np.array(sigma_rz_nodal) - np.array(sigma_rz_analytical))
    abs_error_sigma_phi_phi = np.abs(np.array(sigma_phi_phi_nodal) - np.array(sigma_phi_phi_analytical))


    plt.figure(figsize=(16, 8))

    plt.subplot(2, 3, 1)
    plt.plot(r_values, abs_error_ur, 'bo-', label='Absolute Error in u_r')
    plt.xlabel('Radius (r)')
    plt.ylabel('Absolute Error in u_r')
    plt.title('Absolute Error in Radial Displacement')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(r_nodal, abs_error_sigma_rr, 'bo-', label='Absolute Error in σ_rr')
    plt.xlabel('Radius (r)')
    plt.ylabel(r'Absolute Error in $\sigma_{rr}$')
    plt.title('Absolute Error in Radial Stress')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.plot(r_nodal, abs_error_sigma_zz, 'bo-', label='Absolute Error in σ_zz')
    plt.xlabel('Radius (r)')
    plt.ylabel(r'Absolute Error in $\sigma_{zz}$')
    plt.title('Absolute Error in Axial Stress')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 4)
    plt.plot(r_values, abs_error_uz, 'bo-', label='Absolute Error in u_z')
    plt.xlabel('Radius (r)')
    plt.ylabel('Absolute Error in u_z')
    plt.title('Absolute Error in Axial Displacement')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.plot(r_nodal, abs_error_sigma_rz, 'bo-', label='Absolute Error in σ_rz')
    plt.xlabel('Radius (r)')
    plt.ylabel(r'Absolute Error in $\sigma_{rz}$')
    plt.title('Absolute Error in Shear Stress')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 6)
    plt.plot(r_nodal, abs_error_sigma_phi_phi, 'bo-', label='Absolute Error in σ_φφ')
    plt.xlabel('Radius (r)')
    plt.ylabel(r'Absolute Error in $\sigma_{\phi\phi}$')
    plt.title('Absolute Error in Hoop Stress')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show(block=False)

    nodes_list = list(mesh.nodes.values())
    r_all = np.array([n.r for n in nodes_list])
    z_all = np.array([n.z for n in nodes_list])
    ur_all = np.array([n.displacements[0] for n in nodes_list])
    uz_all = np.array([n.displacements[1] for n in nodes_list])
    
    all_nodal_stresses = recover_all_nodal_stresses(mesh, material, shape_func, tol=compare_eps)
    
    sigma_rr_all = []
    sigma_zz_all = []
    sigma_rz_all = []
    sigma_phi_phi_all = []
    
    for node in nodes_list:
        nid = node.node_id
        if nid in all_nodal_stresses:
            sigma_rr_all.append(all_nodal_stresses[nid]['sigma_rr'])
            sigma_zz_all.append(all_nodal_stresses[nid]['sigma_zz'])
            sigma_rz_all.append(all_nodal_stresses[nid]['sigma_rz'])
            sigma_phi_phi_all.append(all_nodal_stresses[nid]['sigma_tt'])
        else:
            sigma_rr_all.append(0)
            sigma_zz_all.append(0)
            sigma_rz_all.append(0)
            sigma_phi_phi_all.append(0)
    
    sigma_rr_all = np.array(sigma_rr_all)
    sigma_zz_all = np.array(sigma_zz_all)
    sigma_rz_all = np.array(sigma_rz_all)
    sigma_phi_phi_all = np.array(sigma_phi_phi_all)
    
    r_grid = np.linspace(min(r_all), max(r_all), 50)
    z_grid = np.linspace(min(z_all), max(z_all), 50)
    R, Z = np.meshgrid(r_grid, z_grid)
    
    points = np.column_stack((r_all, z_all))
    
    U_r = griddata(points, ur_all, (R, Z), method='linear')
    U_z = griddata(points, uz_all, (R, Z), method='linear')
    Sigma_rr = griddata(points, sigma_rr_all, (R, Z), method='linear')
    Sigma_zz = griddata(points, sigma_zz_all, (R, Z), method='linear')
    Sigma_rz = griddata(points, sigma_rz_all, (R, Z), method='linear')
    Sigma_phi_phi = griddata(points, sigma_phi_phi_all, (R, Z), method='linear')
    
    fig = plt.figure(figsize=(20, 16))
    
    ax1 = fig.add_subplot(231, projection='3d')
    surf1 = ax1.plot_surface(R, Z, U_r, cmap='viridis')
    ax1.set_title(r'Surface Plot of $u_r$')
    ax1.set_xlabel('r')
    ax1.set_ylabel('z')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
    
    ax2 = fig.add_subplot(232, projection='3d')
    surf2 = ax2.plot_surface(R, Z, U_z, cmap='viridis')
    ax2.set_title(r'Surface Plot of $u_z$')
    ax2.set_xlabel('r')
    ax2.set_ylabel('z')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
    
    ax3 = fig.add_subplot(233, projection='3d')
    surf3 = ax3.plot_surface(R, Z, Sigma_rr, cmap='jet')
    ax3.set_title(r'Surface Plot of $\sigma_{rr}$')
    ax3.set_xlabel('r')
    ax3.set_ylabel('z')
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)
    
    ax4 = fig.add_subplot(234, projection='3d')
    surf4 = ax4.plot_surface(R, Z, Sigma_zz, cmap='jet')
    ax4.set_title(r'Surface Plot of $\sigma_{zz}$')
    ax4.set_xlabel('r')
    ax4.set_ylabel('z')
    fig.colorbar(surf4, ax=ax4, shrink=0.5, aspect=10)
    
    ax5 = fig.add_subplot(235, projection='3d')
    surf5 = ax5.plot_surface(R, Z, Sigma_rz, cmap='jet')
    ax5.set_title(r'Surface Plot of $\sigma_{rz}$')
    ax5.set_xlabel('r')
    ax5.set_ylabel('z')
    fig.colorbar(surf5, ax=ax5, shrink=0.5, aspect=10)
    
    ax6 = fig.add_subplot(236, projection='3d')
    surf6 = ax6.plot_surface(R, Z, Sigma_phi_phi, cmap='jet')
    ax6.set_title(r'Surface Plot of $\sigma_{\phi\phi}$')
    ax6.set_xlabel('r')
    ax6.set_ylabel('z')
    fig.colorbar(surf6, ax=ax6, shrink=0.5, aspect=10)
    
    plt.tight_layout()
    plt.show(block=True)

    compute_errors(
        r_values,
        sigma_rr_nodal, sigma_zz_nodal, sigma_rz_nodal, sigma_phi_phi_nodal,
        sigma_rr_analytical, sigma_zz_analytical, sigma_rz_analytical, sigma_phi_phi_analytical,
        ur_values, uz_values, ur_analytical, uz_analytical,
        fixed_z
    )

if __name__ == "__main__":
    main()