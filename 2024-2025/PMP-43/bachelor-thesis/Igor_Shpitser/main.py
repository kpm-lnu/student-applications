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


from material import Material
from shapeFunction import LinearQuadrilateralShapeFunction, Quadratic8ShapeFunction
from mesh import Mesh
from boundaryConditions import DirichletBC, NeumannBC
from axisymmetricFEMSolver import AxisymmetricFEMSolver

from axisymmetric_quadrature_pso import PSOQuadrature
import numpy as np
import quadrature_data

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from node_variations import process_quadrilateral
from quadrature_algorithm import find_optimal_nodes, integrand

import numpy as np
import matplotlib.pyplot as plt

from axisymmetricQuadrature import AxisymmetricQuadrature
from axisymmetric_quadrature_pso import PSOQuadrature
from mesh import Mesh
from boundaryConditions import DirichletBC, NeumannBC
from axisymmetricFEMSolver import AxisymmetricFEMSolver
from material import Material
from shapeFunction import LinearQuadrilateralShapeFunction
import quadrature_data
from node_variations import process_quadrilateral
from quadrature_algorithm import find_optimal_nodes, integrand

# ─── 1) Побудова розв’язків (2×3 субплоти) ────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt

# ─── 1) Побудова розв’язків (2×3 субплоти) ────────────────────────────────────
def plot_convergence(results, r_a, ur_a, uz_a, sig_rr_a, sig_zz_a, sig_rz_a, sig_tt_a, fixed_z):
    styles = {
        (2,4):  ('C0','o-'),
        (2,8):  ('C1','s-'),
        (2,16): ('C2','^-'),
        (2,32): ('C3','d-'),
        (10,10):('C4','x-')  # додано для сітки 10×10
    }
    fig, axes = plt.subplots(2,3, figsize=(16,8))
    axs = axes.flatten()

    # 1) u_r
    ax = axs[0]
    for key,data in results.items():
        c, ls = styles[key]
        ax.plot(data['r_disp'], data['ur'], ls, color=c, label=f'FEM {key[0]}×{key[1]}')
    ax.plot(r_a, ur_a, 'k--', label='Analytical')
    ax.set_title(f'Radial Displacement $u_r$ at z={fixed_z}')
    ax.set_xlabel('r'); ax.set_ylabel('$u_r$'); ax.legend(); ax.grid(True)

    # 2) σ_rr
    ax = axs[1]
    for key,data in results.items():
        c, ls = styles[key]
        ax.plot(data['r_stress'], data['sigma_rr'], ls, color=c, label=f'FEM {key[0]}×{key[1]}')
    ax.plot(r_a, sig_rr_a, 'k--', label='Analytical')
    ax.set_title(r'Radial Stress $\sigma_{rr}$')
    ax.set_xlabel('r'); ax.set_ylabel(r'$\sigma_{rr}$'); ax.legend(); ax.grid(True)

    # 3) σ_zz
    ax = axs[2]
    for key,data in results.items():
        c, ls = styles[key]
        ax.plot(data['r_stress'], data['sigma_zz'], ls, color=c, label=f'FEM {key[0]}×{key[1]}')
    ax.plot(r_a, sig_zz_a, 'k--', label='Analytical')
    ax.set_title(r'Axial Stress $\sigma_{zz}$')
    ax.set_xlabel('r'); ax.set_ylabel(r'$\sigma_{zz}$'); ax.legend(); ax.grid(True)

    # 4) u_z
    ax = axs[3]
    for key,data in results.items():
        c, ls = styles[key]
        ax.plot(data['r_disp'], data['uz'], ls, color=c, label=f'FEM {key[0]}×{key[1]}')
    ax.plot(r_a, uz_a, 'k--', label='Analytical')
    ax.set_title(f'Axial Displacement $u_z$ at z={fixed_z}')
    ax.set_xlabel('r'); ax.set_ylabel('$u_z$'); ax.legend(); ax.grid(True)

    # 5) σ_rz
    ax = axs[4]
    for key,data in results.items():
        c, ls = styles[key]
        ax.plot(data['r_stress'], data['sigma_rz'], ls, color=c, label=f'FEM {key[0]}×{key[1]}')
    ax.plot(r_a, sig_rz_a, 'k--', label='Analytical')
    ax.set_title(r'Shear Stress $\sigma_{rz}$')
    ax.set_xlabel('r'); ax.set_ylabel(r'$\sigma_{rz}$'); ax.legend(); ax.grid(True)

    # 6) σ_φφ
    ax = axs[5]
    for key,data in results.items():
        c, ls = styles[key]
        ax.plot(data['r_stress'], data['sigma_tt'], ls, color=c, label=f'FEM {key[0]}×{key[1]}')
    ax.plot(r_a, sig_tt_a, 'k--', label='Analytical')
    ax.set_title(r'Hoop Stress $\sigma_{\phi\phi}$')
    ax.set_xlabel('r'); ax.set_ylabel(r'$\sigma_{\phi\phi}$'); ax.legend(); ax.grid(True)

    plt.tight_layout()
    plt.show()

# ─── 2) Побудова абсолютних похибок ──────────────────────────────────────────
def plot_errors(all_results, r_a, ur_a, uz_a, sig_rr_a, sig_zz_a, sig_rz_a, sig_tt_a):
    styles = {
        (2,4):  ('C0','o-'),
        (2,8):  ('C1','s-'),
        (2,16): ('C2','^-'),
        (2,32): ('C3','d-'),
        (10,10):('C4','x-')  # додано для сітки 10×10
    }
    fig, axes = plt.subplots(2,3, figsize=(16,8))
    axs = axes.flatten()

    # helper to interp analytic on FE r
    def interp(x_fe, x_a, y_a):
        return np.interp(x_fe, x_a, y_a)

    # 1) Radial disp. error
    ax = axs[0]
    for key, data in all_results.items():
        c, ls = styles[key]
        r_fe = np.array(data['r_disp'])
        err = np.abs(np.array(data['ur']) - interp(r_fe, r_a, ur_a))
        ax.plot(r_fe, err, ls, color=c, label=f'FEM {key[0]}×{key[1]}')
    ax.set_title('Absolute Error in Radial Displacement')
    ax.set_xlabel('r')
    ax.set_ylabel('$|Δu_r|$')
    ax.grid(True)
    ax.legend()

    # 2) Radial stress error
    ax = axs[1]
    for key, data in all_results.items():
        c, ls = styles[key]
        r_fe = np.array(data['r_stress'])
        err = np.abs(np.array(data['sigma_rr']) - interp(r_fe, r_a, sig_rr_a))
        ax.plot(r_fe, err, ls, color=c, label=f'FEM {key[0]}×{key[1]}')
    ax.set_title('Absolute Error in Radial Stress')
    ax.set_xlabel('r')
    ax.set_ylabel('$|Δσ_{rr}|$')
    ax.grid(True)
    ax.legend()

    # 3) Axial stress error
    ax = axs[2]
    for key, data in all_results.items():
        c, ls = styles[key]
        r_fe = np.array(data['r_stress'])
        err = np.abs(np.array(data['sigma_zz']) - interp(r_fe, r_a, sig_zz_a))
        ax.plot(r_fe, err, ls, color=c, label=f'FEM {key[0]}×{key[1]}')
    ax.set_title('Absolute Error in Axial Stress')
    ax.set_xlabel('r')
    ax.set_ylabel('$|Δσ_{zz}|$')
    ax.grid(True)
    ax.legend()

    # 4) Axial disp. error
    ax = axs[3]
    for key, data in all_results.items():
        c, ls = styles[key]
        r_fe = np.array(data['r_disp'])
        err = np.abs(np.array(data['uz']) - interp(r_fe, r_a, uz_a))
        ax.plot(r_fe, err, ls, color=c, label=f'FEM {key[0]}×{key[1]}')
    ax.set_title('Absolute Error in Axial Displacement')
    ax.set_xlabel('r')
    ax.set_ylabel('$|Δu_z|$')
    ax.grid(True)
    ax.legend()

    # 5) Shear stress error
    ax = axs[4]
    for key, data in all_results.items():
        c, ls = styles[key]
        r_fe = np.array(data['r_stress'])
        err = np.abs(np.array(data['sigma_rz']) - interp(r_fe, r_a, sig_rz_a))
        ax.plot(r_fe, err, ls, color=c, label=f'FEM {key[0]}×{key[1]}')
    ax.set_title('Absolute Error in Shear Stress')
    ax.set_xlabel('r')
    ax.set_ylabel('$|Δσ_{rz}|$')
    ax.grid(True)
    ax.legend()

    # 6) Hoop stress error
    ax = axs[5]
    for key, data in all_results.items():
        c, ls = styles[key]
        r_fe = np.array(data['r_stress'])
        err = np.abs(np.array(data['sigma_tt']) - interp(r_fe, r_a, sig_tt_a))
        ax.plot(r_fe, err, ls, color=c, label=f'FEM {key[0]}×{key[1]}')
    ax.set_title('Absolute Error in Hoop Stress')
    ax.set_xlabel('r')
    ax.set_ylabel('$|Δσ_{\phi\phi}|$')
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()

# ─── 3) Головна функція ───────────────────────────────────────────────────────
def main():
    # геометрія та параметри
    r_min, r_max = 1.0, 2.0
    z_min, z_max = 0.0, 1.0
    nu, mu, p = 0.3, 0.7, 1.0
    E = 2 * mu * (1 + nu)
    node_dof = 2
    quadrature_points = 2
    compare_eps = 1e-6
    fixed_z = 1.0

    # material, shape
    mat = Material("Test", E=E, nu=nu)
    shape_func = LinearQuadrilateralShapeFunction()

    # базова квадратура (Gauss-Legendre)
    quad_base = AxisymmetricQuadrature(n_points=quadrature_points)

    # PSO-квадратура тільки для виводу fitness
    best    = max(quadrature_data.data, key=lambda rec: rec["fitness"])
    labels  = best["labels"]; fitness = best["fitness"]
    nG, nn = 2, 2*2
    xi      = np.array(labels[   0:   nn])
    eta     = np.array(labels[nn: 2*nn])
    w       = np.array(labels[2*nn: 3*nn])
    quad_pso = PSOQuadrature(xi=xi, eta=eta, weights=w)
    print(f"▶ Using PSO‐quadrature (fitness = {fitness:.3e})")

    # збір даних для чотирьох розділень + нового
    resolutions = [(2,4),(2,8),(2,16),(2,32),(10,10)]  # додано (10,10)
    all_results = {}
    for rN,zN in resolutions:
        mesh = Mesh(material=mat, shape_func=shape_func,
                    quadrature=quad_base, node_dof=node_dof)
        mesh.generate_rectangles(r_min, r_max, z_min, z_max, rN, zN)
        left   = [nid for nid,node in mesh.nodes.items() if abs(node.r - r_min) < compare_eps]
        right  = [nid for nid,node in mesh.nodes.items() if abs(node.r - r_max) < compare_eps]
        bottom = [nid for nid,node in mesh.nodes.items() if abs(node.z - z_min) < compare_eps]
        top    = [nid for nid,node in mesh.nodes.items() if abs(node.z - z_max) < compare_eps]
        bcs = [NeumannBC(left, dof=0, pressure=p), NeumannBC(right, dof=0, pressure=0.0)]
        for nid in bottom + top:
            bcs.append(DirichletBC(node_id=nid, dof=1, value=0.0))
        AxisymmetricFEMSolver(mesh, bcs).run()

        # displacements @ fixed_z
        r_disp, ur, uz = [],[],[]
        for nid,node in mesh.nodes.items():
            if abs(node.z - fixed_z) < compare_eps:
                r_disp.append(node.r); ur.append(node.displacements[0]); uz.append(node.displacements[1])
        idx = np.argsort(r_disp)
        r_disp = list(np.array(r_disp)[idx])
        ur     = list(np.array(ur)[idx])
        uz     = list(np.array(uz)[idx])

        # stresses @ fixed_z
        nodal = recover_nodal_stresses(mesh, mat, shape_func, fixed_z, tol=compare_eps)
        keys  = sorted(nodal.keys(), key=lambda nid: mesh.nodes[nid].r)
        r_st  = [mesh.nodes[n].r for n in keys]
        sig_rr = [nodal[n]['sigma_rr'] for n in keys]
        sig_zz = [nodal[n]['sigma_zz'] for n in keys]
        sig_rz = [nodal[n]['sigma_rz'] for n in keys]
        sig_tt = [nodal[n]['sigma_tt'] for n in keys]

        all_results[(rN,zN)] = {
            'r_disp':   r_disp, 'ur': ur,   'uz': uz,
            'r_stress': r_st,   'sigma_rr': sig_rr,
            'sigma_zz': sig_zz, 'sigma_rz': sig_rz,
            'sigma_tt': sig_tt
        }

    # аналітичні криві беремо по першій сітці
    r_a = all_results[(10,10)]['r_disp']
    ur_a, uz_a, sig_rr_a, sig_zz_a, sig_rz_a, sig_tt_a = [],[],[],[],[],[]
    for r in r_a:
        u_r,u_z,s_rr,s_zz,s_phi,s_rz = analytical_solution(r, r_min, r_max, p, mu, nu)
        ur_a .append(u_r); uz_a.append(u_z)
        sig_rr_a.append(s_rr); sig_zz_a.append(s_zz)
        sig_tt_a.append(s_phi); sig_rz_a.append(s_rz)

    # побудова графіків
    plot_errors(all_results, r_a, ur_a, uz_a, sig_rr_a, sig_zz_a, sig_rz_a, sig_tt_a)
    plot_convergence(all_results, r_a, ur_a, uz_a, sig_rr_a, sig_zz_a, sig_rz_a, sig_tt_a, fixed_z)

    # ваші інші блоки…
    quad0 = [[0,0],[0,1],[1,1],[1,0]]
    norm_q, vars_ = process_quadrilateral(quad0, 10)
    print("Normalized Quadrilateral:", norm_q)
    print("First 3 Variations:", vars_[:3])
    n_opt, val, err = find_optimal_nodes(integrand, 0.0, np.pi, 2.0)
    print(f"Optimal nodes: {n_opt}, integral={val:.6f}, error={err:.2e}")




def analytical_solution(r, a, b, p, mu, nu):
    ur = (1 / (2 * mu * (b ** 2 - a ** 2))) * ((1 - 2 * nu) * a ** 2 * p * r + p * a ** 2 * b ** 2 / r)
    uz = 0
    sigma_rr = (1 / (b ** 2 - a ** 2)) * (a ** 2 * p - a ** 2 * b ** 2 / r ** 2)
    sigma_zz = (2 * nu * a ** 2 * p) / (b ** 2 - a ** 2)
    sigma_rz = 0
    sigma_phi_phi = (1 / (b ** 2 - a ** 2)) * (a ** 2 * p + a ** 2 * b ** 2 / r ** 2)

    return ur, uz, sigma_rr, sigma_zz, sigma_phi_phi, sigma_rz


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
            xi = gp["xi"]
            eta = gp["eta"]
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
            xi = gp["xi"]
            eta = gp["eta"]
            N, dN_dxi, dN_deta = elem.shape_func.evaluate(xi, eta)
            J = np.zeros((2, 2))
            for i in range(len(nodes_coords)):
                J[0, 0] += dN_dxi[i] * nodes_coords[i, 0]
                J[0, 1] += dN_dxi[i] * nodes_coords[i, 1]
                J[1, 0] += dN_deta[i] * nodes_coords[i, 0]
                J[1, 1] += dN_deta[i] * nodes_coords[i, 1]
            invJ = np.linalg.inv(J)
            dN_dr = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
            dN_dz = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
            r_gp = np.dot(N, nodes_coords[:, 0])
            B = np.zeros((4, 2 * n_el_nodes))
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

    # plt.figure(figsize=(16, 8))
    #
    # plt.subplot(2, 3, 1)
    # plt.plot(r_values, ur_values, 'bo-', label=f"FEM: u_r at z={fixed_z}")
    # plt.plot(r_values, ur_analytical, color='orange', marker='o', label=f"Analytical: u_r at z={fixed_z}")
    # plt.xlabel('Radius (r)')
    # plt.ylabel('Radial Displacement ($u_r$)')
    # plt.title('Radial Displacement $u_r$')
    # plt.legend()
    # plt.grid(True)
    #
    # plt.subplot(2, 3, 2)
    # plt.plot(r_nodal, sigma_rr_nodal, 'bo-', label='Recovered nodal σ_rr')
    # plt.plot(r_values, sigma_rr_analytical, color='orange', label='Analytical')
    # plt.xlabel('Radius (r)')
    # plt.ylabel(r'$\sigma_{rr}$')
    # plt.title('Radial Stress $\\sigma_{rr}$')
    # plt.legend()
    # plt.grid(True)
    #
    # plt.subplot(2, 3, 3)
    # plt.plot(r_nodal, sigma_zz_nodal, 'bo-', label='Recovered nodal σ_zz')
    # plt.plot(r_values, sigma_zz_analytical, color='orange', label='Analytical')
    # plt.xlabel('Radius (r)')
    # plt.ylabel(r'$\sigma_{zz}$')
    # plt.title('Axial Stress $\\sigma_{zz}$')
    # plt.legend()
    # plt.grid(True)
    #
    # plt.subplot(2, 3, 4)
    # plt.plot(r_values, uz_values, 'bo-', label=f"FEM: u_z at z={fixed_z}")
    # plt.plot(r_values, uz_analytical, color='orange', marker='o', label=f"Analytical: u_z at z={fixed_z}")
    # plt.xlabel('Radius (r)')
    # plt.ylabel('Axial Displacement ($u_z$)')
    # plt.title('Axial Displacement $u_z$')
    # plt.legend()
    # plt.grid(True)
    #
    # plt.subplot(2, 3, 5)
    # plt.plot(r_nodal, sigma_rz_nodal, 'bo-', label='Recovered nodal σ_rz')
    # plt.plot(r_values, sigma_rz_analytical, color='orange', label='Analytical')
    # plt.xlabel('Radius (r)')
    # plt.ylabel(r'$\sigma_{rz}$')
    # plt.title('Shear Stress $\\sigma_{rz}$')
    # plt.legend()
    # plt.grid(True)
    #
    # plt.subplot(2, 3, 6)
    # plt.plot(r_nodal, sigma_phi_phi_nodal, 'bo-', label='Recovered nodal σ_φφ')
    # plt.plot(r_values, sigma_phi_phi_analytical, color='orange', label='Analytical')
    # plt.xlabel('Radius (r)')
    # plt.ylabel(r'$\sigma_{\phi\phi}$')
    # plt.title(r'Hoop Stress $\sigma_{\phi\phi}$')

    # plt.legend()
    # plt.grid(True)
    #
    # plt.tight_layout()
    # plt.show(block=False)

    abs_error_ur = np.abs(np.array(ur_values) - np.array(ur_analytical))
    abs_error_uz = np.abs(np.array(uz_values) - np.array(uz_analytical))
    abs_error_sigma_rr = np.abs(np.array(sigma_rr_nodal) - np.array(sigma_rr_analytical))
    abs_error_sigma_zz = np.abs(np.array(sigma_zz_nodal) - np.array(sigma_zz_analytical))
    abs_error_sigma_rz = np.abs(np.array(sigma_rz_nodal) - np.array(sigma_rz_analytical))
    abs_error_sigma_phi_phi = np.abs(np.array(sigma_phi_phi_nodal) - np.array(sigma_phi_phi_analytical))

    # plt.figure(figsize=(16, 8))
    #
    # plt.subplot(2, 3, 1)
    # plt.plot(r_values, abs_error_ur, 'bo-', label='Absolute Error in u_r')
    # plt.xlabel('Radius (r)')
    # plt.ylabel('Absolute Error in u_r')
    # plt.title('Absolute Error in Radial Displacement')
    # plt.legend()
    # plt.grid(True)
    #
    # plt.subplot(2, 3, 2)
    # plt.plot(r_nodal, abs_error_sigma_rr, 'bo-', label='Absolute Error in σ_rr')
    # plt.xlabel('Radius (r)')
    # plt.ylabel(r'Absolute Error in $\sigma_{rr}$')
    # plt.title('Absolute Error in Radial Stress')
    # plt.legend()
    # plt.grid(True)
    #
    # plt.subplot(2, 3, 3)
    # plt.plot(r_nodal, abs_error_sigma_zz, 'bo-', label='Absolute Error in σ_zz')
    # plt.xlabel('Radius (r)')
    # plt.ylabel(r'Absolute Error in $\sigma_{zz}$')
    # plt.title('Absolute Error in Axial Stress')
    # plt.legend()
    # plt.grid(True)
    #
    # plt.subplot(2, 3, 4)
    # plt.plot(r_values, abs_error_uz, 'bo-', label='Absolute Error in u_z')
    # plt.xlabel('Radius (r)')
    # plt.ylabel('Absolute Error in u_z')
    # plt.title('Absolute Error in Axial Displacement')
    # plt.legend()
    # plt.grid(True)
    #
    # plt.subplot(2, 3, 5)
    # plt.plot(r_nodal, abs_error_sigma_rz, 'bo-', label='Absolute Error in σ_rz')
    # plt.xlabel('Radius (r)')
    # plt.ylabel(r'Absolute Error in $\sigma_{rz}$')
    # plt.title('Absolute Error in Shear Stress')
    # plt.legend()
    # plt.grid(True)
    #
    # plt.subplot(2, 3, 6)
    # plt.plot(r_nodal, abs_error_sigma_phi_phi, 'bo-', label='Absolute Error in σ_φφ')
    # plt.xlabel('Radius (r)')
    # plt.ylabel(r'Absolute Error in $\sigma_{\phi\phi}$')
    # plt.title('Absolute Error in Hoop Stress')
    # plt.legend()
    # plt.grid(True)

    # plt.tight_layout()
    # plt.show(block=False)

    nodes_list = list(mesh.nodes.values())

    nodes_list.sort(key=lambda n: (n.z, n.r))
    r_all = np.array([n.r for n in nodes_list])
    z_all = np.array([n.z for n in nodes_list])
    ur_all = np.array([n.displacements[0] for n in nodes_list])
    uz_all = np.array([n.displacements[1] for n in nodes_list])

    r_unique = np.unique(r_all)
    z_unique = np.unique(z_all)
    R, Z = np.meshgrid(r_unique, z_unique)

    U_r = ur_all.reshape((len(z_unique), len(r_unique)))
    U_z = uz_all.reshape((len(z_unique), len(r_unique)))

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
    sigma_rr_all = np.array(sigma_rr_all).reshape((len(z_unique), len(r_unique)))
    sigma_zz_all = np.array(sigma_zz_all).reshape((len(z_unique), len(r_unique)))
    sigma_rz_all = np.array(sigma_rz_all).reshape((len(z_unique), len(r_unique)))
    sigma_phi_phi_all = np.array(sigma_phi_phi_all).reshape((len(z_unique), len(r_unique)))

    # fig = plt.figure(figsize=(20, 16))
    # ax1 = fig.add_subplot(231, projection='3d')
    # surf1 = ax1.plot_surface(R, Z, U_r, cmap='viridis')
    # ax1.set_title(r'Surface Plot of $u_r$')
    # ax1.set_xlabel('r')
    # ax1.set_ylabel('z')
    # fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
    #
    # ax2 = fig.add_subplot(232, projection='3d')
    # surf2 = ax2.plot_surface(R, Z, U_z, cmap='viridis')
    # ax2.set_title(r'Surface Plot of $u_z$')
    # ax2.set_xlabel('r')
    # ax2.set_ylabel('z')
    # fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
    #
    # ax3 = fig.add_subplot(233, projection='3d')
    # surf3 = ax3.plot_surface(R, Z, sigma_rr_all, cmap='jet')
    # ax3.set_title(r'Surface Plot of $\sigma_{rr}$')
    # ax3.set_xlabel('r')
    # ax3.set_ylabel('z')
    # fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)
    #
    # ax4 = fig.add_subplot(234, projection='3d')
    # surf4 = ax4.plot_surface(R, Z, sigma_zz_all, cmap='jet')
    # ax4.set_title(r'Surface Plot of $\sigma_{zz}$')
    # ax4.set_xlabel('r')
    # ax4.set_ylabel('z')
    # fig.colorbar(surf4, ax=ax4, shrink=0.5, aspect=10)
    #
    # ax5 = fig.add_subplot(235, projection='3d')
    # surf5 = ax5.plot_surface(R, Z, sigma_rz_all, cmap='jet')
    # ax5.set_title(r'Surface Plot of $\sigma_{rz}$')
    # ax5.set_xlabel('r')
    # ax5.set_ylabel('z')
    # fig.colorbar(surf5, ax=ax5, shrink=0.5, aspect=10)
    #
    # ax6 = fig.add_subplot(236, projection='3d')
    # surf6 = ax6.plot_surface(R, Z, sigma_phi_phi_all, cmap='jet')
    # ax6.set_title(r'Surface Plot of $\sigma_{\phi\phi}$')
    # ax6.set_xlabel('r')
    # ax6.set_ylabel('z')
    # fig.colorbar(surf6, ax=ax6, shrink=0.5, aspect=10)
    #
    # plt.tight_layout()
    # plt.show(block=True)

    compute_errors(
        r_values,
        sigma_rr_nodal, sigma_zz_nodal, sigma_rz_nodal, sigma_phi_phi_nodal,
        sigma_rr_analytical, sigma_zz_analytical, sigma_rz_analytical, sigma_phi_phi_analytical,
        ur_values, uz_values, ur_analytical, uz_analytical,
        fixed_z
    )


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
                    xi = gp["xi"]
                    eta = gp["eta"]
                    N, dN_dxi, dN_deta = elem.shape_func.evaluate(xi, eta)
                    J = np.zeros((mesh.node_dof, mesh.node_dof))
                    for i in range(len(nodes_coords)):
                        J[0, 0] += dN_dxi[i] * nodes_coords[i, 0]
                        J[0, 1] += dN_dxi[i] * nodes_coords[i, 1]
                        J[1, 0] += dN_deta[i] * nodes_coords[i, 0]
                        J[1, 1] += dN_deta[i] * nodes_coords[i, 1]
                    invJ = np.linalg.inv(J)
                    dN_dr = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
                    dN_dz = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
                    r_gp = np.dot(N, nodes_coords[:, 0])
                    B = np.zeros((4, 2 * n_el_nodes))
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




if __name__ == "__main__":
    main()


