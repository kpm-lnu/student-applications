import time
from dataclasses import dataclass
import numpy as np
from .assembly import assemble_stiffness
from .boundary_conditions import solve_with_dirichlet


@dataclass
class FEMResult:
    displacements: np.ndarray
    strains: np.ndarray
    stresses: np.ndarray
    von_mises: np.ndarray
    element_centers: np.ndarray
    max_displacement: float
    max_von_mises: float
    max_sigma_x: float
    max_sigma_y: float
    max_tau_xy: float
    elapsed_time: float
    n_nodes: int
    n_elements: int
    n_dofs: int


def solve_elasticity(mesh, D: np.ndarray, thickness: float, F: np.ndarray, fixed_dofs: np.ndarray) -> FEMResult:
    """Повний цикл FEM-розрахунку: K, граничні умови, переміщення, деформації, напруження."""
    start = time.perf_counter()

    K, B_list, dof_list, _areas = assemble_stiffness(mesh.nodes, mesh.elements, D, thickness)
    d = solve_with_dirichlet(K, F, fixed_dofs)

    strains = []
    stresses = []
    centers = []

    for elem, B, dofs in zip(mesh.elements, B_list, dof_list):
        de = d[dofs]
        eps = B @ de
        sig = D @ eps
        strains.append(eps)
        stresses.append(sig)
        centers.append(mesh.nodes[elem].mean(axis=0))

    strains = np.asarray(strains)
    stresses = np.asarray(stresses)
    centers = np.asarray(centers)

    sigma_x = stresses[:, 0]
    sigma_y = stresses[:, 1]
    tau_xy = stresses[:, 2]
    von_mises = np.sqrt(sigma_x ** 2 - sigma_x * sigma_y + sigma_y ** 2 + 3.0 * tau_xy ** 2)

    ux = d[0::2]
    uy = d[1::2]
    disp_mag = np.sqrt(ux ** 2 + uy ** 2)

    elapsed = time.perf_counter() - start

    return FEMResult(
        displacements=d,
        strains=strains,
        stresses=stresses,
        von_mises=von_mises,
        element_centers=centers,
        max_displacement=float(np.max(disp_mag)),
        max_von_mises=float(np.max(von_mises)),
        max_sigma_x=float(np.max(np.abs(sigma_x))),
        max_sigma_y=float(np.max(np.abs(sigma_y))),
        max_tau_xy=float(np.max(np.abs(tau_xy))),
        elapsed_time=float(elapsed),
        n_nodes=len(mesh.nodes),
        n_elements=len(mesh.elements),
        n_dofs=2 * len(mesh.nodes),
    )


def element_values_to_nodes(n_nodes: int, elements: np.ndarray, elem_values: np.ndarray) -> np.ndarray:
    """Усереднює елементні значення у вузли для побудови графіків."""
    nodal = np.zeros(n_nodes, dtype=float)
    counts = np.zeros(n_nodes, dtype=float)

    for elem, val in zip(elements, elem_values):
        for node in elem:
            nodal[int(node)] += float(val)
            counts[int(node)] += 1.0

    counts[counts == 0.0] = 1.0
    return nodal / counts
