import numpy as np
from scipy.sparse import csr_matrix


def fixed_dofs_for_nodes(nodes: np.ndarray, components: tuple[bool, bool] = (True, True)) -> np.ndarray:
    """
    Формує список зафіксованих ступенів вільності для вузлів.

    components=(True, False)  -> фіксується тільки u_x;
    components=(False, True)  -> фіксується тільки u_y;
    components=(True, True)   -> фіксуються u_x та u_y.
    """
    dofs = []
    for n in nodes:
        if components[0]:
            dofs.append(2 * int(n))
        if components[1]:
            dofs.append(2 * int(n) + 1)
    return np.array(dofs, dtype=int)


def solve_with_dirichlet(K: csr_matrix, F: np.ndarray, fixed_dofs: np.ndarray, fixed_values: np.ndarray | None = None, solver=None) -> np.ndarray:
    """
    Розв'язує систему Kd=F з нульовими або заданими переміщеннями на fixed_dofs.
    """
    from scipy.sparse.linalg import spsolve

    ndof = K.shape[0]
    fixed_dofs = np.unique(np.asarray(fixed_dofs, dtype=int))

    if fixed_values is None:
        fixed_values = np.zeros_like(fixed_dofs, dtype=float)
    else:
        fixed_values = np.asarray(fixed_values, dtype=float)

    all_dofs = np.arange(ndof)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    d = np.zeros(ndof, dtype=float)
    d[fixed_dofs] = fixed_values

    rhs = F[free_dofs] - K[free_dofs][:, fixed_dofs] @ d[fixed_dofs]
    Kff = K[free_dofs][:, free_dofs]

    if solver is None:
        d[free_dofs] = spsolve(Kff, rhs)
    else:
        d[free_dofs] = solver(Kff, rhs)

    return d
