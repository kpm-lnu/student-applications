import numpy as np


def find_interface_nodes(nodes, tol=1e-8):
    """
    Find all nodes that lie on the interface x = 0.5.
    """
    return [i for i, (x, _) in enumerate(nodes) if abs(x - 0.5) < tol]


def extract_interface_edges(elements, interface_nodes):
    """
    Identify element edges that lie on the interface.
    Returns list of (element_index, local_edge_id).
    local_edge_id: 0-bottom, 1-right, 2-top, 3-left
    """
    edge_map = [(0, 1), (1, 2), (2, 3), (3, 0)]
    interface_set = set(interface_nodes)
    interface_edges = []
    for e_idx, elem in enumerate(elements):
        for loc_edge_id, (i, j) in enumerate(edge_map):
            if elem[i] in interface_set and elem[j] in interface_set:
                interface_edges.append((e_idx, loc_edge_id))
    return interface_edges


def midpoint(p1, p2):
    return 0.5 * (np.array(p1) + np.array(p2))


def create_projection_matrix(nodes_master, edges_master, nodes_slave, edges_slave):
    """
    Construct a basic L2 projection matrix from slave to master interface.
    """
    # NOTE: This is a placeholder logic to be refined for actual mortar projection.
    B_master = []
    B_slave = []

    for (e_m, edge_id_m), (e_s, edge_id_s) in zip(edges_master, edges_slave):
        # Get nodes on each edge
        edge_nodes_m = get_edge_nodes(nodes_master, e_m, edge_id_m)
        edge_nodes_s = get_edge_nodes(nodes_slave, e_s, edge_id_s)

        # Project midpoint of slave onto master
        mp_s = midpoint(nodes_slave[edge_nodes_s[0]], nodes_slave[edge_nodes_s[1]])
        mp_m = midpoint(nodes_master[edge_nodes_m[0]], nodes_master[edge_nodes_m[1]])

        # Build simple projection weights (dummy 1D case)
        B_slave.append([edge_nodes_s[0], edge_nodes_s[1]])
        B_master.append([edge_nodes_m[0], edge_nodes_m[1]])

    return B_master, B_slave

def build_mortar_matrices(nodes1, nodes2, slave_ids, master_ids):
    """
    Build constraint matrices B1 and B2 for the mortar method.
    """
    B1 = np.zeros((len(slave_ids), 2 * len(nodes1)))
    B2 = np.zeros((len(master_ids), 2 * len(nodes2)))

    for i, (s, m) in enumerate(zip(slave_ids, master_ids)):
        B1[i, 2 * s] = 1
        B1[i, 2 * s + 1] = 1
        B2[i, 2 * m] = 1
        B2[i, 2 * m + 1] = 1

    return B1, B2


def get_edge_nodes(nodes, element, edge_id):
    """Return global node indices for the edge of a Q4 element."""
    edge_node_map = [(0, 1), (1, 2), (2, 3), (3, 0)]
    return edge_node_map[edge_id]
