import networkx as nx


def compute_total_loss(G):
    total_loss = 0.0
    for u, v, data in G.edges(data=True):
        loss = data['loss'] * data['capacity']
        total_loss += loss
    return total_loss

def compute_supply_demand_balance(G):
    supply = sum(data.get('supply', 0) for _, data in G.nodes(data=True))
    demand = sum(data.get('demand', 0) for _, data in G.nodes(data=True))
    return supply, demand

def graph_summary(G):
    return {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
        'is_connected': nx.is_weakly_connected(G) if G.is_directed() else nx.is_connected(G)
    }
