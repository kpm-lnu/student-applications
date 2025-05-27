import networkx as nx
from scipy.optimize import linprog
import math

def optimize_energy_network(G):
    edge_list = list(G.edges)
    edge_index = {e: i for i, e in enumerate(edge_list)}
    c = [G[u][v]['loss'] for u, v in edge_list]
    bounds = [(0, G[u][v]['capacity']) for u, v in edge_list]
    A_eq, b_eq = [], []

    for node in G.nodes:
        row = [0] * len(edge_list)
        for u, v in G.in_edges(node): row[edge_index[(u, v)]] = 1
        for u, v in G.out_edges(node): row[edge_index[(u, v)]] -= 1
        supply = G.nodes[node].get('supply', 0)
        demand = G.nodes[node].get('demand', 0)

        if not isinstance(supply, (int, float)) or math.isnan(supply): supply = 0
        if not isinstance(demand, (int, float)) or math.isnan(demand): demand = 0

        net_supply = supply - demand
        b_eq.append(-net_supply)
        A_eq.append(row)

    res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    print(res)
    flow_dict = {f"{u}->{v}": res.x[i] for i, (u, v) in enumerate(edge_list)} if res.success else {}

    return {
        "success": res.success,
        "status": res.status,
        "message": res.message,
        "total_loss": res.fun if res.success else None,
        "flows": flow_dict
    }

def analyze_solution(G, flows):
    stats = {"total_supply": 0, "total_demand": 0, "delivered": 0, "used_edges": 0, "saturated_edges": 0}
    for n, d in G.nodes(data=True):
        stats["total_supply"] += d.get('supply', 0)
        stats["total_demand"] += d.get('demand', 0)

    for u, v in G.edges:
        f = flows.get(f"{u}->{v}", 0)
        stats["used_edges"] += f > 0
        stats["saturated_edges"] += f >= G[u][v]['capacity'] * 0.99

    for h in [n for n, d in G.nodes(data=True) if d.get('demand', 0) > 0]:
        delivered = sum(flows.get(f"{u}->{h}", 0) for u in G.predecessors(h))
        stats["delivered"] += delivered

    stats["unmet_demand"] = stats["total_demand"] - stats["delivered"]
    return stats

def greedy_energy_distribution(G):
    result, flow_dict = {"flows": {}, "delivered": 0, "used_edges": 0, "saturated_edges": 0}, {}
    sources = [n for n, d in G.nodes(data=True) if d.get('supply', 0) > 0]
    consumers = [n for n, d in G.nodes(data=True) if d.get('demand', 0) > 0]

    for consumer in consumers:
        demand, delivered = G.nodes[consumer].get('demand', 0), 0
        for source in sources:
            if delivered >= demand: break
            supply = G.nodes[source].get('supply', 0)
            try:
                path = nx.shortest_path(G, source, consumer, weight='loss')
                capacities = [G[path[i]][path[i + 1]]['capacity'] for i in range(len(path) - 1)]
                flow_possible = min(supply, demand - delivered, *capacities)
                if flow_possible <= 0: continue
                G.nodes[source]['supply'] -= flow_possible
                delivered += flow_possible
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    G[u][v]['capacity'] -= flow_possible
                    flow_dict[f"{u}->{v}"] = flow_dict.get(f"{u}->{v}", 0) + flow_possible
            except nx.NetworkXNoPath:
                continue
        result["delivered"] += delivered

    result["flows"] = flow_dict
    result["used_edges"] = sum(1 for f in flow_dict.values() if f > 0)
    result["saturated_edges"] = sum(1 for u, v in G.edges if G[u][v]['capacity'] == 0)
    result["unmet_demand"] = sum(G.nodes[n].get('demand', 0) for n in consumers) - result["delivered"]
    print(result)
    return result

def dijkstra_energy_routing(G):
    result, flow_dict = {"flows": {}, "delivered": 0, "used_edges": 0, "saturated_edges": 0}, {}
    sources = [n for n, d in G.nodes(data=True) if d.get('supply', 0) > 0]
    consumers = [n for n, d in G.nodes(data=True) if d.get('demand', 0) > 0]

    for c in consumers:
        demand = G.nodes[c].get('demand', 0)
        for s in sources:
            supply = G.nodes[s].get('supply', 0)
            if supply <= 0 or demand <= 0: continue
            try:
                path = nx.dijkstra_path(G, s, c, weight='loss')
                capacities = [G[path[i]][path[i + 1]]['capacity'] for i in range(len(path) - 1)]
                flow_possible = min(demand, supply, *capacities)
                if flow_possible <= 0: continue
                G.nodes[s]['supply'] -= flow_possible
                G.nodes[c]['demand'] -= flow_possible
                result['delivered'] += flow_possible
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    G[u][v]['capacity'] -= flow_possible
                    flow_dict[f"{u}->{v}"] = flow_dict.get(f"{u}->{v}", 0) + flow_possible
            except nx.NetworkXNoPath:
                continue

    result["flows"] = flow_dict
    result["used_edges"] = sum(1 for f in flow_dict.values() if f > 0)
    result["saturated_edges"] = sum(1 for u, v in G.edges if G[u][v]['capacity'] <= 0.01)
    result["unmet_demand"] = sum(G.nodes[n].get('demand', 0) for n in consumers)
    print(result)
    return result
