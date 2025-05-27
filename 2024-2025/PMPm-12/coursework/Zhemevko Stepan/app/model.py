import networkx as nx
import pandas as pd

def create_default_graph():
    G = nx.DiGraph()
    G.add_node("S1", type="source", source_type="main", supply=500)
    G.add_node("T1", type="transformer")
    G.add_node("H1", type="consumer", demand=100)
    G.add_node("H2", type="consumer", demand=120)
    G.add_node("H3", type="consumer", demand=80)
    G.add_edge("S1", "T1", capacity=400, loss=0.03)
    G.add_edge("T1", "H1", capacity=100, loss=0.01)
    G.add_edge("T1", "H2", capacity=150, loss=0.01)
    G.add_edge("T1", "H3", capacity=150, loss=0.01)
    return G

def load_graph_from_csv(file_storage):
    df = pd.read_csv(file_storage)
    G = nx.DiGraph()

    # Вузли зі сторони source
    for _, row in df.iterrows():
        s = row['source']
        if not G.has_node(s):
            G.add_node(s,
                       type=row.get('type_source', 'node'),
                       source_type=row.get('source_type_source', ''),
                       supply=try_parse_float(row.get('supply_source')),
                       demand=try_parse_float(row.get('demand_source')),
                       capacity=try_parse_float(row.get('capacity_source')),
                       charge=try_parse_float(row.get('charge_source')))

    # Вузли зі сторони target
    for _, row in df.iterrows():
        t = row['target']
        if not G.has_node(t):
            G.add_node(t,
                       type=row.get('type_target', 'node'),
                       source_type=row.get('source_type_target', ''),
                       supply=try_parse_float(row.get('supply_target')),
                       demand=try_parse_float(row.get('demand_target')),
                       capacity=try_parse_float(row.get('capacity_target')),
                       charge=try_parse_float(row.get('charge_target')))

    # Ребра
    for _, row in df.iterrows():
        G.add_edge(row['source'], row['target'],
                   capacity=row['capacity'],
                   loss=row['loss'])

    return G

def try_parse_float(val):
    try:
        return float(val)
    except:
        return 0.0
