from pyvis.network import Network
import tempfile

def draw_graph_interactive(G):
    net = Network(height="800px", width="100%", directed=True)
    net.barnes_hut()

    color_map = {
        "source": "#00BFFF",
        "consumer": "#FF6347",
        "transformer": "#FFD700",
        "hybrid": "#32CD32",
        "battery": "#9370DB"
    }

    for node, data in G.nodes(data=True):
        node_type = data.get("type", "unknown")
        color = color_map.get(node_type, "#D3D3D3")

        # Формування списку зв’язків
        neighbors = []
        for _, tgt, d in G.out_edges(node, data=True):
            neighbor_info = {
                "to": tgt,
                "capacity": d.get("capacity", 0),
                "loss": d.get("loss", 0),
                "target_demand": G.nodes[tgt].get("demand", 0)
            }
            neighbors.append(neighbor_info)

        data["links_info"] = neighbors

        net.add_node(
            node,
            label=node,
            color=color,
            title="Click for details",
            **data
        )

    for u, v, data in G.edges(data=True):
        label = f"cap: {data.get('capacity')}, loss: {data.get('loss')}"
        net.add_edge(u, v, label=label)

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.write_html(tmp_file.name)

    with open(tmp_file.name, "r", encoding="utf-8") as f:
        html = f.read()

    return html
