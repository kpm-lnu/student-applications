# generate_graph.py
import json
import random
import os

def generate_connected_graph(num_nodes=50, extra_edges=100):
    nodes = []
    edges = []

    # Створюємо вузли
    for i in range(num_nodes):
        nodes.append({
            "id": i,
            "x": random.uniform(0, 100),
            "y": random.uniform(0, 100)
        })



    # Спочатку створимо мінімальне  дерево (зв’язний граф)
    connected = set([0])
    while len(connected) < num_nodes:
        a = random.choice(list(connected))
        b = random.choice([i for i in range(num_nodes) if i not in connected])
        length = random.uniform(1, 20)
        edges.append({"source": a, "target": b, "length": length})
        connected.add(b)

    # Додаємо ще трохи випадкових ребер
    for _ in range(extra_edges):
        a, b = random.sample(range(num_nodes), 2)
        if a != b:
            length = random.uniform(1, 20)
            edges.append({"source": a, "target": b, "length": length})

    graph = {"nodes": nodes, "edges": edges}
    os.makedirs("data", exist_ok=True)
    with open("data/graph.json", "w") as f:
        json.dump(graph, f, indent=2)
    print(f"[✓] Створено зв’язний graph.json з {num_nodes} вузлів і {len(edges)} ребер.")

if __name__ == "__main__":
    generate_connected_graph(num_nodes=50, extra_edges=150)
