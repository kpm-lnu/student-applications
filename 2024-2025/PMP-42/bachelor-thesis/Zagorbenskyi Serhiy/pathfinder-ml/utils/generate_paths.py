import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
from data.dataset import load_graph
from utils.dijkstra import find_shortest_path  # Якщо у тебе є окрема реалізація

def generate_paths(num_paths=5000):
    graph, _ = load_graph()
    all_nodes = list(graph.keys())
    paths = []

    attempts = 0
    while len(paths) < num_paths and attempts < num_paths * 10:
        start, end = random.sample(all_nodes, 2)
        path = find_shortest_path(graph, int(start), int(end))
        if path:
            paths.append({
                "start": int(start),
                "goal": int(end),     # ← гарантуємо наявність 'goal'
                "path": path
            })
        attempts += 1

    with open("data/paths.json", "w") as f:
        json.dump(paths, f, indent=2)

    print(f"[✓] Згенеровано {len(paths)} шляхів у data/paths.json")
    if paths:
        print(f"Тестовий шлях {paths[0]['start']} -> {paths[0]['goal']}: {paths[0]['path']}")

if __name__ == "__main__":
    generate_paths(1000)
