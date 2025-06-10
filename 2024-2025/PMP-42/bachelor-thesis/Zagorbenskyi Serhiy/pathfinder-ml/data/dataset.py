import json
import numpy as np
from sklearn.model_selection import train_test_split

GRAPH_PATH = 'data/graph.json'
PATHS_PATH = 'data/paths.json'

def load_graph():
    with open(GRAPH_PATH, 'r') as f:
        data = json.load(f)

    graph = {}
    coords = {}

    for node in data['nodes']:
        node_id = node['id']
        coords[node_id] = (node['x'], node['y'])
        graph[node_id] = []

    for edge in data['edges']:
        graph[edge['source']].append((edge['target'], edge['length']))
        graph[edge['target']].append((edge['source'], edge['length']))  # якщо граф неорієнтований

    return graph, coords


def load_paths():
    with open(PATHS_PATH, 'r') as f:
        return json.load(f)


def load_dataset():
    graph, coords = load_graph()

    data = load_paths()

    X = []
    y = []

    for entry in data:
        start_id = entry['start']
        goal_id = entry['goal']
        path = entry['path']

        # Перевірка валідності індексів
        if start_id not in coords or goal_id not in coords:
            raise ValueError(f"start_id {start_id} or goal_id {goal_id} не знайдені у coords")

        goal_x, goal_y = coords[goal_id]

        for i in range(len(path) - 1):
            curr_node = path[i]
            next_node = path[i + 1]

            if curr_node not in coords or next_node not in coords:
                raise ValueError(f"Вузол {curr_node} або {next_node} не знайдені у coords")

            curr_x, curr_y = coords[curr_node]

            X.append([curr_x, curr_y, goal_x, goal_y])
            y.append(next_node)

    X = np.array(X)
    y = np.array(y)

    return X, y