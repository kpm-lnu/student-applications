import numpy as np
from tensorflow.keras.models import load_model
from data.dataset import load_graph
import sys

def prepare_input(current_node, goal_node, coords):
    """Готує вхідний вектор [curr_x, curr_y, goal_x, goal_y]"""
    curr_x, curr_y = coords[current_node]
    goal_x, goal_y = coords[goal_node]
    return np.array([[curr_x, curr_y, goal_x, goal_y]], dtype=np.float32)

def predict_next_node(model, input_vec, node_ids):
    """Повертає наступний вузол, передбачений моделлю"""
    predictions = model.predict(input_vec, verbose=0)
    predicted_index = np.argmax(predictions[0])
    return node_ids[predicted_index]

def predict_path(model, start, goal, coords, node_ids, max_steps=100):
    """Будує маршрут від start до goal на основі моделі"""
    current = start
    path = [current]
    visited = set()
    visited.add(current)

    for _ in range(max_steps):
        if current == goal:
            break
        input_vec = prepare_input(current, goal, coords)
        next_node = predict_next_node(model, input_vec, node_ids)

        if next_node in visited:
            print("Цикл або застрягання! Зупинено.")
            break

        path.append(next_node)
        visited.add(next_node)
        current = next_node

    return path

def main():
    # Завантаження графа та координат
    graph, coords = load_graph()
    node_ids = list(coords.keys())

    # Завантаження моделі
    model = load_model('model/model.h5')  # заміни шлях, якщо інший

    # Вибір початкового та кінцевого вузлів
    start_node = node_ids[0]    # або конкретний вузол, напр. 'A'
    goal_node = node_ids[-1]    # або конкретний вузол, напр. 'Z'

    print(f"Пошук маршруту з {start_node} до {goal_node} за допомогою ML-моделі...")

    path = predict_path(model, start_node, goal_node, coords, node_ids)
    print("Отриманий маршрут:", path)

if __name__ == "__main__":
    main()
