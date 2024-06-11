from queue import PriorityQueue

def dijkstra(edges, start_vertex):
    # Ініціалізація відстаней до всіх вершин як нескінченність
    D = {v: float('inf') for v in edges}
    D[start_vertex] = 0

    # Пріоритетна черга для обробки вершин
    pq = PriorityQueue()
    pq.put((0, start_vertex))

    # Множина для відстеження відвіданих вершин
    visited = set()

    while not pq.empty():
        (dist, current_vertex) = pq.get()
        
        # Якщо вершина вже відвідана, пропускаємо її
        if current_vertex in visited:
            continue

        # Додаємо вершину до відвіданих
        visited.add(current_vertex)

        # Обробляємо сусідів поточної вершини
        for neighbor, distance in edges[current_vertex].items():
            if neighbor not in visited:
                old_cost = D[neighbor]
                new_cost = D[current_vertex] + distance

                # Якщо знайдено коротший шлях, оновлюємо відстань та додаємо до черги
                if new_cost < old_cost:
                    pq.put((new_cost, neighbor))
                    D[neighbor] = new_cost

    return D
