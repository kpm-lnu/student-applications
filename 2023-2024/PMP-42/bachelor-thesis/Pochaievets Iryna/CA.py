from itertools import product
from math import ceil
from dijkstra import dijkstra
from astar import astar
import numpy as np
from random import shuffle, choice, random


class CACell(object):
    EMPTY_STATE = 0
    PERSON_STATE = 1
    OBSTACLE_STATE = 2
    EXIT_STATE = 3

    def __init__(self, state, properties):
        self.state = state
        self.properties = properties


class CARoom(object):
    PERSON_M2_MAX = 6
    
    # Ініціалізує кімнату з вказаною шириною та висотою
    def __init__(self, room_w=5, room_h=5):
        self._cell_size = 0.5
        self.room_w = room_w
        self.room_h = room_h
        self.nx = ceil(self.room_w/self._cell_size)
        self.ny = ceil(self.room_h/self._cell_size)
        self.area = self.room_w * self.room_h
        self._cells = {(r, c): CACell(CACell.EMPTY_STATE, {}) for r, c in product(range(self.ny), range(self.nx))}
        self._exits = []
        self._graph = None
        self._dijkstra_map = None
        self.moore_neighborhood = {}

        for r in range(self.ny):
            for c in range(self.nx):
                self.moore_neighborhood[(r, c)] = self._moore_neighb(r, c)

    # Метод для отримання сусідніх клітин за правилом Мура
    def _moore_neighb(self, r, c):
        cells = []
        for i, j in [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1), (r + 1, c + 1), (r + 1, c - 1), (r - 1, c - 1), (r - 1, c + 1)]:
            if 0 <= j < self.nx and 0 <= i < self.ny:
                cells.append((i, j))
        return cells

    # Метод для отримання клітини за її координатами
    def getCell(self, pos):
        return self._cells[pos]

    # Метод для встановлення стану клітини за її координатами
    def setCell(self, pos, cell):
        self._cells[pos] = cell

        if cell.state == CACell.EXIT_STATE:
            self._exits.append(pos)
            self._graph = None
            self._dijkstra_map = None
        elif cell.state == CACell.OBSTACLE_STATE:
            self._graph = None
            self._dijkstra_map = None    

    # Метод для отримання всіх виходів з кімнати
    def getExits(self):
        return self._exits

    # Метод для отримання мапи Дейкстри
    def get_dijkistra_map(self):
        if self._dijkstra_map is not None:
            return self._dijkstra_map

        g = self.get_graph()
        D_min = {}

        for exit in self.getExits():            
            D = dijkstra(g, exit)
            

            for vertex, dist in D.items():
                if vertex not in D_min:
                    D_min[vertex] = dist
                else:
                    D_min[vertex] = min(D_min[vertex], dist)

        vertex = list(D_min.keys())
        dist = list(D_min.values())
        order = np.argsort(dist)

        self._dijkstra_map = D_min, [(vertex[i], dist[i]) for i in order]
        return self._dijkstra_map

    # Метод для отримання графа кімнати
    def get_graph(self, max_dist2=2):
        if self._graph is not None:
            return self._graph

        vertex = self._cells.keys()
        w = {}

        for vi in vertex:
            if self.getCell(vi).state == CACell.OBSTACLE_STATE:
                continue

            for vj in self.moore_neighborhood[vi]:
                if vi == vj:
                    continue
            
                if self.getCell(vj).state == CACell.OBSTACLE_STATE:
                    continue

                dx = (vi[0] - vj[0])**2
                dy = (vi[1] - vj[1])**2
                d2 = dx**2 + dy**2

                if d2 <= max_dist2:                
                    w.setdefault(vi, {})[vj] = d2**1

        self._graph = w
        return self._graph
        


    # Метод для виконання одного кроку симуляції
    def step_parallel(self):
        evac = 0
        djikistra_map, djikistra_map_sorted = self.get_dijkistra_map()        
        new_state = {v:[] for v in djikistra_map}

        for cell_idx, dist_to_exit in djikistra_map_sorted: 
            if self.getCell(cell_idx).state == CACell.PERSON_STATE:
                empty_exit_candidates = []                

                for i, j in self.moore_neighborhood[cell_idx]:                
                    neighbour_state = self.getCell((i, j)).state
                    
                    if (neighbour_state == CACell.EMPTY_STATE or neighbour_state == CACell.EXIT_STATE) and djikistra_map[(i, j)] <= dist_to_exit:                        
                        empty_exit_candidates.append((i, j))                        

                if len(empty_exit_candidates) > 0:
                    panic_prob = self.getCell(cell_idx).properties['panic_prob']                    
                    if random() < panic_prob:
                        continue
                    
                    neighbour_dist_exit = [djikistra_map[pos] for pos in empty_exit_candidates]                        
                    min_dist = np.min(neighbour_dist_exit)                        
                    options = [pos for pos in empty_exit_candidates if djikistra_map[pos] == min_dist]                    
                    i, j = choice(options)
                    new_state[(i, j)].append(cell_idx)

        # Перемішування клітин для випадкового вирішення конфліктів        
        cell_keys = list(new_state.keys())
        shuffle(cell_keys)
        new_state = {k:new_state[k] for k in cell_keys}

        for idx, person_idx in new_state.items():
            if not person_idx:
                continue            
            if self.getCell(idx).state == CACell.EXIT_STATE:                
                idx_previous = choice(person_idx)
                self.setCell(idx_previous, CACell(CACell.EMPTY_STATE, {}))                    
                evac += 1                
            else:
                if len(person_idx) == 1:
                    idx_previous = person_idx[0]                    
                    person = self.getCell(idx_previous)                
                    self.setCell(idx_previous, CACell(CACell.EMPTY_STATE, {}))                    
                    self.setCell(idx, person)                
                elif len(person_idx) > 1:
                    idx_previous = choice(person_idx)                        
                    person = self.getCell(idx_previous)
                    self.setCell(idx_previous, CACell(CACell.EMPTY_STATE, {}))                    
                    self.setCell(idx, person)
                
        return evac

"""
    # Метод для виведення стану кімнати
    def __str__(self):
        s = ''
        for r in range(self.ny):
            for c in range(self.nx):
                if self.getCell((r, c)).state == CACell.EMPTY_STATE:
                    s += '_ '
                elif self.getCell((r, c)).state == CACell.OBSTACLE_STATE:
                    s += '# '
                elif self.getCell((r, c)).state == CACell.PERSON_STATE:
                    s += '* '
                elif self.getCell((r, c)).state == CACell.EXIT_STATE:
                    s += 'E '
            s += '\n'
        return s
"""

if __name__ == '__main__':
    room_w = 5
    room_h = 5
    room = CARoom(room_w, room_h)

    # Встановлення зовнішніх верхньої та нижньої стін
    for row in [0, room.ny - 1]:
        for col in range(room.nx):
            room.setCell((row, col), CACell(CACell.OBSTACLE_STATE, {}))

    # Встановлення зовнішніх лівої та правої стін
    for col in [0, room.nx - 1]:
        for row in range(room.ny):            
            room.setCell((row, col), CACell(CACell.OBSTACLE_STATE, {}))

    # Додавання блочного перешкоди в центрі кімнати
    obstacle_size = 4
    col = int(room.nx / 2)
    row = 1 + int(room.ny / 2) - obstacle_size
    row_cols = product(range(row, row + obstacle_size), range(col, col + obstacle_size))

    for r_c in row_cols:    
        room.setCell(r_c, CACell(CACell.OBSTACLE_STATE, {}))

    # Додавання виходу
    half_rows = int(room.ny / 2)
    room.setCell((half_rows, room.nx - 1), CACell(CACell.EXIT_STATE, {}))

    # Додавання сталого числа людей, наприклад, 100
    people = 500
    empty_cells = [(i, j) for i in range(room.ny) for j in range(room.nx) if room.getCell((i, j)).state == CACell.EMPTY_STATE]

    # Перевірити, чи достатньо порожніх клітин для розміщення всіх людей
    if len(empty_cells) < people:
        raise ValueError("Недостатньо місця для розміщення всіх людей.")

    # Випадково перемішати список порожніх клітин і вибрати 100 з них
    shuffle(empty_cells)
    selected_cells = empty_cells[:people]

    # Додати людей до вибраних клітин
    for pos in selected_cells:
        room.setCell(pos, CACell(CACell.PERSON_STATE, {'panic_prob': 0.001}))

    evacuees = 0
    running = True
    print(room)

    while running:
        evacuees += room.step_parallel()
        print(room)
        running = evacuees < people
