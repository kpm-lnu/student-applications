import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import Delaunay
import networkx as nx
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union


class ObstacleAvoidancePathPlanner:
    def __init__(self, start, goal, obstacles, map_size=15, num_points=10):
        self.start = start
        self.goal = goal
        self.obstacles = [Polygon(obs) for obs in obstacles]
        self.obstacles_union = unary_union(self.obstacles)
        self.map_size = map_size
        self.num_points = num_points
        self.fig, self.ax = plt.subplots(figsize=(10, 10))

    def is_point_in_free_space(self, point):
        return not self.obstacles_union.contains(Point(point))

    def generate_sample_points(self):
        points = []
        while len(points) < self.num_points:
            x = np.random.uniform(0, self.map_size)
            y = np.random.uniform(0, self.map_size)
            if self.is_point_in_free_space((x, y)):
                points.append([x, y])
        points.extend([self.start, self.goal])
        return np.array(points)

    def constrained_delaunay_triangulation(self, points):
        tri = Delaunay(points)
        valid_triangles = []
        for simplex in tri.simplices:
            triangle = Polygon(points[simplex])
            if not triangle.intersects(self.obstacles_union):
                valid_triangles.append(simplex)
        return points, np.array(valid_triangles)

    def build_centroid_graph(self, points, triangles):
        G = nx.Graph()
        centroids = [np.mean(points[tri], axis=0) for tri in triangles]
        for i, c in enumerate(centroids):
            G.add_node(i, pos=c)
        for i in range(len(triangles)):
            for j in range(i + 1, len(triangles)):
                if len(set(triangles[i]) & set(triangles[j])) >= 2:
                    dist = np.linalg.norm(centroids[i] - centroids[j])
                    G.add_edge(i, j, weight=dist)
        return G, centroids

    def find_closest_centroids(self, point, centroids):
        return np.argsort([np.linalg.norm(np.array(point) - np.array(c)) for c in centroids])

    def find_shortest_path(self, G, centroids):
        start_idx = self.find_closest_centroids(self.start, centroids)
        goal_idx = self.find_closest_centroids(self.goal, centroids)
        shortest_path = None
        min_length = float('inf')
        for s in start_idx[:3]:
            for g in goal_idx[:3]:
                try:
                    path = nx.shortest_path(G, source=s, target=g, weight='weight')
                    length = nx.shortest_path_length(G, source=s, target=g, weight='weight')
                    if length < min_length:
                        shortest_path = path
                        min_length = length
                except nx.NetworkXNoPath:
                    continue
        return [centroids[i] for i in shortest_path] if shortest_path else None

    def smooth_path(self, path):
        if not path or len(path) <= 2:
            return path
        smooth = [path[0]]
        for i in range(1, len(path) - 1):
            line = LineString([smooth[-1], path[i + 1]])
            if not line.intersects(self.obstacles_union):
                continue
            smooth.append(path[i])
        smooth.append(path[-1])
        return smooth

    def plan_path(self):
        points = self.generate_sample_points()
        points, triangles = self.constrained_delaunay_triangulation(points)
        G, centroids = self.build_centroid_graph(points, triangles)
        path = self.find_shortest_path(G, centroids)
        
        if not path:
            return None, points, triangles, False, 0
        
        full_path = [self.start] + path + [self.goal]
        smoothed_path = self.smooth_path(full_path)
        
        total_distance = 0
        for i in range(len(smoothed_path)-1):
            p1 = smoothed_path[i]
            p2 = smoothed_path[i+1]
            total_distance += np.linalg.norm(np.array(p1) - np.array(p2))
        
        return smoothed_path, points, triangles, True, total_distance

    def visualize(self, path, points, triangles, path_exists, total_distance=None):
        self.ax.clear()
        
        for poly in self.obstacles:
            x, y = poly.exterior.xy
            self.ax.fill(x, y, alpha=0.5, fc='red', ec='black', label='Перешкоди')
        
        for tri in triangles:
            t = patches.Polygon(points[tri], closed=True, fill=False, edgecolor='blue', alpha=0.3, label='Трикутники Делоне')
            self.ax.add_patch(t)
        
        self.ax.plot(points[:, 0], points[:, 1], 'o', color='blue', markersize=3, label='Випадкові точки')
        
        if path_exists:
            x, y = zip(*path)
            self.ax.plot(x, y, 'go-', linewidth=2, markersize=8, label='Оптимальний шлях')
            if total_distance is not None:
                self.fig.text(0.5, 0.02, f'Загальна довжина маршруту: {total_distance:.2f} одиниць', 
                            fontsize=12, color='green', ha='center', va='bottom',
                            transform=self.fig.transFigure)
        else:
            self.fig.text(0.5, 0.02, 'Шлях не знайдено', fontsize=14, color='red', ha='center', va='bottom',
                        transform=self.fig.transFigure)
        
        self.ax.plot(*self.start, 'yo', markersize=12, label='Стартова точка')
        self.ax.plot(*self.goal, 'mo', markersize=12, label='Кінцева точка')
        
        self.ax.set_xlim(0, self.map_size)
        self.ax.set_ylim(0, self.map_size)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.3, 1))
        
        plt.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.8)
        plt.show()
