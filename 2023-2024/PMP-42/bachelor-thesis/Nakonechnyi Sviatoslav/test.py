from random import choice, random, shuffle
from typing import List

import matplotlib.pyplot as plt

from Astar.Graph import Graph
from Astar.Point import Point


class Line:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def __str__(self):
        return f"Line({self.p1}, {self.p2})"

    def plot(self, color='red'):
        plt.plot([self.p1.x, self.p2.x], [self.p1.y, self.p2.y], color=color)

    def as_points(self):
        return [self.p1, self.p2]

    def as_neighbours(self):
        result = []
        points = [self.as_points(), reversed(self.as_points())]
        distance = lambda p1, p2: abs((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
        for point1, point2 in points:
            result.append([point1, (point2, distance(point1, point2))])
        return result

# generate random graph
class LinesGraph:
    def __init__(self, lines: List[Line]):
        self.lines = lines

    def visualize(self):
        """Visualize graph"""
        plt.figure()
        for line in self.lines:
            plt.plot([line.p1.x, line.p2.x], [line.p1.y, line.p2.y], color='blue')

    def to_neighbour_list(self):
        """Converts graph to neighbour list"""
        neighbour_list = {}
        for line in self.lines:
            for key, val in line.as_neighbours():
                if key not in neighbour_list:
                    neighbour_list[key] = []
                neighbour_list[key].append(val)
        return neighbour_list

    @staticmethod
    def from_neighbour_list(neighbour_list: dict):
        """Converts neighbour list to graph"""
        lines = []
        for key, values in neighbour_list.items():
            for value, distance in values:
                lines.append(Line(key, value))
        return LinesGraph(lines)

    @staticmethod
    def generate(grid_size=20, num_lines=-1):
        """Generate random connected graph with only vertical and horizontal lines"""
        lines = []
        intersection_points = set()

        # Generate a grid of points
        grid_points = [Point(i, j) for i in range(grid_size) for j in range(grid_size)]

        # Randomly shuffle the points
        shuffle(grid_points)

        # Initialize the set of connected points with the first point
        connected_points = {grid_points[0]}

        # Connect all points in the shuffled order with vertical or horizontal lines
        for i in range(1, len(grid_points)):
            current_point = grid_points[i]
            previous_point = choice(list(connected_points))

            # Check if the line already exists or overlaps with another line
            if any(line.p1 == current_point and line.p2 == previous_point or
                   line.p1 == previous_point and line.p2 == current_point for line in lines):
                continue

            # Check if the line would overlap with an existing line
            if any(line.p1.x == current_point.x == line.p2.x or
                   line.p1.y == current_point.y == line.p2.y for line in lines):
                continue

            # Connect either vertically or horizontally
            if current_point.x == previous_point.x or current_point.y == previous_point.y:
                lines.append(Line(current_point, previous_point))
                connected_points.add(current_point)

                # Check if it's a point of intersection
                if current_point.x != previous_point.x and current_point.y != previous_point.y:
                    intersection_points.add(current_point)

            # Check if enough lines have been generated
            if len(lines) == num_lines:
                break

        return LinesGraph(lines + [Line(point, point) for point in intersection_points])

    @staticmethod
    def from_points(points: List[Point]):
        lines = []
        for i in range(len(points) - 1):
            lines.append(Line(points[i], points[i + 1]))
        return LinesGraph(lines)

    def __str__(self):
        return f"LinesGraph: \n{self.lines}"

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    linesGraph = LinesGraph.generate()
    linesGraph.visualize()
    neighbours_lst = linesGraph.to_neighbour_list()

    graph1 = Graph(neighbours_lst)
    p_1 = choice(linesGraph.lines).p1
    p_2 = choice(linesGraph.lines).p1
    plt.plot(p_1.x, p_1.y, 'y*', markersize=14)
    plt.plot(p_2.x, p_2.y, 'g*', markersize=14)
    path = graph1.a_star_algorithm(p_1, p_2)
    found_path = LinesGraph.from_points(path)
    for i in found_path.lines:
        i.plot()
    plt.show()
