from matplotlib import pyplot as plt

from Astar.Graph import Graph
from Astar.Point import Point

A = Point(-4.0, 2.0)
B = Point(-2.0, 0.0)
C = Point(-1.0, -3.0)
D = Point(2.0, 0.0)
a = [A, B, C, D]

adjac_lis = {
    A: [(B, 1), (C, 3), (D, 7)],
    B: [(D, 5)],
    C: [(D, 12)]
}
graph1 = Graph(adjac_lis)
print(A, B, C, D)
print(graph1.a_star_algorithm(A, D))

plt.plot([item.x for item in a], [item.y for item in a])
plt.show()
# g = GraphVisualization()
# g.addEdge(A, B)
# g.addEdge(A, C)
# g.addEdge(A, D)
# g.addEdge(B, D)
# g.addEdge(C, D)
# g.visualize()