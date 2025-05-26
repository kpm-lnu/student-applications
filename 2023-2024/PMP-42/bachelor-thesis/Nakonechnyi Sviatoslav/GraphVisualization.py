import networkx as nx
import matplotlib.pyplot as plt


# Defining a Class
class GraphVisualization:

    def __init__(self):
        self.visual = []

    def addEdge(self, a, b):
        """Adds new edge to the graph with vertices a and b"""
        temp = [a, b]
        self.visual.append(temp)

    def visualize(self):
        G = nx.DiGraph()
        G.add_edges_from(self.visual)
        nx.draw_networkx(G)
        plt.grid()
        plt.show()
