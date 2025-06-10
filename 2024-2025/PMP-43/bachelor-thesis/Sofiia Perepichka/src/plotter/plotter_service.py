import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np

class PlotterService:
    
    @staticmethod
    def plot_mesh(triangulation_result: dict) -> None:
        _, ax = plt.subplots()
        triangles = triangulation_result["triangles"]
        vertices = triangulation_result["vertices"]

        coll = PolyCollection(vertices[triangles], edgecolors="black", facecolors="none")
        ax.add_collection(coll)

        ax.scatter(vertices[:, 0], vertices[:, 1], color="magenta", s=4.5)
        plt.margins(0.1)
        plt.gca().set_aspect('equal')
        plt.show()

    @staticmethod
    def plot_solution(vertices: np.ndarray, C: np.ndarray, S: np.ndarray, title="Solution") -> None:
    
        X = vertices[:, 0]
        Y = vertices[:, 1]

        fig = plt.figure(figsize=(12, 6))
        

        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        surf1 = ax1.plot_trisurf(X, Y, C, cmap="viridis")
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
        ax1.set_title(title + " (С)")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("C value")
        

        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        surf2 = ax2.plot_trisurf(X, Y, S, cmap="viridis")
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
        ax2.set_title(title + " (S)")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("S value")
        
        plt.tight_layout()
        plt.show()