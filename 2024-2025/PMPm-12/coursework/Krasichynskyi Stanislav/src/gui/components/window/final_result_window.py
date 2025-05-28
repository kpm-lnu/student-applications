import customtkinter as ctk
from matplotlib import pyplot as plt
import numpy as np

from gui.helpers.window_geometry_helper import center_window_to_display
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


import customtkinter as ctk
from matplotlib import pyplot as plt
import numpy as np

from gui.helpers.window_geometry_helper import center_window_to_display
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ResultsWindow(ctk.CTkToplevel):
    def __init__(self, boundary_indices, vertices, pressure, concentration) -> None:
        super().__init__()

        self.current_iteration = 1
        self.dt = 1

        self.boundary_indices = np.array(list(boundary_indices))
        self.vertices = vertices.copy()
        self.pressure = pressure
        self.concentration = concentration

        self.velocity = self.concentration + self.pressure

        self.title("Results Window")
        self.geometry(center_window_to_display(self, 900, 500))
        self.resizable(True, True)
        self.attributes("-topmost", True)

        self.iteration_label = ctk.CTkLabel(
            self, text=f"Current iteration: {self.current_iteration}"
        )
        self.iteration_label.pack(pady=10)

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.boundary_plot = self.ax.scatter(
            self.vertices[self.boundary_indices, 0],
            self.vertices[self.boundary_indices, 1],
            c="red",
            label="Boundary Points",
        )
        
        self.ax.set_xlim(-10, 10)  
        self.ax.set_ylim(-10, 10)  
        
        self.ax.set_title("Boundary Points")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.legend()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.change_iteration_button = ctk.CTkButton(
            self, text="Next Iteration", command=self.change_iteration
        )
        self.change_iteration_button.pack(pady=10)

    def compute_normals(self, points):
        normals = []
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]

            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            normal = normal / np.linalg.norm(normal)
            normals.append(normal)
        return np.array(normals)

    def update_positions(self):
        boundary_points = self.vertices[self.boundary_indices]
        normals = self.compute_normals(boundary_points)

        for i, index in enumerate(self.boundary_indices):
            movement = self.dt * self.velocity[index]
            self.vertices[index] += movement * normals[i]

    def change_iteration(self):
        self.current_iteration += 1
        self.update_positions()

        self.boundary_plot.set_offsets(self.vertices[self.boundary_indices])

        self.iteration_label.configure(
            text=f"Current iteration: {self.current_iteration}, dt: {self.dt}"
        )
        self.dt += 0.1

        self.canvas.draw()
