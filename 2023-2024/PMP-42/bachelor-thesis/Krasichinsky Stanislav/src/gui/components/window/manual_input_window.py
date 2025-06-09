from matplotlib import pyplot as plt
import numpy as np
from gui.helpers.window_geometry_helper import center_window_to_display
import customtkinter as ctk
import tkinter as tk


class ManualInputWindow(ctk.CTkToplevel):
    def __init__(self, parent) -> None:
        super().__init__()
        self.parent = parent

        self.title("Manual input")
        self.geometry(center_window_to_display(self, 700, 700))
        self.resizable(False, False)
        self.attributes("-topmost", True)

        self.canvas = ctk.CTkCanvas(self, width=700, height=600, bg="white")
        self.canvas.pack()

        self.points = []
        self.lines = []

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        button_frame = ctk.CTkFrame(self)
        button_frame.pack(fill=tk.X, ipadx=5, ipady=5)

        btn_get_points = ctk.CTkButton(
            button_frame, text="Get Points", command=self.get_points
        )
        btn_get_points.pack(side=tk.LEFT, padx=10, pady=10)

        btn_clear = ctk.CTkButton(button_frame, text="Clear", command=self.clear)
        btn_clear.pack(side=tk.RIGHT, padx=10, pady=10)

    def paint(self, event):
        x2, y2 = event.x, event.y
        if self.points:
            x1, y1 = self.points[-1]
            self.lines.append(
                self.canvas.create_line(x1, y1, x2, y2, fill="black", width=2)
            )
        self.points.append((x2, y2))

    def reset(self, event):
        self.points.append((event.x, event.y))

    ## TODO. Add abillity to change down_size and scale
    def get_points(self, scale=3):
        points = np.array(self.points)
        points[:, 1] = self.canvas.winfo_height() - points[:, 1]

        min_vals = points.min(axis=0)
        max_vals = points.max(axis=0)
        points = (points - min_vals) / (max_vals - min_vals) * scale

        if len(points) > 1:
            segments = np.array([(i, i + 1) for i in range(len(points) - 1)])
        else:
            segments = np.array([])
                    
        self.parent.area_boundary.points = points
        self.parent.area_boundary.segments = segments

        self.parent.plot_area.update_area_plot(self.parent, points)

        if len(self.parent.area_boundary.points) != 0:
            self.parent.tabs.update_area_ready_label()

    def clear(self):
        self.canvas.delete("all")
        self.points = []
        self.lines = []
