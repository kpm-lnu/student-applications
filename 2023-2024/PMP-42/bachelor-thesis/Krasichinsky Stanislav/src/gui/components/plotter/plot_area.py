from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import PolyCollection
import numpy as np

from common.constants import (
    PLOTTER_HEIGTH_INCHES,
    PLOTTER_WIDTH_INCHES,
    PLOTTER_X_PLACEMENT,
    PLOTTER_Y_PLACEMENT,
)


class AreaPlotArea(FigureCanvasTkAgg):
    def __init__(self, parent) -> None:
        fig, ax = plt.subplots()
        fig.set_size_inches(PLOTTER_WIDTH_INCHES, PLOTTER_HEIGTH_INCHES)
        ax.axis("on")
        ax.set_aspect("equal", adjustable="datalim")
        plt.grid(True)

        super().__init__(fig, master=parent)
        self.ax = ax
        self.fig = fig

        self.get_tk_widget().place(relx=PLOTTER_X_PLACEMENT, rely=PLOTTER_Y_PLACEMENT)
        self.draw()

    def update_area_plot(self, parent, points):
        fig, ax = plt.subplots()
        fig.set_size_inches(PLOTTER_WIDTH_INCHES, PLOTTER_HEIGTH_INCHES)

        x = points[:, 0]
        y = points[:, 1]
        ax.scatter(x, y)

        ax.axis("on")
        plt.grid(True)

        self.ax.relim()
        self.ax.autoscale_view()

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().place(relx=PLOTTER_X_PLACEMENT, rely=PLOTTER_Y_PLACEMENT)

        parent.update()

    def update_area_triangulation(self, parent, triangulation_result):
        fig, ax = plt.subplots()
        fig.set_size_inches(PLOTTER_WIDTH_INCHES, PLOTTER_HEIGTH_INCHES)

        triangles = triangulation_result["triangles"]
        vertices = triangulation_result["vertices"]

        coll = PolyCollection(
            vertices[triangles], edgecolors="black", facecolors="none"
        )
        ax.add_collection(coll)

        ax.scatter(vertices[:, 0], vertices[:, 1], color="blue")

        annotation = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
        )

        annotation.set_visible(False)

        def update_poly(ind):
            colors = ["none"] * len(triangles)
            if ind is not None:
                colors[ind] = "lightgrey"
                annotation.set_text(f"Index: {ind}")
                annotation.set_visible(True)
            else:
                annotation.set_visible(False)
            coll.set_facecolor(colors)
            fig.canvas.draw_idle()

        def on_move(event):
            if event.inaxes == ax:
                x, y = np.array([event.xdata, event.ydata])
                for i, triangle in enumerate(triangles):
                    t = vertices[triangle]
                    P = np.array([x, y])
                    A = t[0]
                    B = t[1]
                    C = t[2]
                    v0 = B - A
                    v1 = C - A
                    v2 = P - A
                    d00 = np.dot(v0, v0)
                    d01 = np.dot(v0, v1)
                    d11 = np.dot(v1, v1)
                    d20 = np.dot(v2, v0)
                    d21 = np.dot(v2, v1)
                    denom = d00 * d11 - d01 * d01
                    v = (d11 * d20 - d01 * d21) / denom
                    w = (d00 * d21 - d01 * d20) / denom
                    u = 1.0 - v - w

                    if (u >= 0) and (v >= 0) and (w >= 0):
                        update_poly(i)
                        annotation.xy = (x, y)
                        return
                update_poly(None)

        fig.canvas.mpl_connect("motion_notify_event", on_move)
        ax.axis("on")
        plt.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().place(relx=PLOTTER_X_PLACEMENT, rely=PLOTTER_Y_PLACEMENT)

        parent.update()

        parent.tabs.update_mesh_ready_label()

    def clear_area(self, parent):
        fig, ax = plt.subplots()
        fig.set_size_inches(PLOTTER_WIDTH_INCHES, PLOTTER_HEIGTH_INCHES)

        ax.axis("on")
        plt.grid(True)

        self.ax.relim()
        self.ax.autoscale_view()

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().place(relx=PLOTTER_X_PLACEMENT, rely=PLOTTER_Y_PLACEMENT)

        parent.update()
