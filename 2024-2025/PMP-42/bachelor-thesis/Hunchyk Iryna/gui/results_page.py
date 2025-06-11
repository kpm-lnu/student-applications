from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPushButton, QMessageBox
)
from core.plotting import plot_trisurf
from core.mesh_generator import generate_mesh
from core.exact_solution import exact_solution_vector
from core.error_metrics import l2_error, max_abs_error
import numpy as np
import os


def show_results_page():
    dialog = QDialog()
    dialog.setWindowTitle("Результати")
    dialog.setFixedSize(400, 250)

    layout = QVBoxLayout()

    label = QLabel("Результати останнього розрахунку:")
    layout.addWidget(label)

    plot_btn = QPushButton("Показати графік u(x, y)")
    layout.addWidget(plot_btn)

    result_label = QLabel("")
    layout.addWidget(result_label)

    def show_graph():
        try:
            data_path = os.path.join("data", "output.csv")
            data = np.loadtxt(data_path, delimiter=",", skiprows=1)
            u_h = data[:, 0]
            u_exact = data[:, 1]

            nodes, elements, _ = generate_mesh(10, 10)  # припущення: nx=ny=10
            plot_trisurf(nodes, elements, u_h)

            l2 = l2_error(u_h, u_exact)
            maxerr = max_abs_error(u_h, u_exact)

            result_label.setText(f"L2-похибка: {l2:.4e}\n"
                                 f"Максимальна похибка: {maxerr:.4e}")

        except Exception as e:
            QMessageBox.warning(dialog, "Помилка", f"Не вдалося зчитати файл результатів:\n{str(e)}")

    plot_btn.clicked.connect(show_graph)

    dialog.setLayout(layout)
    dialog.exec_()
