from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QLineEdit,
    QPushButton, QMessageBox, QHBoxLayout
)
from core.mesh_generator import generate_mesh
from core.fem_solver import assemble_global_matrix, apply_dirichlet, solve_system
from core.exact_solution import exact_solution_vector
from core.error_metrics import l2_error
from core.plotting import plot_mesh

import numpy as np
import os


def show_calculate_page():
    dialog = QDialog()
    dialog.setWindowTitle("Розрахунок FEM")
    dialog.setFixedSize(400, 300)

    layout = QVBoxLayout()

    layout.addWidget(QLabel("Введіть параметри сітки (nx, ny):"))

    nx_input = QLineEdit()
    ny_input = QLineEdit()
    nx_input.setPlaceholderText("nx (наприклад, 10)")
    ny_input.setPlaceholderText("ny (наприклад, 10)")

    hbox = QHBoxLayout()
    hbox.addWidget(nx_input)
    hbox.addWidget(ny_input)
    layout.addLayout(hbox)

    run_button = QPushButton("Запустити розрахунок")
    layout.addWidget(run_button)

    def run_fem():
        try:
            nx = int(nx_input.text())
            ny = int(ny_input.text())
        except ValueError:
            QMessageBox.warning(dialog, "Помилка", "nx та ny мають бути цілими числами")
            return

        nodes, elements, boundary_nodes = generate_mesh(nx, ny)

        K, F = assemble_global_matrix(nodes, elements)

        K, F = apply_dirichlet(K, F, boundary_nodes)

        u_h = solve_system(K, F)

        u_exact = exact_solution_vector(nodes)

        err = l2_error(u_h, u_exact)

        mesh_path = os.path.join("data", "mesh_preview.png")
        plot_mesh(nodes, elements, save_path=mesh_path, show=False)

        out_path = os.path.join("data", "output.csv")
        np.savetxt(out_path, np.column_stack((u_h, u_exact)), delimiter=",", header="u_h,u_exact", comments='')

        QMessageBox.information(
            dialog,
            "Готово",
            f"Розрахунок завершено успішно.\n"
            f"Похибка (L2): {err:.4e}\n"
            f"Сітка збережена у data/mesh_preview.png"
        )

    run_button.clicked.connect(run_fem)
    dialog.setLayout(layout)
    dialog.exec_()
