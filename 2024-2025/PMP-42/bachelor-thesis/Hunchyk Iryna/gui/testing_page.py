# gui/testing_page.py
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QTextEdit
import subprocess

def show_testing_page():
    dialog = QDialog()
    dialog.setWindowTitle("Результати тестування")
    dialog.setMinimumSize(600, 400)

    layout = QVBoxLayout()

    output_box = QTextEdit()
    output_box.setReadOnly(True)
    layout.addWidget(output_box)

    def run_tests():
        try:
            result = subprocess.run(
                ["pytest", "tests", "-v"],
                capture_output=True,
                text=True
            )
            output_box.setPlainText(result.stdout + "\n" + result.stderr)
        except Exception as e:
            output_box.setPlainText(f"Помилка запуску тестів:\n{e}")

    btn_run = QPushButton("Запустити тести")
    btn_run.clicked.connect(run_tests)
    layout.addWidget(btn_run)

    dialog.setLayout(layout)
    dialog.exec_()
