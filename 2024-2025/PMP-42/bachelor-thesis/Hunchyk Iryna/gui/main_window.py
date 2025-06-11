from PyQt5.QtWidgets import (
    QMainWindow, QPushButton, QWidget,
    QVBoxLayout, QLabel
)
from PyQt5.QtGui import QFont, QIcon, QPixmap
from PyQt5.QtCore import Qt
from gui.about_page import show_about_page
from gui.calculate_page import show_calculate_page
from gui.results_page import show_results_page
from gui.calculator import show_calculator
from PyQt5.QtCore import QTimer
from gui.testing_page import show_testing_page

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Система тестування задач FEM")
        self.setWindowIcon(QIcon("assets/logo.ico"))

        QTimer.singleShot(0, self.showMaximized)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)

        title_label = QLabel("System for Automated Testing of Practical Tasks")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 68px;
            font-weight: bold;
            color: #323;
        """)
        main_layout.addWidget(title_label)


        btn_about = QPushButton("Про програму")
        btn_about.clicked.connect(show_about_page)
        main_layout.addWidget(btn_about)

        btn_calc = QPushButton("Розрахунок FEM")
        btn_calc.clicked.connect(show_calculate_page)
        main_layout.addWidget(btn_calc)

        btn_results = QPushButton("Результати")
        btn_results.clicked.connect(show_results_page)
        main_layout.addWidget(btn_results)

        btn_tool = QPushButton("Калькулятор")
        btn_tool.clicked.connect(show_calculator)
        main_layout.addWidget(btn_tool)

        btn_test = QPushButton("Тестування")
        btn_test.clicked.connect(show_testing_page)
        main_layout.addWidget(btn_test)

        central_widget.setLayout(main_layout)
