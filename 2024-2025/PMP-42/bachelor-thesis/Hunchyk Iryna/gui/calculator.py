from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLineEdit,
    QPushButton, QGridLayout
)


def show_calculator():
    dialog = QDialog()
    dialog.setWindowTitle("Калькулятор")
    dialog.setFixedSize(300, 350)

    layout = QVBoxLayout()

    input_field = QLineEdit()
    input_field.setReadOnly(True)
    input_field.setObjectName("calc-display")
    layout.addWidget(input_field)

    buttons = [
        '7', '8', '9', '/',
        '4', '5', '6', '*',
        '1', '2', '3', '-',
        '0', '.', 'C', '+',
        '(', ')', '=', ''
    ]

    grid = QGridLayout()
    row, col = 0, 0

    def button_clicked(text):
        if text == '=':
            try:
                result = str(eval(input_field.text()))
                input_field.setText(result)
            except Exception:
                input_field.setText("Помилка")
        elif text == 'C':
            input_field.clear()
        else:
            input_field.setText(input_field.text() + text)

    for label in buttons:
        if label:
            btn = QPushButton(label)
            btn.setFixedSize(50, 40)
            btn.setObjectName("calc-btn")
            btn.clicked.connect(lambda _, t=label: button_clicked(t))
            grid.addWidget(btn, row, col)
        col += 1
        if col > 3:
            col = 0
            row += 1

    layout.addLayout(grid)
    dialog.setLayout(layout)
    dialog.exec_()
