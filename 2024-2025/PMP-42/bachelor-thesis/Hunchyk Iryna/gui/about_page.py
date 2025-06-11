from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QFont


def show_about_page():
    dialog = QDialog()
    dialog.setWindowTitle("Про програму")
    dialog.setFixedSize(550, 680)

    layout = QVBoxLayout()

    label_univ = QLabel("Факультет прикладної математики та інформатики\n"
                        "Національний університет імені Івана Франка")
    label_univ.setFont(QFont("Arial", 11))
    label_univ.setStyleSheet("margin-bottom: 15px;")
    layout.addWidget(label_univ)

    label_course = QLabel("Проєкт з дисципліни\n"
                          "«Чисельні методи математичної фізики»\n"
                          "На тему:")
    label_course.setFont(QFont("Arial", 11))
    label_course.setStyleSheet("margin-bottom: 10px;")
    layout.addWidget(label_course)

    label_topic = QLabel("«Побудова системи для автоматизованого тестування практичних завдань»")
    label_topic.setFont(QFont("Arial", 12, QFont.Bold))
    label_topic.setStyleSheet("margin-bottom: 20px;")
    label_topic.setWordWrap(True)
    layout.addWidget(label_topic)

    label_author = QLabel("Виконала студентка групи ПМп-32\nГунчик Ірина")
    label_author.setFont(QFont("Arial", 11))
    label_author.setStyleSheet("margin-bottom: 20px;")
    layout.addWidget(label_author)

    label_desc = QLabel(
        "Цей додаток реалізує систему автоматизованого тестування задач з курсу "
        "«Чисельні методи математичної фізики». Основна мета — спростити процес "
        "перевірки практичних завдань, які розв’язуються методом скінченних елементів.\n\n"
        "Програма дозволяє:\n"
        "— виконувати розрахунки розв’язків крайових задач;\n"
        "— автоматично будувати сітки та матриці МСЕ;\n"
        "— перевіряти точність наближеного розв’язку;\n"
        "— запускати тести для контролю правильності обчислень.\n\n"
        "Простий інтерфейс дозволяє студентам інтерактивно працювати з задачами та "
        "оцінювати результати власних рішень у зручній формі."
    )
    label_desc.setFont(QFont("Arial", 10))
    label_desc.setWordWrap(True)
    layout.addWidget(label_desc)

    close_button = QPushButton("Закрити")
    close_button.clicked.connect(dialog.close)
    layout.addWidget(close_button)

    dialog.setLayout(layout)
    dialog.exec_()
