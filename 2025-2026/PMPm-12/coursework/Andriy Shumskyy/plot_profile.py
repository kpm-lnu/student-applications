import numpy as np
import matplotlib.pyplot as plt

print("Читання даних з center_profile.csv...")

# Завантажуємо дані, пропускаючи перший рядок з назвами колонок
try:
    data = np.loadtxt("advanced_output/center_profile.csv", delimiter=",", skiprows=1)
    y = data[:, 0]
    u_x = data[:, 1]

    # Налаштовуємо полотно
    plt.figure(figsize=(8, 6))

    # Будуємо графік
    plt.plot(u_x, y, color='blue', linewidth=2.5, label='Швидкість $u_x$')

    # Додаємо академічну естетику
    plt.title("Профіль горизонтальної швидкості по центру каверни ($x=0.5$)", fontsize=14, pad=15)
    plt.xlabel("Швидкість $u_x$", fontsize=12)
    plt.ylabel("Координата $y$", fontsize=12)

    # Сітка та нульова вісь для наочності
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axvline(0, color='black', linewidth=1)
    plt.legend(fontsize=12)

    # Зберігаємо з високою роздільною здатністю
    plt.savefig("velocity_profile.png", dpi=300, bbox_inches='tight')
    print("Графік успішно збережено як velocity_profile.png!")

except FileNotFoundError:
    print("Помилка: Файл advanced_output/center_profile.csv не знайдено.")