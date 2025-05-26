import matplotlib.pyplot as plt
import numpy as np


class EllipticCurvePoint:
    def __init__(self, x, y, a, b, p):
        self.x = x
        self.y = y
        self.a = a
        self.b = b
        self.p = p

    def __eq__(self, other):
        if isinstance(other, EllipticCurvePoint):
            return (
                    self.x == other.x and
                    self.y == other.y and
                    self.a == other.a and
                    self.b == other.b and
                    self.p == other.p
            )
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __add__(self, other):
        if isinstance(other, EllipticCurvePoint):
            # Перевірка, чи точки належать одній еліптичній кривій
            if self.a != other.a or self.b != other.b or self.p != other.p:
                raise ValueError("Точки не належать одній еліптичній кривій")

            # Додавання точок на еліптичній кривій
            if self == other:
                # Перевірка, чи точка є нейтральним елементом
                if self.y == 0:
                    return EllipticCurvePoint(float('inf'), float('inf'), self.a, self.b, self.p)

                # Обчислення коефіцієнта наклона дотичної
                slope = ((3 * self.x ** 2 + 2 * self.a * self.x + self.b) * pow(2 * self.y, -1, self.p)) % self.p
            else:
                # Перевірка, чи точки знаходяться на вертикальній лінії
                if self.x == other.x:
                    return EllipticCurvePoint(float('inf'), float('inf'), self.a, self.b, self.p)

                # Обчислення коефіцієнта наклона прямої
                slope = ((other.y - self.y) * pow(other.x - self.x, -1, self.p)) % self.p

            # Обчислення координати x третьої точки
            x3 = (slope ** 2 - self.x - other.x) % self.p

            # Обчислення координати y третьої точки
            y3 = (slope * (self.x - x3) - self.y) % self.p

            # Повернення нової точки
            return EllipticCurvePoint(x3, y3, self.a, self.b, self.p)

        raise TypeError("Метод __add__ підтримує тільки об'єкти класу EllipticCurvePoint")

    def __str__(self):
        return f"({self.x}, {self.y})"


p = 23  # Поле простого числа
# Створення точок на еліптичній кривій
p1 = EllipticCurvePoint(3, 10, 1, 1, p)
p2 = EllipticCurvePoint(9, 7, 1, 1, p)

# Додавання точок
p3 = p1 + p2
print(f"p1 + p2 = ({p3.x}, {p3.y})")

# Порівняння точок
print(f"p1 == p2: {p1 == p2}")
print(f"p1 != p2: {p1 != p2}")

# Генерація координат для графіку еліптичної кривої
x = np.linspace(0, p - 1, 1000)
y = np.linspace(0, p - 1, 1000)
X, Y = np.meshgrid(x, y)
Z = (Y ** 2 - X ** 3 - p1.a * X - p1.b) % p

# Графік еліптичної кривої та точок
plt.contour(X, Y, Z, [0], colors='b')  # Еліптична крива
plt.plot(p1.x, p1.y, 'ro', label='P1')  # Точка P1
plt.plot(p2.x, p2.y, 'ro', label='P2')  # Точка P2
plt.plot(p3.x, p3.y, 'ro', label='P3')  # Точка P3

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.title('Elliptic Curve')
plt.show()
