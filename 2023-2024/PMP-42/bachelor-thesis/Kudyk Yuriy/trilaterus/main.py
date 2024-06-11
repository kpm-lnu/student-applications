import numpy as np

def trilateration(p1, p2, p3, r1, r2, r3):
    # Координати трьох відомих точок
    P1 = np.array(p1)
    P2 = np.array(p2)
    P3 = np.array(p3)

    # Відстані до цих точок
    R1 = r1
    R2 = r2
    R3 = r3

    # Вектор між P1 і P2
    ex = (P2 - P1) / np.linalg.norm(P2 - P1)

    # Вектор між P1 і P3
    i = np.dot(ex, P3 - P1)
    ey = (P3 - P1 - i * ex) / np.linalg.norm(P3 - P1 - i * ex)

    # Вектор між P1 і P3 вздовж екс-спрямування
    ez = np.cross(ex, ey)

    d = np.linalg.norm(P2 - P1)
    j = np.dot(ey, P3 - P1)

    x = (R1**2 - R2**2 + d**2) / (2 * d)
    y = (R1**2 - R3**2 + i**2 + j**2) / (2 * j) - (i / j) * x
    z = np.sqrt(R1**2 - x**2 - y**2)

    # Обчислення координат невідомої точки
    P = P1 + x * ex + y * ey + z * ez

    return P

# Відомі точки
p1 = [0, 0, 0]
p2 = [100, 0, 0]
p3 = [50, 100, 0]

# Відстані до невідомої точки
r1 = 70.71
r2 = 70.71
r3 = 100.71

# Виклик функції
result = trilateration(p1, p2, p3, r1, r2, r3)
print("Координати невідомої точки:", result)
