import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use("TkAgg")

# Створюємо координати для чотирьох вершин прямокутника
rect_points = np.array([[0, 0], [10, 0], [10, 5], [0, 5]])


# Функція для створення рівних трикутників в прямокутнику
def create_equal_triangles(width, height, rows, cols):
    x = np.linspace(0, width, cols + 1)
    y = np.linspace(0, height, rows + 1)
    xv, yv = np.meshgrid(x, y)
    points = np.vstack([xv.ravel(), yv.ravel()]).T
    triangles = []
    for i in range(rows):
        for j in range(cols):
            p1 = i * (cols + 1) + j
            p2 = i * (cols + 1) + (j + 1)
            p3 = (i + 1) * (cols + 1) + j
            triangles.append([p1, p2, p3])
            p1 = i * (cols + 1) + (j + 1)
            p2 = (i + 1) * (cols + 1) + (j + 1)
            p3 = (i + 1) * (cols + 1) + j
            triangles.append([p1, p2, p3])
    return points, triangles


# Функція для мутації сітки з новими граничними умовами
def mutate(points, triangles, rect_points, mutation_strength=0.5):
    new_points = points.copy()

    for i, point in enumerate(points):
        x, y = point

        # Перевірка на кутові точки (все 4 кути прямокутника)
        if (x == rect_points[0][0] and y == rect_points[0][1]) or \
                (x == rect_points[1][0] and y == rect_points[1][1]) or \
                (x == rect_points[2][0] and y == rect_points[2][1]) or \
                (x == rect_points[3][0] and y == rect_points[3][1]):
            continue  # Пропускаємо кутові точки, вони не повинні рухатись

        # Точки на лівій чи правій межі можуть рухатись тільки по осі Y
        if x == rect_points[0][0] or x == rect_points[1][0]:  # Ліва і права межа
            direction = np.random.randn(1)  # Випадковий напрямок по осі Y
            direction /= np.abs(direction)  # Нормалізуємо напрямок
            move_distance = np.random.uniform(-mutation_strength, mutation_strength)
            new_points[i][1] += direction * move_distance  # Зміщуємо тільки по осі Y

        # Точки на верхній і нижній межах можуть рухатись тільки по осі X
        elif y == rect_points[0][1] or y == rect_points[2][1]:  # Верхня і нижня межа
            direction = np.random.randn(1)  # Випадковий напрямок по осі X
            direction /= np.abs(direction)  # Нормалізуємо напрямок
            move_distance = np.random.uniform(-mutation_strength, mutation_strength)
            new_points[i][0] += direction * move_distance  # Зміщуємо тільки по осі X

        # Внутрішні точки (не на межах і не на кутах) можуть рухатись в будь-якому напрямку
        else:
            direction = np.random.randn(2)
            direction /= np.linalg.norm(direction)
            move_distance = np.random.uniform(-mutation_strength, mutation_strength)
            new_points[i] += direction * move_distance

    return new_points


# Функція для обчислення пристосованості для одного трикутника
def triangle_fitness(triangle, points):
    p1, p2, p3 = points[triangle]
    side_lengths = [np.linalg.norm(p1 - p2), np.linalg.norm(p2 - p3), np.linalg.norm(p3 - p1)]
    mean_side_length = np.mean(side_lengths)
    deviations = [abs(side - mean_side_length) for side in side_lengths]
    fitness = np.sum(deviations)
    return fitness


# Функція для обробки "поганих трикутників"
def handle_bad_fitness(fitness, sides, threshold=0.553, scale_factor=10, multiplier=10):
    # Перевірка сторін трикутників
    count = 0
    for side in sides:
        if side > np.sqrt(2) or side < 1:
            count += 1
    if count > 0:
        fitness = fitness * multiplier  # Множимо на 10, якщо сторона більше за корінь з 2 або менше за 1

    # Якщо пристосованість більша за поріг, підвищуємо її значення (як штраф)
    if fitness > threshold :
        return fitness * scale_factor  # Чим більше відхилення, тим сильніше штрафуємо
    return fitness


# Функція для обчислення пристосованості сітки
def grid_fitness(points, triangles):
    total_fitness = 0
    for triangle in triangles:
        # Отримуємо сторони трикутника
        p1, p2, p3 = points[triangle]
        sides = [np.linalg.norm(p1 - p2), np.linalg.norm(p2 - p3), np.linalg.norm(p3 - p1)]

        # Обчислюємо пристосованість для цього трикутника
        fitness = triangle_fitness(triangle, points)

        # Обробляємо погані трикутники з урахуванням їх сторін
        fitness = handle_bad_fitness(fitness, sides)

        # Додаємо результат до загальної пристосованості сітки
        total_fitness += fitness

    return total_fitness


# Функція для обчислення h (максимальна довжина сторони)
def calculate_h(triangles, points):
    h_max = 0
    for triangle in triangles:
        p1, p2, p3 = points[triangle]
        sides = [np.linalg.norm(p1 - p2), np.linalg.norm(p2 - p3), np.linalg.norm(p3 - p1)]
        h_max = max(h_max, max(sides))
    return h_max


# Функція для обчислення θ (мінімальний кут трикутника)
def calculate_theta(triangles, points):
    theta_min = np.inf
    for triangle in triangles:
        p1, p2, p3 = points[triangle]
        a = np.linalg.norm(p2 - p3)
        b = np.linalg.norm(p1 - p3)
        c = np.linalg.norm(p1 - p2)
        # Знаходимо кути за допомогою теореми косинусів
        angle_A = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
        angle_B = np.arccos((a ** 2 + c ** 2 - b ** 2) / (2 * a * c))
        angle_C = np.pi - angle_A - angle_B
        theta_min = min(theta_min, angle_A, angle_B, angle_C)
    return theta_min


# Функція зворотної мутації з контролем пристосованості
# Функція зворотної мутації з контролем пристосованості
def reverse_mutate_with_fitness_check(points, triangles, rect_points, rows, cols, mutation_strength=0.1):
    new_points = points.copy()
    current_fitness = grid_fitness(new_points, triangles)

    for r in range(rows + 1):  # Ітерація по всіх точках, включаючи межі
        for c in range(cols + 1):  # Ітерація по всіх точках, включаючи межі
            i = r * (cols + 1) + c  # Визначаємо індекс точки в сітці
            x, y = new_points[i]  # Отримуємо координати точки

            # Пропускаємо тільки кутові точки (вони не змінюються)
            if (x == rect_points[0][0] and y == rect_points[0][1]) or \
               (x == rect_points[1][0] and y == rect_points[1][1]) or \
               (x == rect_points[2][0] and y == rect_points[2][1]) or \
               (x == rect_points[3][0] and y == rect_points[3][1]):
                continue  # Кутові точки не мутуються

            # Точки на лівій і правій межі можуть рухатись тільки по осі Y
            if x == 0 or x == 10:  # Ліва межа (x == 0) або права межа (x == 10)
                direction = np.random.randn(1)  # Випадковий напрямок по осі Y
                direction /= np.abs(direction)  # Нормалізуємо напрямок
                move_distance = np.random.uniform(-mutation_strength, mutation_strength)
                new_points[i][1] += direction * move_distance  # Зміщуємо тільки по осі Y

            # Точки на верхній і нижній межах можуть рухатись тільки по осі X
            elif y == rect_points[0][1] or y == rect_points[2][1]:  # Верхня і нижня межа
                direction = np.random.randn(1)  # Випадковий напрямок по осі X
                direction /= np.abs(direction)  # Нормалізуємо напрямок
                move_distance = np.random.uniform(-mutation_strength, mutation_strength)
                new_points[i][0] += direction * move_distance  # Зміщуємо тільки по осі X

            # Внутрішні точки можуть рухатись в будь-якому напрямку
            else:
                direction = np.random.randn(2)
                direction /= np.linalg.norm(direction)
                move_distance = np.random.uniform(-mutation_strength, mutation_strength)
                new_points[i] += direction * move_distance

            # Оцінка нової пристосованості після зміщення точки
            trial_fitness = grid_fitness(new_points, triangles)

            # Якщо мутація покращує пристосованість (тобто зменшується відхилення від ідеальних трикутників)
            if trial_fitness < current_fitness:  # Чим менше пристосованість, тим краще
                current_fitness = trial_fitness  # Оновлюємо поточну пристосованість
            else:
                # Якщо пристосованість не покращується, відкидаємо зміщення
                new_points[i] = points[i]  # Повертаємо точку на місце

    return new_points





# Функція для відображення сітки
def plot_triangulation(points, triangles, title="Grid"):
    plt.figure(figsize=(6, 6))
    for triangle in triangles:
        p1, p2, p3 = points[triangle]
        x_vals = [p1[0], p2[0], p3[0], p1[0]]
        y_vals = [p1[1], p2[1], p3[1], p1[1]]
        plt.fill(x_vals, y_vals, edgecolor='blue', fill=False)
    plt.plot(points[:, 0], points[:, 1], 'ro')
    plt.xlim(-1, 11)
    plt.ylim(-1, 6)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.show()


# Параметри сітки
width = 10
height = 5
rows = 5
cols = 10

# Створення початкової сітки
points, triangles = create_equal_triangles(width, height, rows, cols)

# Підрахунок пристосованості початкової сітки
initial_fitness = grid_fitness(points, triangles)
print(f"Initial grid fitness: {initial_fitness}")

# Виведення початкової сітки (рівні трикутники)
plot_triangulation(points, triangles, title="Initial Equal Grid")

# Створення мутованої сітки з нерівними трикутниками
mutated_points = mutate(points, triangles, rect_points, mutation_strength=0.5)

# Виведення мутованої сітки з нерівними трикутниками
plot_triangulation(mutated_points, triangles, title="Unequal Grid")

# Оптимізація: повернення сітки до рівного стану
max_generations = 500
tolerance = 1e-5
optimized_points_reverse = mutated_points

# Графік для h/θ
h_theta_values = []

for generation in range(max_generations):
    new_points = reverse_mutate_with_fitness_check(optimized_points_reverse, triangles, rect_points, rows, cols,
                                                   mutation_strength=0.1)

    # Обчислюємо h/θ для поточного покоління
    h = calculate_h(triangles, new_points)
    theta = calculate_theta(triangles, new_points)
    h_theta = h / theta
    h_theta_values.append(h_theta)

    fitness = grid_fitness(new_points, triangles)
    print(f"Generation {generation + 1}: Total fitness = {fitness}")
    if fitness < tolerance:
        print(f"Converged after {generation + 1} generations.")
        break
    optimized_points_reverse = new_points

# Виведення результату оптимізації
plot_triangulation(optimized_points_reverse, triangles, title="Optimized Mutated Reverse Grid")

# Графік h/θ
plt.plot(h_theta_values)
plt.xlabel('Generation')
plt.ylabel('h/θ')
plt.title('h/θ over generations')
plt.show()
