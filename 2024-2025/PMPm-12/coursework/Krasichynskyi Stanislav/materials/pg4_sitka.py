import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import math
import triangle as tr

from compute_fucns import (
    assemble_global_matrix,
    assemble_rhs,
    compute_ke,
    compute_qe,
)  # Імпортуємо пакет triangle


class Point:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar: float):
        return Point(self.x * scalar, self.y * scalar)

    def as_tuple(self):
        return (self.x, self.y)


def distance(p1: Point, p2: Point) -> float:
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def polygon_centroid(points: list[Point]) -> Point:
    # Простіше взяти середнє координат,
    # хоча для нерівномірного полігону centroid зважений за площею.
    # Для нашого випадку припустимо просте середнє:
    x_mean = sum(p.x for p in points) / len(points)
    y_mean = sum(p.y for p in points) / len(points)
    return Point(x_mean, y_mean)


class Area:
    def __init__(self, points: list[Point]) -> None:
        self.points = points
        self.tri = None  # Змінна для зберігання триангуляції

    def display_area(self):
        plt.clf()
        x = [p.x for p in self.points]
        y = [p.y for p in self.points]
        if len(self.points) > 1:
            x.append(self.points[0].x)
            y.append(self.points[0].y)
        plt.plot(x, y, c="blue", linewidth=1, marker="o", markersize=2, label="Полігон")
        plt.title("Defined Area from PNG")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()

        # Відображення триангуляції, якщо вона існує
        if self.tri and "triangles" in self.tri:
            plt.triplot(
                self.tri["vertices"][:, 0],
                self.tri["vertices"][:, 1],
                self.tri["triangles"],
                color="gray",
                linewidth=0.5,
                label="Триангуляція",
            )

        plt.draw()
        plt.pause(0.01)

    def smooth_boundary(self, iterations=1, alpha=0.5):
        """
        Просте лапласівське згладжування границі полігона.
        iterations : int    – кількість ітерацій згладжування
        alpha      : float  – коефіцієнт змішування (0 < alpha <= 1)
                            alpha ближче до 1 дає сильніше згладжування,
                            але може "стискати" полігон
        """
        if len(self.points) < 3:
            return

        n = len(self.points)
        for _ in range(iterations):
            new_points = []
            # Збережемо старий список, аби оновлення не впливало на обчислення в межах ітерації
            old_points = self.points[:]

            for i in range(n):
                p_prev = old_points[i - 1]
                p_curr = old_points[i]
                p_next = old_points[(i + 1) % n]

                # Лапласівське згладжування:
                # 1) знайдемо середнє двох сусідів (p_prev і p_next)
                mid_x = 0.5 * (p_prev.x + p_next.x)
                mid_y = 0.5 * (p_prev.y + p_next.y)

                # 2) змішаємо поточну точку з цим середнім
                new_x = p_curr.x + alpha * (mid_x - p_curr.x)
                new_y = p_curr.y + alpha * (mid_y - p_curr.y)

                new_points.append(Point(new_x, new_y))

            self.points = new_points

    def triangulate_polygon(self):
        """
        Виконує триангуляцію полігону за допомогою пакету triangle.
        """
        # Підготовка даних для триангуляції
        vertices = [p.as_tuple() for p in self.points]
        segments = [(i, (i + 1) % len(vertices)) for i in range(len(vertices))]
        data = dict(vertices=np.array(vertices), segments=np.array(segments))
        self.segments = segments
        # Виконання триангуляції
        self.tri = tr.triangulate(data, "pq")

        # Перевірка, чи триангуляція успішна
        if "triangles" not in self.tri:
            print("Триангуляція не вдалася. Перевірте правильність полігону.")
            self.tri = None

    def triangulate_and_display(self):
        """
        Виконує триангуляцію та відображає полігон з триангуляцією.
        """
        self.triangulate_polygon()
        self.display_area()

    def display_normals(self, scale=5.0):
        n = len(self.points)
        if n < 2:
            return
        centroid = polygon_centroid(self.points)

        # Списки для зберігання координат та компонент нормалей
        xs = []
        ys = []
        us = []
        vs = []

        for i in range(n):
            prev_p = self.points[i - 1]
            cur_p = self.points[i]
            next_p = self.points[(i + 1) % n]

            v1 = prev_p - cur_p
            v2 = next_p - cur_p

            n1 = Point(v1.y, -v1.x)
            n2 = Point(v2.y, -v2.x)
            avg_n = Point((n1.x + n2.x) / 2.0, (n1.y + n2.y) / 2.0)
            length = math.sqrt(avg_n.x**2 + avg_n.y**2)
            if length != 0:
                avg_n = Point(avg_n.x / length, avg_n.y / length)
            else:
                avg_n = Point(0, 0)

            # Перевіряємо напрямок нормалі відносно центроїда
            cp_vec = Point(
                cur_p.x - centroid.x, cur_p.y - centroid.y
            )  # вектор від центру до точки
            dot_prod = avg_n.x * cp_vec.x + avg_n.y * cp_vec.y
            if dot_prod < 0:
                # Перевертаємо нормаль
                avg_n = Point(-avg_n.x, -avg_n.y)

            # Зберігаємо координати та компоненти нормалі
            xs.append(cur_p.x)
            ys.append(cur_p.y)
            us.append(avg_n.x * scale)
            vs.append(avg_n.y * scale)

        # Відображаємо всі нормалі за допомогою quiver
        plt.quiver(
            xs,
            ys,  # Позиції стрілок
            us,
            vs,  # Компоненти векторів
            angles="xy",
            scale_units="xy",
            scale=1,
            color="green",
            width=0.003,
            headwidth=3,
            headlength=5,
            label="Зовнішні нормалі",
        )
        plt.legend()
        plt.draw()
        plt.pause(0.01)

    def move_points(self, delta_t: float):
        old_points = self.points[:]
        n = len(old_points)
        if n < 2:
            return
        new_points = []
        centroid = polygon_centroid(self.points)

        for i in range(n):
            prev_p = old_points[i - 1]
            cur_p = old_points[i]
            next_p = old_points[(i + 1) % n]

            v1 = prev_p - cur_p
            v2 = next_p - cur_p

            n1 = Point(v1.y, -v1.x)
            n2 = Point(v2.y, -v2.x)

            avg_n = Point((n1.x + n2.x) / 2.0, (n1.y + n2.y) / 2.0)
            length = math.sqrt(avg_n.x**2 + avg_n.y**2)
            if length != 0:
                avg_n = Point(avg_n.x / length, avg_n.y / length)
            else:
                avg_n = Point(0, 0)

            # Перевірка напрямку нормалі відносно центроїда
            cp_vec = Point(cur_p.x - centroid.x, cur_p.y - centroid.y)
            dot_prod = avg_n.x * cp_vec.x + avg_n.y * cp_vec.y
            if dot_prod < 0:
                # Якщо нормаль дивиться всередину, перевертаємо її
                avg_n = Point(-avg_n.x, -avg_n.y)

            # Випадкова швидкість [0,10]
            V = random.uniform(0, 10)
            S = V * delta_t

            new_p = Point(cur_p.x + avg_n.x * S, cur_p.y + avg_n.y * S)
            new_points.append(new_p)

        self.points = new_points
        return old_points

    def triangulate_and_display(self):
        """
        Виконує триангуляцію та відображає полігон з триангуляцією.
        """
        self.triangulate_polygon()
        self.display_area()


def find_contour(points: list[Point]) -> list[Point]:
    if not points:
        return []
    contour = [points.pop(0)]
    while points:
        last_point = contour[-1]
        nearest = min(
            points, key=lambda p: (p.x - last_point.x) ** 2 + (p.y - last_point.y) ** 2
        )
        contour.append(nearest)
        points.remove(nearest)
    return contour


class SystemConfig:
    def __init__(self):
        self.MAXIMUM_LENGTH_PER_TIME_LIMIT = 0.1
        self.MAX_DISTANCE_BETWEEN_POINTS = 0.5
        self.MIN_DISTANCE_BETWEEN_POINTS = 0.5


def load_area_from_png(image_path: str, threshold=2, sampling_rate=0.1) -> Area:
    img = Image.open(image_path).convert("L")
    width, height = img.size
    pixels = np.array(img)

    points = []
    for y in range(height):
        for x in range(width):
            if pixels[y, x] < threshold and random.random() < sampling_rate:
                points.append(Point(x, height - y))
    points = find_contour(points)
    return Area(points)


if __name__ == "__main__":
    image_path = "t2.bmp"  # Замініть на шлях до вашого зображення
    sampling_rate = 0.02
    area = load_area_from_png(image_path, sampling_rate=sampling_rate)

    plt.ion()
    area.triangulate_and_display()  # Виконуємо триангуляцію та відображаємо полігон

    delta_t = 0.1
    MAX_JUMP = 2.0
    MAX_DIST_BETWEEN_POINTS = 10.0
    MIN_DIST_BETWEEN_POINTS = 0.5
    MIN_DELTA_T = 1e-5
    debug_mode = True

    while True:
        old_points = area.points[:]
        prev_points = area.move_points(delta_t)
        if prev_points is None:
            area.triangulate_and_display()
            if debug_mode:
                area.display_normals()
            input("Press Enter to continue...")
            continue

        jumps = [
            distance(prev_points[i], area.points[i]) for i in range(len(area.points))
        ]
        if any(j > MAX_JUMP for j in jumps):
            if delta_t > MIN_DELTA_T:
                delta_t /= 2.0

        inserted_points = []
        n = len(area.points)
        for i in range(n):
            p1 = area.points[i]
            p2 = area.points[(i + 1) % n]
            inserted_points.append(p1)
            dist_p = distance(p1, p2)
            if dist_p > MAX_DIST_BETWEEN_POINTS:
                mid = Point((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0)
                inserted_points.append(mid)
        area.points = inserted_points

        if len(area.points) > 2:
            merged_points = []
            i = 0
            n = len(area.points)
            while i < n:
                p1 = area.points[i]
                p2 = area.points[(i + 1) % n]
                dist_p = distance(p1, p2)
                if dist_p < MIN_DIST_BETWEEN_POINTS and n > 3:
                    mid = Point((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0)
                    merged_points.append(mid)
                    i += 2
                else:
                    merged_points.append(p1)
                    i += 1
            area.points = merged_points

        area.smooth_boundary(iterations=1, alpha=0.5)

        area.triangulate_and_display()
        if debug_mode:
            area.display_normals()

        vertices = area.tri["vertices"]
        triangles = area.tri["triangles"]

        edges = set()

        for tri in triangles:
            edges.add(tuple(sorted((tri[0], tri[1]))))
            edges.add(tuple(sorted((tri[1], tri[2]))))
            edges.add(tuple(sorted((tri[0], tri[2]))))

        triangle_vertices = np.array([[vertices[j] for j in i] for i in triangles])

        assembled_system = np.zeros((len(vertices), len(vertices)))

        rhs = np.zeros(len(vertices))

        all_edges = set(map(tuple, map(sorted, area.segments)))

        boundary_points = {pt for edge in all_edges for pt in edge}

        for i in range(len(triangles)):
            ke = np.array(
                compute_ke(
                    triangle_vertices[i],
                    a_11=1,
                    a_22=1,
                )
            )

            qe = compute_qe(triangle_vertices[i], fe=[0.1, 0.1, 0.1])

            assembled_system = assemble_global_matrix(
                assembled_system, ke, triangles[i]
            )
            rhs = assemble_rhs(qe, triangles[i], rhs)

        for i in range(len(vertices)):
            if i in boundary_points or i in boundary_points:
                assembled_system[i, :] = 0
                assembled_system[i, i] = 1e7
                rhs[i] = 1e7

        concentration_solution = np.linalg.solve(assembled_system, rhs)

        X_concentration = area.tri["vertices"][:, 0]
        Y_concentration = area.tri["vertices"][:, 1]
        Z_concentration = concentration_solution

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection="3d")
        # surf = ax.plot_trisurf(
        #     X_concentration,
        #     Y_concentration,
        #     Z_concentration,
        #     cmap="jet",
        #     edgecolor="none",
        # )
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        # ax.set_title("Concentration")
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("u(x, y)")
        # plt.show()

        input("Press Enter to continue...")
        # plt.close(fig)