import math
import random
import numpy as np
from PIL import Image
import triangle as tr

from .models import Point


def distance(p1: Point, p2: Point) -> float:
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def polygon_centroid(points: list[Point]) -> Point:
    x_mean = sum(p.x for p in points) / len(points)
    y_mean = sum(p.y for p in points) / len(points)
    return Point(x_mean, y_mean)


def find_contour(points: list[Point]) -> list[Point]:
    if not points:
        return []
    points = points[:]
    contour = [points.pop(0)]
    while points:
        last_point = contour[-1]
        nearest = min(
            points, key=lambda p: (p.x - last_point.x) ** 2 + (p.y - last_point.y) ** 2
        )
        contour.append(nearest)
        points.remove(nearest)
    return contour


class Area:
    def __init__(self, points: list[Point]) -> None:
        self.points = points
        self.tri = None
        self.segments = []

    def smooth_boundary(self, iterations=1, alpha=0.5):
        if len(self.points) < 3:
            return

        n = len(self.points)
        for _ in range(iterations):
            new_points = []
            old_points = self.points[:]

            for i in range(n):
                p_prev = old_points[i - 1]
                p_curr = old_points[i]
                p_next = old_points[(i + 1) % n]

                mid_x = 0.5 * (p_prev.x + p_next.x)
                mid_y = 0.5 * (p_prev.y + p_next.y)

                new_x = p_curr.x + alpha * (mid_x - p_curr.x)
                new_y = p_curr.y + alpha * (mid_y - p_curr.y)

                new_points.append(Point(new_x, new_y))

            self.points = new_points

    def triangulate_polygon(
        self,
        min_angle: float = 20.0,
        max_area: float = 0.0,
        max_steiner_points: int = 0,
    ):
        vertices = [p.as_tuple() for p in self.points]
        segments = [(i, (i + 1) % len(vertices)) for i in range(len(vertices))]
        data = dict(vertices=np.array(vertices), segments=np.array(segments))
        self.segments = segments

        switches = "p"
        if min_angle > 0:
            switches += f"q{float(min_angle):g}"
        else:
            switches += "q"
        if max_area > 0:
            switches += f"a{float(max_area):g}"
        if max_steiner_points > 0:
            switches += f"S{int(max_steiner_points)}"

        self.tri = tr.triangulate(data, switches)

        if "triangles" not in self.tri:
            self.tri = None
            raise ValueError("Триангуляція не вдалася")

    def move_points(self, delta_t: float):
        old_points = self.points[:]
        n = len(old_points)
        if n < 2:
            return old_points

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

            cp_vec = Point(cur_p.x - centroid.x, cur_p.y - centroid.y)
            dot_prod = avg_n.x * cp_vec.x + avg_n.y * cp_vec.y
            if dot_prod < 0:
                avg_n = Point(-avg_n.x, -avg_n.y)

            v = random.uniform(0, 10)
            s = v * delta_t
            new_p = Point(cur_p.x + avg_n.x * s, cur_p.y + avg_n.y * s)
            new_points.append(new_p)

        self.points = new_points
        return old_points


def load_area_from_image_bytes(image_bytes: bytes, threshold=2, sampling_rate=0.1) -> Area:
    from io import BytesIO

    img = Image.open(BytesIO(image_bytes)).convert("L")
    width, height = img.size
    pixels = np.array(img)

    points = []
    for y in range(height):
        for x in range(width):
            if pixels[y, x] < threshold and random.random() < sampling_rate:
                points.append(Point(x, height - y))

    points = find_contour(points)

    if len(points) < 3:
        raise ValueError("Недостатньо точок для побудови полігону")

    return Area(points)


def load_area_from_points(points_data: list[dict] | list[Point]) -> Area:
    points: list[Point] = []

    for p in points_data:
        if isinstance(p, Point):
            points.append(p)
        else:
            points.append(Point(float(p["x"]), float(p["y"])))

    if len(points) < 3:
        raise ValueError("Полігон повинен містити хоча б 3 точки")

    return Area(points)
