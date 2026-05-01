from .models import Point
from .geometry import Area, distance
from .fem_service import solve_fields


def points_from_dto(point_dtos):
    return [Point(x=p.x, y=p.y) for p in point_dtos]


def serialize_state(area, vertices, triangles, concentration, pressure, delta_t):
    return {
        "points": [{"x": p.x, "y": p.y} for p in area.points],
        "vertices": [{"x": float(v[0]), "y": float(v[1])} for v in vertices],
        "triangles": [
            {"a": int(t[0]), "b": int(t[1]), "c": int(t[2])}
            for t in triangles
        ],
        "concentration": [float(c) for c in concentration],
        "pressure": [float(p) for p in pressure],
        "delta_t": float(delta_t),
    }


def build_area_from_state(state):
    points = points_from_dto(state.points)
    area = Area(points)
    return area


def enrich_area(area, params):
    area.triangulate_polygon(
        min_angle=params.triangulation_min_angle,
        max_area=params.triangulation_max_area,
        max_steiner_points=params.triangulation_max_steiner_points,
    )
    return area



def do_single_step(area, params, delta_t):
    prev_points = area.move_points(delta_t)

    jumps = [
        distance(prev_points[i], area.points[i]) for i in range(len(area.points))
    ]
    if any(j > params.max_jump for j in jumps):
        if delta_t > params.min_delta_t:
            delta_t /= 2.0

    inserted_points = []
    n = len(area.points)
    for i in range(n):
        p1 = area.points[i]
        p2 = area.points[(i + 1) % n]
        inserted_points.append(p1)
        dist_p = distance(p1, p2)
        if dist_p > params.max_dist_between_points:
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
            if dist_p < params.min_dist_between_points and n > 3:
                mid = Point((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0)
                merged_points.append(mid)
                i += 2
            else:
                merged_points.append(p1)
                i += 1
        area.points = merged_points

    area.smooth_boundary(
        iterations=params.smoothing_iterations,
        alpha=params.smoothing_alpha,
    )

    area.triangulate_polygon(
        min_angle=params.triangulation_min_angle,
        max_area=params.triangulation_max_area,
        max_steiner_points=params.triangulation_max_steiner_points,
    )

    vertices, triangles, concentration, pressure = solve_fields(
        area,
        a_11=params.a_11,
        a_22=params.a_22,
        force_value=params.force_value,
        boundary_value=params.boundary_value,
        pressure_a=params.pressure_a,
        pressure_g=params.pressure_g,
        pressure_chi=params.pressure_chi,
        pressure_k=params.pressure_k,
        pressure_dimension=params.pressure_dimension,
    )

    return area, vertices, triangles, concentration, pressure, delta_t
