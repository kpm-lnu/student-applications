import triangle as tr


def create_triangulation_options_string(
    mesh_type, nodes_order, minimum_angle=20, maximum_area=1
) -> str:
    if mesh_type == "Triangular":
        ## TODO. Add support of a0.00+
        return "pq" + str(minimum_angle) + "a0." + str(maximum_area)


def triangulate_given_area(points, segments, triangulation_options, nodes_order):
    print(nodes_order)
    area = dict(vertices=points, segments=segments)
    triangulation_results = tr.triangulate(area, triangulation_options)

    return triangulation_results
