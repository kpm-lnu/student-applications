import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation
from scipy.interpolate import splprep, splev
from PIL import Image

def triangulate_boundary(shape_points, inner_points, grid_size=0.1, threshold_area=0.01, refinement_iterations=5):
    def is_point_inside_polygon(x, y, poly):
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def compute_triangle_area(points, triangle):
        a, b, c = points[triangle]
        return 0.5 * np.abs(np.cross(b - a, c - a))

    def refine_triangulation(triang, polygon, threshold_area):
        points = np.c_[triang.x, triang.y]
        new_points = []

        for triangle in triang.triangles:
            area = compute_triangle_area(points, triangle)
            if area > threshold_area:
                centroid = np.mean(points[triangle], axis=0)
                if is_point_inside_polygon(centroid[0], centroid[1], polygon):
                    new_points.append(centroid)

        if new_points:
            all_points = np.vstack([points, new_points])
            return Triangulation(all_points[:, 0], all_points[:, 1])
        else:
            return triang

    def filter_triangulation(triang, polygon):
        mask = np.zeros(triang.triangles.shape[0], dtype=bool)
        points = np.c_[triang.x, triang.y]
        for i, triangle in enumerate(triang.triangles):
            centroid = np.mean(points[triangle], axis=0)
            if is_point_inside_polygon(centroid[0], centroid[1], polygon):
                mask[i] = True
        return Triangulation(triang.x, triang.y, triangles=triang.triangles[mask])

    all_points = np.vstack([shape_points, inner_points])
    tri = Triangulation(all_points[:, 0], all_points[:, 1])

    for _ in range(refinement_iterations):
        tri = refine_triangulation(tri, shape_points, threshold_area)

    filtered_tri = filter_triangulation(tri, shape_points)

    return filtered_tri

def generate_shape_points(radii, angles):
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    tck, u = splprep([x, y], s=0, per=True)
    unew = np.linspace(0, 1.0, 1000)
    out = splev(unew, tck)
    return np.c_[out[0], out[1]]

num_points = 35
angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
initial_radii = np.array([2.5, 2.6, 2.8, 2.9, 2.7, 2.4, 2.3, 2.1, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.3,
                          3.1, 3.0, 2.8, 2.7, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 2.1, 2.3, 2.5, 2.7, 2.9, 3.0, 2.8, 2.5])

def boundary_motion(x_g, t):
    return 1 - 0.5 * t * np.sin(x_g)

def generate_radii(t):
    return initial_radii * boundary_motion(angles, t)

fig, ax = plt.subplots(figsize=(6, 6))
ax.axis('equal')
ax.axis('off')

max_radius = initial_radii.max() * 1.6
ax.set_xlim(-max_radius, max_radius)
ax.set_ylim(-max_radius, max_radius)

shape_points = generate_shape_points(initial_radii, angles)
inner_points = np.random.uniform(-max_radius, max_radius, (300, 2))
tri = triangulate_boundary(shape_points, inner_points)

line, = ax.plot(shape_points[:, 0], shape_points[:, 1], color='blue', alpha=0.6)
triplot = ax.triplot(tri, 'b-', lw=0.5, alpha=0.5)

frames = []
previous_radii = initial_radii.copy()
for t in np.linspace(0, 1, 30):
    updated_radii = generate_radii(t)
    if not np.allclose(updated_radii, previous_radii):
        updated_shape_points = generate_shape_points(updated_radii, angles)
        updated_tri = triangulate_boundary(updated_shape_points, inner_points)
        previous_radii = updated_radii.copy()

        line.set_data(updated_shape_points[:, 0], updated_shape_points[:, 1])

        for tri_line in triplot:
            tri_line.remove()
        triplot = ax.triplot(updated_tri, 'b-', lw=0.5, alpha=0.5)

    fig.canvas.draw()
    image = np.array(fig.canvas.renderer.buffer_rgba())
    frames.append(Image.fromarray(image))

frames[0].save('res.gif', save_all=True, append_images=frames[1:], duration=20, loop=0)

plt.show()