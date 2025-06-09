import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull
from shapely.geometry import Polygon, Point
from Input import read_from_file
from thin import thin_spot
from triang import get_vertices
import math
from matplotlib.collections import LineCollection

def random_points_inside_circle(center, radius, num_points):
    angles = np.random.uniform(0, 2*np.pi, num_points)
    radii = radius * np.sqrt(np.random.uniform(0, 1, num_points))
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    return np.column_stack((x, y))

def angle_between(v0, v1):
    return math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))


def calc_angle(c0, c1, c2):
    return angle_between(np.array(c1) - np.array(c0), np.array(c2) - np.array(c1))


def is_convex(polygon):
    temp_coords = np.array(polygon.exterior.coords)
    temp_coords = np.vstack([temp_coords, temp_coords[1, :]])

    for i, (c0, c1, c2) in enumerate(zip(temp_coords, temp_coords[1:], temp_coords[2:])):
        if i == 0:
            first_angle_crit = calc_angle(c0, c1, c2) > 0
        elif (calc_angle(c0, c1, c2) > 0) != first_angle_crit:
            return False
    return True


def infinite_segments(vor_):
    line_segments = []
    center = vor_.points.mean(axis=0)
    for pointidx, simplex in zip(vor_.ridge_points, vor_.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.any(simplex < 0):
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor_.points[pointidx[1]] - vor_.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor_.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n

            line_segments.append([(vor_.vertices[i, 0], vor_.vertices[i, 1]),
                                  (direction[0], direction[1])])
    return line_segments


class NotConvexException(Exception):
    def __str__(self):
        return 'The Polygon is not Convex!!!'


class NotAllPointsAreInException(Exception):
    def __str__(self):
        return 'Not all points are in the polygon!!!'


def intersect(p0, u, q0, q1):
    v = (q1 - q0)[np.newaxis].T
    A = np.hstack([u, -v])
    b = q0 - p0
    try:
        inv_A = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    return np.dot(inv_A, b)


def _adjust_bounds(ax__, points_):
    ptp_bound = points_.ptp(axis=0)
    ax__.set_xlim(points_[:, 0].min() - 0.1*ptp_bound[0], points_[:, 0].max() + 0.1*ptp_bound[0])
    ax__.set_ylim(points_[:, 1].min() - 0.1*ptp_bound[1], points_[:, 1].max() + 0.1*ptp_bound[1])


def in_polygon(polygon, points_):
    return [polygon.contains(Point(x)) for x in points_]


def voronoi_plot_2d_inside_convex_polygon(vor_, polygon, ax__=None, **kw):
    from matplotlib.collections import LineCollection

    if not all(in_polygon(polygon, vor_.points)):
        raise NotAllPointsAreInException()

    #if not is_convex(polygon):
        #raise NotConvexException()

    if vor_.points.shape[1] != 2:
        raise ValueError("Voronoi diagram is not 2-D")

    vor_inside_ind = np.array([i for i, x in enumerate(vor_.vertices) if Point(x).within(polygon)])

    vor_inside_ind = np.array([i if polygon.contains(Point(x)) else -1 for i, x in enumerate(vor_.vertices)])
    vor_inside_ind = vor_inside_ind[vor_inside_ind != -1]


    vor_outside_ind = np.array([i for i, x in enumerate(vor_.vertices) if not polygon.contains(Point(x))])
    #ax__.plot(vor_.points[:, 0], vor_.points[:, 1], '.')
    #if kw.get('show_vertices', True):
        #ax__.plot(vor_.vertices[vor_inside_ind, 0], vor_.vertices[vor_inside_ind, 1], 'o')

    temp_coords = np.array(polygon.exterior.coords)
    line_segments = []
    for t0, t1 in zip(temp_coords, temp_coords[1:]):
        line_segments.append([t0, t1])
    ax__.add_collection(LineCollection(line_segments, colors='k', linestyle='solid'))#b
    line_segments = []
    for simplex in vor_.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            if not all(in_polygon(polygon, vor_.vertices[simplex])):
                continue
            line_segments.append([(x, y) for x, y in vor_.vertices[simplex]])

    ax__.add_collection(LineCollection(line_segments, colors='k', linestyle='solid'))#k

    line_segments = infinite_segments(vor_)
    from_inside = np.array([x for x in line_segments if polygon.contains(Point(x[0]))])

    line_segments = []

    for f in from_inside:
        for coord0, coord1 in zip(temp_coords, temp_coords[1:]):
            s, t = intersect(f[0], f[1][np.newaxis].T, coord0, coord1)
            if 0 < t < 1 and s > 0:
                line_segments.append([f[0], f[0] + s * f[1]])
                break

    ax__.add_collection(LineCollection(np.array(line_segments), colors='k', linestyle='solid'))#dashed

    line_segments = []

    for v_o_ind in vor_outside_ind:
        for simplex in vor_.ridge_vertices:
            simplex = np.asarray(simplex)
            if np.any(simplex < 0):
                continue
            if np.any(simplex == v_o_ind):
                i = simplex[simplex != v_o_ind][0]
                for coord0, coord1 in zip(temp_coords, temp_coords[1:]):
                    s, t = intersect(
                        vor_.vertices[i],
                        (vor_.vertices[v_o_ind] - vor_.vertices[i])[np.newaxis].T,
                        coord0,
                        coord1
                    )
                    if 0 < t < 1 and 0 < s < 1:
                        line_segments.append([
                            vor_.vertices[i],
                            vor_.vertices[i] + s * (vor_.vertices[v_o_ind] - vor_.vertices[i])
                        ])
                        break

    ax__.add_collection(LineCollection(np.array(line_segments), colors='k', linestyle='solid'))#rdashed

    _adjust_bounds(ax__, temp_coords)

    return ax__.figure

def sort_vertices(pts):
    center = np.mean(pts, axis=0)  # Знаходимо центр багатокутника
    pts_sorted = sorted(pts, key=lambda x: np.arctan2(x[1] - center[1], x[0] - center[0]))
    return pts_sorted


lines = read_from_file('output.txt')
pts1 = []
pts_result=[]
for i in range(len(lines)):
    #if i%3==0:
    pts1.append([float(lines[i].split()[0]), float(lines[i].split()[1])])
    #pts_result.append([float(lines[i].split()[0]), float(lines[i].split()[1])])

#pts1, pts2 =thin_spot()
#print(len(pts1))


x = [point[0] for point in pts1]
y = [point[1] for point in pts1]


#pts1= sort_vertices(pts1)

#circle_center = (0, 0)
#circle_radius = 100

#num_points = 100
#points = random_points_inside_circle(circle_center, circle_radius, num_points)


points=get_vertices(pts1)





my_polygon = Polygon(pts1)

points=[x for x in points if my_polygon.contains(Point(x[0], x[1]))]



x1 = [point[0] for point in points]
y1 = [point[1] for point in points]


points=np.column_stack((x1, y1))



#circle_polygon = Point(circle_center).buffer(circle_radius)



vor = Voronoi(points)



fig, ax = plt.subplots()
#ax.scatter(x, y, color='b', s=2)
#ax.plot(points[:, 0], points[:, 1], 'o', markersize=2,color='b')



#voronoi_plot_2d_inside_convex_polygon(vor, my_polygon, ax__=ax)
voronoi_plot_2d(vor, ax__=ax)
plt.show()

#ax.add_patch(plt.Circle(circle_center, circle_radius, color='b', fill=False))
