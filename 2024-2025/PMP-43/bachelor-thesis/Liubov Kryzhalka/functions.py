import numpy as np
import triangle as tr
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, LineString
                 
def projection(axis, points):
    projections = [np.dot(point, axis) for point in points]
    return min(projections), max(projections)


def overlap(axis, points1, points2):
    min_proj1, max_proj1 = projection(axis, points1)
    min_proj2, max_proj2 = projection(axis, points2)
    return max_proj1 >= min_proj2 and max_proj2 >= min_proj1


def check_collision(points1, normals1, points2, normals2):
    axes = normals1 + normals2

    for axis in axes:
        if not overlap(axis, points1, points2):
            return False 
    return True


def compute_normals(points):
    normals = []
    n = len(points)
    
    for i in range(n):
        p_prev = points[i - 1]
        p_next = points[(i + 1) % n]

        tangent = np.array(p_next) - np.array(p_prev)

        normal = np.array([-tangent[1], tangent[0]])

        norm_length = np.linalg.norm(normal)
        if norm_length == 0:
            normals.append(np.array([0.0, 0.0]))
        else:
            normals.append(normal / norm_length)
    
    return normals


def rect_inter_inner(x1, x2):
    n1 = x1.shape[0]-1
    n2 = x2.shape[0]-1
    X1 = np.c_[x1[:-1], x1[1:]]
    X2 = np.c_[x2[:-1], x2[1:]]
    S1 = np.tile(X1.min(axis=1), (n2, 1)).T
    S2 = np.tile(X2.max(axis=1), (n1, 1))
    S3 = np.tile(X1.max(axis=1), (n2, 1)).T
    S4 = np.tile(X2.min(axis=1), (n1, 1))
    return S1, S2, S3, S4


def rectangle_intersection(x1, x2):
    S1, S2, S3, S4 = rect_inter_inner(x1, x2)
    C1 = np.less_equal(S1, S2)
    C2 = np.greater_equal(S3, S4)
    return C1, C2


def rectangle_intersection_check(x1, y1, x2, y2):
    C1, C2 = rectangle_intersection(x1, x2)
    C3, C4 = rectangle_intersection(y1, y2)
    ii, jj = np.nonzero(C1 & C2 & C3 & C4)
    return ii, jj


def intersection(x1, y1, x2, y2):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    ii, jj = rectangle_intersection_check(x1, y1, x2, y2)
    n = len(ii)

    dxy1 = np.diff(np.c_[x1, y1], axis=0)
    dxy2 = np.diff(np.c_[x2, y2], axis=0)

    T = np.zeros((4, n))
    AA = np.zeros((4, 4, n))
    AA[0:2, 2, :] = -1
    AA[2:4, 3, :] = -1
    AA[0::2, 0, :] = dxy1[ii, :].T
    AA[1::2, 1, :] = dxy2[jj, :].T

    BB = np.zeros((4, n))
    BB[0, :] = -x1[ii].ravel()
    BB[1, :] = -x2[jj].ravel()
    BB[2, :] = -y1[ii].ravel()
    BB[3, :] = -y2[jj].ravel()

    for i in range(n):
        try:
            T[:, i] = np.linalg.solve(AA[:, :, i], BB[:, i])
        except:
            T[:, i] = np.Inf

    in_range = (T[0, :] >= 0) & (T[1, :] >= 0) & (
        T[0, :] <= 1) & (T[1, :] <= 1)

    ij0 = [ii[in_range], jj[in_range]] + T[:2, in_range]
    ij0 = ij0.T
    xy0 = T[2:, in_range]
    xy0 = xy0.T

    return xy0[:, 0], xy0[:, 1], ij0[:, 0], ij0[:, 1]


def find_nearest_points(points, x, y):
    nearest_indices = []
    for xi, yi in zip(x, y):
        distances = np.sqrt((points[:, 0] - xi)**2 + (points[:, 1] - yi)**2) 
        nearest_index = np.argmin(distances) 
        nearest_indices.append(nearest_index)
    return nearest_indices


def merging(x,y,points1,points2):
    nearest_indices1 = find_nearest_points(points1, x, y)
    nearest_indices2 = find_nearest_points(points2, x, y)

    ind11=nearest_indices1[0]
    ind12=nearest_indices1[1]

    if ind11>ind12:
        temp = ind11
        ind11=ind12
        ind12=temp

    nearest_indices1[0]=ind11
    nearest_indices1[1]=ind12

    ind21=nearest_indices2[0]
    ind22=nearest_indices2[1]

    if ind21>ind22:
        temp = ind21
        ind21=ind22
        ind22=temp
    
    merged_points=[]

    if  ind11<=len(points1)//2 and ind12<=len(points1)//2:
        for i in range(nearest_indices1[1],len(points1),1):
            merged_points.append(points1[i])
        for i in range(ind11):
            merged_points.append(points1[i])
    else:
        for i in range(nearest_indices1[0],nearest_indices1[1],1):
            merged_points.append(points1[i])

    if ind21<=len(points2)//4 and ind22>len(points2)//2:
        for i in range(ind21,ind22+1,1):
            merged_points.append(points2[i])
    else:
        for i in range(ind22,len(points2)):
            merged_points.append(points2[i])
        for i in range(ind21):
            merged_points.append(points2[i])

    merged_points.append(merged_points[0])
    return merged_points


def find_bottleneck(contour, threshold):
    polygon = Polygon(contour)
    vor = Voronoi(contour)
    is_thin = False
    medial_lines = []
    for i, vpair in enumerate(vor.ridge_vertices):
        if -1 in vpair:
            continue
        p1, p2 = vor.vertices[vpair]
        line = LineString([p1, p2])
        if polygon.contains(line):
            medial_lines.append((line, vor.ridge_points[i]))
    min_dist = np.inf
    closest_pair = None
    for line, (idx1, idx2) in medial_lines:
        midpoint = line.interpolate(0.5, normalized=True)
        dist = polygon.exterior.distance(midpoint)
        if dist < min_dist:
            min_dist = dist
            closest_pair = (contour[idx1], contour[idx2])
    if min_dist < threshold:
        is_thin = True
    return medial_lines, closest_pair, is_thin


def find_threshold(contour ):
    x_min, y_min = np.min(contour, axis=0)
    x_max, y_max = np.max(contour, axis=0)
    width = x_max - x_min
    height = y_max - y_min
    max_dim = max(width, height)
    return 0.025 * max_dim


def find_point_index(contour, point):
    for i, pt in enumerate(contour):
        if np.allclose(pt, point):
            return i
    raise ValueError("Point not found in contour")


def triangulation(ax, contour, area): 
    if len(contour) > 1 and np.allclose(contour[0], contour[-1]):
                contour = contour[:-1]

    segments = [[i, i + 1] for i in range(len(contour) - 1)]
    segments.append([len(contour) - 1, 0])

    A = {
        "vertices": contour,
        "segments": segments
    }

    try:
        B = tr.triangulate(A, f'pq20a{area}')
        ax.triplot(
            B['vertices'][:, 0],
            B['vertices'][:, 1],
            B['triangles'],
            color='black',
            linewidth=0.8
        )
    except Exception as e:
        print(f"Тріангуляція не вдалася: {e}")


def read_contours_from_file(path):
    contours = []
    current_contour = []

    with open(path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("Contour"):
                if current_contour:
                    contours.append(np.array(current_contour, dtype=np.float32))
                    current_contour = []
            elif line:
                x_str, y_str = line.split(",")
                x, y = float(x_str.strip()), float(y_str.strip())
                current_contour.append([x, y])

    if current_contour:
        contours.append(np.array(current_contour, dtype=np.float32))

    return contours


def write_contours_to_file(contours, filepath, min_length=0):
    with open(filepath, "w") as f:
        for i, cnt in enumerate(contours):
            if len(cnt) > min_length:
                f.write(f"Contour {i+1}:\n")
                pts = cnt.reshape(-1, 2)
                for x, y in pts:
                    f.write(f"{x}, {y}\n")
                f.write("\n")