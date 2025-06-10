import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString
from functions import triangulation, find_bottleneck, find_point_index, find_threshold

def shift_polygons(part1, part2, shift_val):
    result = []
    direction=1
    shifted1 = [(x, y + direction * shift_val) for x, y in part1]
    direction=-1
    shifted2 = [(x, y + direction * shift_val) for x, y in part2]
    result.append(np.array(shifted1))
    result.append(np.array(shifted2))
    return result

def create_pinched_contour(num_points, center, radius, pinch_strength):
    cx, cy = center
    theta = np.linspace(0, 2 * np.pi, num_points)
    pinch_profile = 1 - pinch_strength * np.exp(-(np.sin(theta))**2 / 0.2)
    x = cx + radius * np.cos(theta) * pinch_profile
    y = cy + radius * np.sin(theta)
    return np.column_stack((x, y))


def create_double_arc(p1, p2, bulge=1.0, num_points=80):
    p1 = np.array(p1)
    p2 = np.array(p2)
    midpoint = (p1 + p2) / 2
    normal = np.array([p2[1] - p1[1], -(p2[0] - p1[0])])
    normal = normal / np.linalg.norm(normal)

    upper_control = midpoint + normal * bulge
    lower_control = midpoint - normal * bulge

    t = np.linspace(0, 1, num_points // 2)
    upper_arc = [(1 - tt)**2 * p1 + 2 * (1 - tt) * tt * upper_control + tt**2 * p2 for tt in t]
    lower_arc = [(1 - tt)**2 * p1 + 2 * (1 - tt) * tt * lower_control + tt**2 * p2 for tt in reversed(t)]

    return (upper_arc, lower_arc)



def split_object(contour, p1, p2):
    part1, part2 = create_double_arc(p1, p2, bulge=0.5, num_points=10)
    
    i1 = find_point_index(contour, p1)
    i2 = find_point_index(contour, p2)

    if i1 > i2:
        i1, i2 = i2, i1
        part1, part2 = part2, part1 

    upper_part = list(contour[i1:i2]) + part2
    lower_part = list(contour[i2:] ) + part1

    return lower_part, upper_part

center = (0, 0)
radius = 5
num_points = 300
pinch_values = np.linspace(0.0, 0.995, 100)

plt.ion() 
fig, ax = plt.subplots(figsize=(6, 6))
frame=0
i=0
for pinch_strength in pinch_values:
    i=i+1
    ax.clear()
    ax.set_xlim(-6, 6)
    ax.set_ylim(-8.5, 8.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"Step {i}", fontsize=14, fontweight='bold')

    contour = create_pinched_contour(num_points, center, radius, pinch_strength)
    triangulation(ax, contour, 2)

    threshold = find_threshold(contour)
    medial_lines, closest_pair, is_thin= find_bottleneck(contour, threshold)
    (p1, p2)=closest_pair

    if not is_thin:
        ax.plot(contour[:, 0], contour[:, 1], 'b-', lw=2)
    
    #for line, _ in medial_lines:
        #x, y = line.xy
        #ax.plot(x, y, 'r-', linewidth=0.8)

    if is_thin:
        i=i-1
        break

    plt.draw()
    plt.pause(7)


if is_thin:
    for frame in range(1,16,1):
        ax.clear()
        ax.set_xlim(-6, 6)
        ax.set_ylim(-8.5, 8.5)
        ax.set_aspect('equal')
        ax.axis('off')
        i=i+1
        ax.set_title(f"Step {i}", fontsize=14, fontweight='bold')

        polygon = Polygon(contour)
        cut_line = LineString([p1, p2])
        lower_geom, upper_geom = split_object(contour, p1, p2)
        shifted_coords = shift_polygons( upper_geom,lower_geom, frame * 0.2)
        upper_part = shifted_coords[0]
        lower_part = shifted_coords[1]

        ax.plot(upper_part[:, 0], upper_part[:, 1], 'b-', lw=2)
        ax.plot(lower_part[:, 0], lower_part[:, 1], 'b-', lw=2)

        triangulation(ax, upper_part, 2)
        triangulation(ax, lower_part, 2)

        plt.draw()
        plt.pause(7) 

plt.ioff()
plt.show()
