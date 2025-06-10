import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from functions import triangulation, read_contours_from_file, find_point_index

def compute_local_thickness(contour, search_radius=20):
    n = len(contour)
    thickness_values = []
    closest_pairs = []
    tree = cKDTree(contour)

    for i, pt in enumerate(contour):
        prev_pt = contour[i - 1]
        next_pt = contour[(i + 1) % n]
        tangent = next_pt - prev_pt
        if np.linalg.norm(tangent) == 0:
            continue
        tangent = tangent / np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])
        indices = tree.query_ball_point(pt, r=search_radius)
        min_dist = np.inf
        min_pt = None
        for j in indices:
            if j == i:
                continue
            direction = contour[j] - pt
            norm = np.linalg.norm(direction)
            if norm == 0:
                continue
            direction_norm = direction / norm
            if abs(np.dot(direction_norm, normal)) > 0.9:
                if norm < min_dist:
                    min_dist = norm
                    min_pt = contour[j]
        if min_pt is not None and not np.isnan(min_dist):
            thickness_values.append(min_dist)
            closest_pairs.append((pt, min_pt))

    return thickness_values, closest_pairs

def split_and_shift(contour, p1, p2, shift_val, axis="x"):
    part1 = np.array([p1, p2])
    part2 = np.array([p2, p1])
    i1 = find_point_index(contour, p1)
    i2 = find_point_index(contour, p2)
    if i1 > i2:
        i1, i2 = i2, i1
        p1, p2 = p2, p1
        part1 = np.array([p1, p2])
        part2 = np.array([p2, p1])

    part1 = np.concatenate([contour[i1 : i2 + 1]])
    part2 = np.concatenate([contour[i2:], contour[:i1 + 1]])


    if axis == "x":
        part1_shifted = part1 + np.array([shift_val, 0])
        part2_shifted = part2 - np.array([shift_val, 0])
    else:
        part1_shifted = part1 + np.array([0, shift_val])
        part2_shifted = part2 - np.array([0, shift_val])

    return part1_shifted, part2_shifted

contour = read_contours_from_file("./data/smoothed_contours_of_one_tumor.txt")[0]
thickness_values, closest_pairs = compute_local_thickness(contour)
thickness_values = np.array(thickness_values)
top_indices = np.argsort(thickness_values)[:4]
top_thin_pairs_pts = [closest_pairs[i] for i in top_indices]
top_thin_pairs = [
    (find_point_index(contour, p1), find_point_index(contour, p2))
    for p1, p2 in top_thin_pairs_pts
]

steps = 15
all_frames = []

for step in range(steps):
    print(f"Thinning phase – step {step + 1} / {steps}")
    scale = 1.0 - step * 0.03
    scaled = contour.copy()
    for i1, i2 in top_thin_pairs:
        mid = 0.5 * (scaled[i1] + scaled[i2])
        for i in [i1, i2]:
            scaled[i] = mid + scale * (scaled[i] - mid)
    all_frames.append([scaled.copy()])

latest_contour = all_frames[-1][0].copy()


max_shift = 30
for step in range(steps):
    print(f"Splitting phase – step {step + 1} / {steps}")
    shift_val = step * (max_shift / steps)
    current_parts = []

    i1, i2 = top_thin_pairs[1]
    p1 = latest_contour[i1] 
    p2 = latest_contour[i2]
    part1, part2 = split_and_shift(latest_contour, p1, p2, shift_val, axis="x")
    part1 = np.vstack([part1, part1[0]])
    part2 = np.vstack([part2, part2[0]])
    current_parts.append(part1)
    current_parts.append(part2)
    all_frames.append(current_parts)


total_steps = len(all_frames)
print(f"Загальна кількість кроків анімації: {total_steps}")

fig, ax = plt.subplots(figsize=(12, 8))

x_min, x_max = contour[:, 0].min() - 50, contour[:, 0].max() + 50
y_min, y_max = contour[:, 1].min() - 40, contour[:, 1].max() + 40

for idx, frame_parts in enumerate(all_frames):
    ax.clear()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(
        x_min + 2.0,            
        y_max - 2.0,               
        f"Step {idx + 1} / {total_steps}",
        fontsize=12,
        color="red",
        verticalalignment="top",
    )

    if len(frame_parts) == 1:
        triangulation(ax, frame_parts[0], 200)
        ax.plot(frame_parts[0][:, 0], frame_parts[0][:, 1], "b-")
    else:
        for part in frame_parts:
            triangulation(ax, part, 200)
            ax.plot(part[:, 0], part[:, 1], "b-")

    plt.draw()
    plt.pause(0.5)

plt.show()
