import numpy as np


def circle(N, R):
    i = np.arange(N)
    theta = i * 2 * np.pi / N
    pts = np.stack([np.cos(theta), np.sin(theta)], axis=1) * R
    seg = np.stack([i, i + 1], axis=1) % N
    return pts, seg


def generate_preset_area(preset_type: str, number_of_points: int):
    if preset_type == "Circle":
        points, segments = circle(number_of_points, 1.5)
        return np.vstack(points), np.vstack(segments)
    elif preset_type == "Donut":
        pts0, seg0 = circle(60, 1.4)
        pts1, seg1 = circle(32, 0.6)
        points = np.vstack([pts0, pts1])
        segments = np.vstack([seg0, seg1 + seg0.shape[0]])
        return points, segments
