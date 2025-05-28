import numpy as np
from scipy.spatial import Delaunay
from PIL import Image, ImageDraw
import cv2

def load_image(path):
    return Image.open(path).convert("RGB")

def get_random_points(img, num_points, seed=None):
    width, height = img.size
    rng = np.random.default_rng(seed)  
    points = set()
    while len(points) < num_points:
        x = rng.integers(0, width)
        y = rng.integers(0, height)
        points.add((x, y))
    points.update([(0, 0), (0, height - 1), (width - 1, 0), (width - 1, height - 1)])
    return np.array(list(points))

def get_edge_points(img, num_points, seed=None):
    gray = np.array(img.convert("L"))
    edges = cv2.Canny(gray, 100, 200)
    
    
    y, x = np.nonzero(edges)
    points = list(zip(x, y))

    width, height = img.size
    points.extend([(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)])

    rng = np.random.default_rng(seed)
    if len(points) > num_points:
        indices = rng.choice(len(points), num_points, replace=False)
        points = [points[i] for i in indices]

    return np.array(points)

def draw_delaunay(img, points, tri):
    width, height = img.size
    output = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(output)
    pixels = img.load()

    for triangle in tri.simplices:
        pts = [tuple(points[i]) for i in triangle]
        cx = int(sum(p[0] for p in pts) / 3)
        cy = int(sum(p[1] for p in pts) / 3)

        if 0 <= cx < width and 0 <= cy < height:
            color = pixels[cx, cy]
        else:
            color = (0, 0, 0)

        draw.polygon(pts, fill=color)

    return output

def compress_image_with_delaunay(image_path, output_path, num_points=800, use_edges=True, seed=None):
    img = load_image(image_path)

    if use_edges:
        points = get_edge_points(img, num_points, seed=seed)
    else:
        points = get_random_points(img, num_points, seed=seed)

    tri = Delaunay(points)
    compressed = draw_delaunay(img, points, tri)
    compressed.save(output_path)
    return compressed