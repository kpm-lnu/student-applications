#import logging
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

from geovoronoi import coords_to_points, points_to_coords, voronoi_regions_from_coords, calculate_polygon_areas
#from geovoronoi.plotting import subplot_for_map, plot_voronoi_polys_with_points_in_area

from Input import read_from_file, plot_functions, random_vertices
from shapely.geometry import MultiPoint, Point, Polygon
from scipy.spatial import Voronoi

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1] 
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

lines = read_from_file('output.txt')
      
pts=[]
for i in range(len(lines)):
    if i%100==0:
       pts.append([float(lines[i].split()[0]), float(lines[i].split()[1])])
    
#print(pts)



x = [point[0] for point in pts]
y = [point[1] for point in pts] 
#plot_functions(x,y)
#plt.show()
#area_shape=Polygon(pts)

coords = random_vertices()
coords = np.array(coords)
vor = Voronoi(coords)
regions, vertices = voronoi_finite_polygons_2d(vor)



mask = Polygon([Point(i) for i in coords])
new_vertices = []
for region in regions:
    polygon = vertices[region]
    shape = list(polygon.shape)
    shape[0] += 1
    p = Polygon(np.append(polygon, polygon[0]).reshape(*shape)).intersection(mask)
    poly = np.array(list(zip(p.boundary.coords.xy[0][:-1], p.boundary.coords.xy[1][:-1])))
    new_vertices.append(poly)
    plt.fill(*zip(*poly), alpha=0.4)
plt.plot(coords[:,0], coords[:,1], 'ko')
plt.title("Clipped Voronois")
plt.show()












#coords = np.array(coords, dtype=np.int32)
#print(coords)

#%%

#
# calculate the Voronoi regions, cut them with the geographic area shape and assign the points to them
#

#region_polys, region_pts = voronoi_regions_from_coords(coords, area_shape, per_geom=False)

#print(region_polys)