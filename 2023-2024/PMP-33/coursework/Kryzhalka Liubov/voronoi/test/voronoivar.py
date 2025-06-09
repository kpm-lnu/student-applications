from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

from geovoronoi import coords_to_points, points_to_coords, voronoi_regions_from_coords, calculate_polygon_areas

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
pts = Polygon([Point(i) for i in pts])
#plot_functions(x,y)
#plt.show()
#area_shape=Polygon(pts)

coords = random_vertices()
coords = np.array(coords)
#vor = Voronoi(coords)
#sregions, vertices = voronoi_finite_polygons_2d(vor)



if len(coords) > 3: #otherwise the tesselation won't work
    vor = Voronoi(coords)
    regions, vertices = voronoi_finite_polygons_2d(vor)


    new_vertices = []
    for region in regions:
        poly_reg = vertices[region]
        shape = list(poly_reg.shape)
        shape[0] += 1
        p = Polygon(np.append(poly_reg, poly_reg[0]).reshape(*shape))
        if p.is_valid:
            p = p.intersection(pts)
        else:
            p = p.buffer(0).intersection(pts)
        poly = (np.array(p.exterior.coords)).tolist()
        new_vertices.append(poly)

    #plots the results
    fig, ax = plt.subplots()
    ax.imshow(pts,cmap='Greys_r')
    for poly in new_vertices:
        ax.fill(*zip(*poly), alpha=0.7)
    ax.plot(coords[:,0],coords[:,1],'ro',ms=2)
    plt.show()
