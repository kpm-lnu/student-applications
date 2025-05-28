import matplotlib.pyplot as plt
import triangle as tr
import numpy as np
import random
import Input
from scipy.spatial import Voronoi, voronoi_plot_2d

lines = Input.read_from_file('output.txt')
      
pts=[]
for i in range(len(lines)):
    #if i%2==0:
    pts.append([float(lines[i].split()[0]), float(lines[i].split()[1])])
    
print(pts)

x = [point[0] for point in pts]
y = [point[1] for point in pts] 
Input.plot_functions(x,y)

res = Input.random_vertices()
#points=np.array(list(zip(x_points, y_points)))
A1 = dict(vertices=pts)

B1 = tr.triangulate(A1, 'q')

A = dict(vertices=res)

points, edges, ray_origin, ray_direct = tr.voronoi(res)
B = dict(
     vertices=points, edges=edges,
     ray_origins=ray_origin, ray_directions=ray_direct
)
tr.compare(plt, A, B)
fig, ax = plt.subplots(figsize=(12, 12))
tr.plot(ax,**B)
tr.plot(ax,**B1)
#vor=Voronoi(points)
#voronoi_plot_2d(vor, show_vertices=False)
#plt.plot(vor)
plt.show()
