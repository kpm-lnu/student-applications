import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from Input import read_from_file, plot_functions, random_vertices

#rng = np.random.default_rng()
#points = rng.random((10,2))

#vor = Voronoi(points)

lines = read_from_file('output.txt')
      
pts=[]
for i in range(len(lines)):
    #if i%100==0:
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



fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange',
                      line_width=2, line_alpha=0.6, point_size=2)

plot_functions(x, y)
plt.show()