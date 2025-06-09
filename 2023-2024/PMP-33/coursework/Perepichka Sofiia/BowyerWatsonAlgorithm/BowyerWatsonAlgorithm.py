from cProfile import label
import math
from turtle import color
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np



LENGTH_TO_ADD=10
EPS=1e-6
MAX_AREA=50
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def equals(self, point):
        return (abs(self.x-point.x)<EPS and abs(self.y-point.y)<EPS)
        

class Edge:
    def __init__(self, v0, v1):
        self.v0 = v0
        self.v1 = v1

    def equals(self, edge):
       return (self.v0.equals(edge.v0) and self.v1.equals(edge.v1)) or (self.v0.equals(edge.v1) and self.v1.equals(edge.v0))

class Triangle:
    def __init__(self, v0, v1, v2):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.edges = [Edge(v0, v1), Edge(v1, v2), Edge(v2, v0)]
        self.vertices = [v0, v1, v2]
        self.additional_points = []


        
class CircumCircle:
    def __init__(self,x,y,radius):
        self.x=x
        self.y=y
        self.radius=radius
    
        
        
        
def super_triangle(points):
    minx = miny = maxx = maxy = 0
   
    
    for point in points:
        if point.x < minx:
            minx = point.x
        elif point.x > maxx:
            maxx = point.x

        if point.y < miny:
            miny = point.y
        elif point.y > maxy:
            maxy = point.y
    
    length_x = (maxx - minx) * LENGTH_TO_ADD
    length_y = (maxy - miny) * LENGTH_TO_ADD
    
    v0 = Point(minx - length_x, miny -length_y * LENGTH_TO_ADD)
    v1 = Point(minx -  length_x, maxy + length_y)
    v2 = Point(maxx + length_x * LENGTH_TO_ADD, maxy + length_y)
 
        

    
    return Triangle(v0, v1, v2)  


def find_circumcircle(v0, v1, v2):
    d = 2 * (v0.x * (v1.y - v2.y) + v1.x * (v2.y - v0.y) + v2.x * (v0.y - v1.y))
    ux = ((v0.x * v0.x + v0.y * v0.y) * (v1.y - v2.y) + (v1.x * v1.x + v1.y * v1.y) * (v2.y - v0.y) + (v2.x * v2.x + v2.y * v2.y) * (v0.y - v1.y)) / d
    uy = ((v0.x * v0.x + v0.y * v0.y) * (v2.x - v1.x) + (v1.x * v1.x + v1.y * v1.y) * (v0.x - v2.x) + (v2.x * v2.x + v2.y * v2.y) * (v1.x - v0.x)) / d
    
   
    radius = math.sqrt((v0.x - ux) ** 2 + (v0.y - uy) ** 2)
    
    return CircumCircle(ux,uy, radius)

def in_circumcircle(point, circum_circle):
    length_x = circum_circle.x - point.x
    length_y = circum_circle.y - point.y
    return math.sqrt(length_x **2 + length_y **2) <= circum_circle.radius


def calculate_area(triangle):
    x1, y1 = triangle.v0.x, triangle.v0.y
    x2, y2 = triangle.v1.x, triangle.v1.y
    x3, y3 = triangle.v2.x, triangle.v2.y
    return abs((x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2.0)

def calculate_centroid(triangle):
    x1, y1 = triangle.v0.x, triangle.v0.y
    x2, y2 = triangle.v1.x, triangle.v1.y
    x3, y3 = triangle.v2.x, triangle.v2.y
    cx = (x1 + x2 + x3) / 3
    cy = (y1 + y2 + y3) / 3
    return (cx, cy)

def find_midpoint(point1, point2):
    midpoint_x = (point1.x + point2.x) / 2
    midpoint_y = (point1.y + point2.y) / 2
    return Point(midpoint_x, midpoint_y)

def generate_triangulation(points):
    triangulation=[]
    
    s_triangle= super_triangle(points)
    
    triangulation.append(s_triangle)
    for point in points:
     triangulation = add_point(point, triangulation)
     
    triangulation=remove_super_triangle(triangulation,s_triangle)
    
    while True:
        new_points = []
        for triangle in triangulation:
            area = calculate_area(triangle)
            if area > MAX_AREA:
                centroid = calculate_centroid(triangle)
                new_points.append(Point(centroid[0], centroid[1])) 
                
        if not new_points:
            break
        
        for point in new_points:
            triangulation = add_point(point, triangulation)
        # x_coordinates,y_coordinates,vertices_indices,additional_x_coordinates,additional_y_coordinates=triangulation_data(triangulation)
        # plt.ion()
        # plt.margins(0.1)
        # plt.gca().set_aspect('equal')
        
        # for triangle_indices in vertices_indices:
        #    triangle_x = [x_coordinates[i] for i in triangle_indices]
        #    triangle_y = [y_coordinates[i] for i in triangle_indices]
        #    plt.plot(triangle_x, triangle_y,'ko-',markersize=3,linewidth=1) 
        # plt.plot([],[],'k-',label='Implemented Triangulation')
        # plt.plot(xs, ys, label='Contour of area',color='darkblue')
        # plt.legend()
        
        # plt.draw() 
        # plt.pause(10)
        # plt.clf()
        
       

    
    for triangle in triangulation:
     triangle.additional_points.append(find_midpoint(triangle.v0, triangle.v1))
     triangle.additional_points.append(find_midpoint(triangle.v1, triangle.v2))
     triangle.additional_points.append(find_midpoint(triangle.v2, triangle.v0))
    
    return triangulation      
  
def add_point(point,triangulation):
    bad_triangles = []
    

    for triangle in triangulation:
        circle= find_circumcircle(triangle.v0, triangle.v1, triangle.v2)
        if in_circumcircle(point,circle):
                
                bad_triangles.append(triangle)

    polygon = []

   
    for i in range(len(bad_triangles)):
     current_triangle = bad_triangles[i]
     for edge in current_triangle.edges:
        isNeighbour = False
        for k in range(i):
            other_triangle = bad_triangles[k]
            for other_edge in other_triangle.edges:
                
                if edge.equals(other_edge):
                    isNeighbour = True
        for k in range(i + 1, len(bad_triangles)):
            other_triangle = bad_triangles[k]
            for other_edge in other_triangle.edges:
                
                if edge.equals(other_edge):    
                    isNeighbour = True
        if not isNeighbour:
            polygon.append(edge)
     
    for triangle in bad_triangles:
     triangulation.remove(triangle)

    for edge in polygon:
      new_triangle = Triangle(edge.v0, edge.v1, point)
      triangulation.append(new_triangle)
    return triangulation

def remove_super_triangle(triangulation,s_triangle):
    new_triangulation=[]
    for triangle in triangulation:
        if not (triangle.v0 == s_triangle.v0 or triangle.v0 == s_triangle.v1 or triangle.v0 == s_triangle.v2 \
        or triangle.v1 == s_triangle.v0 or triangle.v1 == s_triangle.v1 or triangle.v1 == s_triangle.v2 \
        or triangle.v2 == s_triangle.v0 or triangle.v2 == s_triangle.v1 or triangle.v2 == s_triangle.v2):
            new_triangulation.append(triangle)
    return new_triangulation

def triangulation_data(triangulation):
 points = []
 for triangle in triangulation:
    for point in triangle.vertices:
        points.append(point)

 x_coordinates = []
 y_coordinates = []
 additional_x_coordinates = []
 additional_y_coordinates = []
 for point in points:
   x_coordinates.append(point.x)
   y_coordinates.append(point.y)

 vertices_indices  = []
 for triangle in triangulation:
    indices = []
    for point in triangle.vertices:
        indices.append(points.index(point))
    vertices_indices .append(tuple(indices))
    for point in triangle.additional_points:
        additional_x_coordinates.append(point.x)
        additional_y_coordinates.append(point.y)

 return x_coordinates,y_coordinates,vertices_indices,additional_x_coordinates,additional_y_coordinates



file_path = "contours2.txt"


data = np.loadtxt(file_path)

xs = data[:, 0]
ys = data[:, 1]


n=len(xs)
ni = 15

last_index = n - math.floor(n / ni)

indexes = [math.floor(i) for i in np.linspace(0, last_index, ni)]

xi = [xs[i] for i in indexes]
yi = [ys[i] for i in indexes]


points = [Point(x, y) for x, y in zip(xi, yi)]


triangulation = generate_triangulation(points)

x_coordinates,y_coordinates,vertices_indices,additional_x_coordinates,additional_y_coordinates=triangulation_data(triangulation)


plt.margins(0.1)
plt.gca().set_aspect('equal')


for triangle_indices in vertices_indices:
    triangle_x = [x_coordinates[i] for i in triangle_indices]
    triangle_y = [y_coordinates[i] for i in triangle_indices]
    plt.plot(triangle_x, triangle_y,'ko-',markersize=3,linewidth=1) 
plt.plot(additional_x_coordinates, additional_y_coordinates, 'ro', markersize=2, label='Additional Points')
plt.legend()

# # plt.margins(0.1)
# plt.gca().set_aspect('equal')
# # plt.triplot(tri.Triangulation(xi, yi), 'bo-',color='red',label='Built-in Triangulation')

plt.plot(xs, ys, label='Contour of area',color='darkblue')
# plt.plot(xi, yi, label='Figure''bo-',color='red')
# # plt.triplot(tri.Triangulation(x_coordinates, y_coordinates,  vertices_indices), 'bo--',markersize=4,color='black',label='Implemented Triangulation')
# plt.title('Plot of Delaunay triangulation')
# plt.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize='small', fancybox=True, framealpha=0.2)
plt.show()


