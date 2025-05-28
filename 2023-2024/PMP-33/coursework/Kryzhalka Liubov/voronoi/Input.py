import math
import numpy as np
import matplotlib.pyplot as plt
import triangle as tr
import random
from shapely.geometry import Polygon, Point

def random_vertices(x1=618.0, y1=332.0, x2=417.0, y2=103.0, x3=238.0, y3=334.0):
    
    A = np.array([
        [2*(x2 - x1), 2*(y2 - y1)],
        [2*(x3 - x1), 2*(y3 - y1)]
    ])
    B = np.array([
        [x2**2 - x1**2 + y2**2 - y1**2],
        [x3**2 - x1**2 + y3**2 - y1**2]
    ])
    C = np.linalg.solve(A, B)
    
    h = C[0][0]
    k = C[1][0]
    r = np.sqrt((x1 - h)**2 + (y1 - k)**2)
    

    center_x = h
    center_y = k
    radius = r
    
    x=[]
    y=[]
    res=[]
    while len(x)<10:
        pointx = random.uniform(0, 500)
        pointy = random.uniform(0, 500)
        if np.sqrt((pointx - h)**2 + (pointy - k)**2)<radius:
            x.append(pointx)
            y.append(pointy)
            res.append([pointx, pointy])
   


    plt.scatter(x, y, s = 4)
    return res



def plot_functions(xi, yi):
    plt.plot(xi, yi, label='Input data', color='royalblue')
    plt.xlabel('x')
    plt.ylabel('f(x)')
   
    plt.grid(True)
    
    
    
def read_from_file(name):#output.txt
    with open(name, 'r') as file:
        lines = file.read().splitlines()
    return lines
    