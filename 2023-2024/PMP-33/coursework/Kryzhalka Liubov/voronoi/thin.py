import numpy as np
import matplotlib.pyplot as plt
from Input import read_from_file
import math
from collections import OrderedDict

def remove_duplicates_ordered(points):
    unique_points = list(OrderedDict.fromkeys(map(tuple, points)))
    return list(map(list, unique_points))



def distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


def thin_spot():
    lines = read_from_file('output.txt')
    pts = []
    for i in range(len(lines)):
        pts.append([float(lines[i].split()[0]), float(lines[i].split()[1])])

    pts1=pts[:len(pts) // 2]
    pts2=pts[len(pts) // 2:]
    x = [point[0] for point in pts]
    y = [point[1] for point in pts]

    min_distance = float('inf')
    point_thin1=0
    point_thin2=0
    for i in range(len(pts1)):
        if i<100:
            cut_vector = pts2[:len(pts2) - 100 + i]
        elif i>(len(pts1)-100):
            cut_vector = pts2[100-(len(pts1)-i):] 
        else:
            cut_vector = pts2
        for point2 in cut_vector:
                dist = distance(pts1[i], point2)
                if dist < min_distance:
                    min_distance = dist
                    index=i
                    point_thin1=pts1[i]
                    point_thin2=point2
    plt.plot(x, y, color='b')#, s=10)
    plt.plot([point_thin1[0], point_thin2[0]],[point_thin1[1], point_thin2[1]], color='red')


    plt.show()
    

    ob1=pts1[:(index+1)]+pts2[pts2.index(point_thin2):]
    
    ob2=pts1[index+1:]+pts2[:(pts2.index(point_thin2))]
   
    ob2 = ob2 + [ob2[0]]

    ob2 = [[x - 100, y] for x, y in ob2]#+[[707.2, 329.8],[698.5, 356.5],[690.1, 381.7]]

    x1 = [point[0] for point in ob1]
    y1 = [point[1] for point in ob1]

    x2 = [point[0] for point in ob2]
    y2 = [point[1] for point in ob2]

    

    plt.plot(x1, y1, color='b')#, s=10)
    plt.plot(x2, y2, color='red')#, s=10)

    plt.show()
    return ob1, ob2

#thin_spot()
