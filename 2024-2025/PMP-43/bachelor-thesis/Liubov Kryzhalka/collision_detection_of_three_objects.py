import matplotlib.pyplot as plt
import numpy as np
from functions import triangulation, check_collision, compute_normals, intersection, merging

def draw_modified_circle(center, radius, stretch_factor, direction):

    theta = np.linspace(0, 2 * np.pi, 200)

    x_radius=np.ones(len(theta))
    y_radius=np.ones(len(theta))

    for i in range(len(theta)):
 
        if direction==0:
            if np.cos(theta[i]) > 0:
                x_radius[i]=radius * stretch_factor
                y_radius[i]=radius
            else:
                x_radius[i]=radius
                y_radius[i]=radius
        elif direction==np.pi/2:
            if np.sin(theta[i])>0:
                x_radius[i]=radius
                y_radius[i]=radius * stretch_factor
            else:
                x_radius[i]=radius
                y_radius[i]=radius
        elif direction==np.pi:
            if np.cos(theta[i])<0:
                x_radius[i]=radius * stretch_factor
                y_radius[i]=radius
            else:
                x_radius[i]=radius
                y_radius[i]=radius
        elif direction==3*np.pi/2:
            if np.sin(theta[i])< 0:
                x_radius[i]=radius
                y_radius[i]=radius* stretch_factor
            else:
                x_radius[i]=radius
                y_radius[i]=radius

    x = center[0] + x_radius * np.cos(theta)
    y = center[1] + y_radius * np.sin(theta)
    
    return np.column_stack((x, y))

def clear_plot(ax):
    ax.clear()
    ax.axis('off')
    ax.set_xlim(-5, 20)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal') 
    ax.set_title(f"Step {i}", fontsize=14, fontweight='bold')


circle1 = {"center": (7, 4), "radius": 2}
circle2 = {"center": (15, 6), "radius": 3}
circle3 = {"center": (0, 7), "radius": 3}

max_stretch1 = 4
max_stretch2 = 4
direction1 = 0
direction2 = np.pi
steps = 70

circles = [circle1, circle2, circle3]

fig, ax = plt.subplots()
collision1=False
collision2=False
collision3=False

result_array=[]
for i in range(steps):
    clear_plot(ax)

    stretch_factor1 = 1 + (max_stretch1 - 1) * (i / steps)
    stretch_factor2 = 1 + (max_stretch2 - 1) * (i / steps)

    points1 = draw_modified_circle(circles[0]["center"], circles[0]["radius"], stretch_factor1, direction1)
    points2 = draw_modified_circle(circles[1]["center"], circles[1]["radius"], stretch_factor2, direction2)
    points3 = draw_modified_circle(circles[2]["center"], circles[2]["radius"], stretch_factor1, direction1)
    if not collision1:
        ax.plot(points1[:, 0], points1[:, 1], 'k', alpha=0.8)
        ax.plot(points2[:, 0], points2[:, 1], 'g', alpha=0.8)
    if not collision2:
        ax.plot(points3[:, 0], points3[:, 1], 'c', alpha=0.8)
    normals1 = compute_normals(points1)
    normals2 = compute_normals(points2)
    normals3 = compute_normals(points3)
    if not collision1:
        triangulation(ax, points1,1)
        triangulation(ax, points2,2)
    if not collision2:
        triangulation(ax, points3,1)
    
    if check_collision(points1, normals1, points2, normals2):
        collision1=True
        x, y, _, _ = intersection(points1[:, 0], points1[:, 1], points2[:, 0], points2[:, 1])
        
        result_list = merging(x,y, points1, points2)

        result_array = np.array(result_list)
        
        if not collision2:
            clear_plot(ax)
            ax.plot(points3[:, 0], points3[:, 1], 'c', alpha=0.8)
            triangulation(ax, points3,1)
            ax.plot(result_array[:, 0], result_array[:, 1], 'b', linewidth=2)
            triangulation(ax, result_array,2)

    if check_collision(points1, normals1, points3, normals3):
        collision2 = True
        if collision1:
            x, y, _, _ = intersection(points3[:, 0], points3[:, 1], result_array[:, 0], result_array[:, 1])
            result_list = merging(x, y, result_array, points3)
        else:
            x, y, _, _ = intersection(points1[:, 0], points1[:, 1], points3[:, 0], points3[:, 1])
            result_list = merging(x, y, points1, points3)

        result_array = np.array(result_list)

        clear_plot(ax)
        ax.plot(result_array[:, 0], result_array[:, 1], 'b', linewidth=2)
        triangulation(ax, result_array,2)

    plt.draw()
    plt.pause(0.5)

plt.show()