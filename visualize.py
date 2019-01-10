import numpy as np
from open3d import *
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize(pointcloud, window_name):
    pcd = read_point_cloud(pointcloud, format='xyz')
    print("pcd size",len(pcd.points))
    draw_geometries([pcd])


def matplotVisual(pointcloud, position, fig, title_name):

    ax = fig.add_subplot(position, projection='3d')
    ax.scatter(pointcloud[:,0], pointcloud[:,1], pointcloud[:,2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title(title_name)
