import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def matplotVisual(pointcloud, position, fig, title_name):

    ax = fig.add_subplot(position, projection='3d')
    ax.scatter(pointcloud[:,0], pointcloud[:,1], pointcloud[:,2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title(title_name)
