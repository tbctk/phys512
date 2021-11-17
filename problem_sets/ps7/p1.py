import numpy as np
from matplotlib import pyplot as plt

# I want to minimize number of planes

def plot3d(x,y,z):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x,y,z,marker='.')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

