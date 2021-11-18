import numpy as np
from matplotlib import pyplot as plt

c_pts = np.loadtxt("rand_points.txt")
py_pts = np.random.randint(1e8,size=c_pts.shape)

def plot3d(x,y,z):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x,y,z,marker='.',s=1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

