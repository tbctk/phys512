import numpy as np
from matplotlib import pyplot as plt

### PART A

n = 16

#def point_charge_potential(n):
    # Generate axes
x = np.arange(n)-n//2
x = np.fft.fftshift(x)
xmat,ymat = np.meshgrid(x,x)
rmat = np.sqrt(xmat**2+ymat**2)
rmat[0,0] = 1
pot = -np.log(rmat)/2/np.pi
pot = pot-pot[n//2,n//2]
pot[0,0] = 4*pot[1,0]-pot[2,0]-pot[1,-1]-pot[1,1]
#pot = pot-pot[0,0]+1
