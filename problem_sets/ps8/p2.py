import numpy as np
from matplotlib import pyplot as plt

### PART A

def point_charge_potential(n):
    # Generate axes
    x = np.arange(n)-n//2
    x = np.fft.fftshift(x)
    xmat,ymat = np.meshgrid(x,x)
    rmat = np.sqrt(xmat**2+ymat**2)
    rmat[0,0] = 1
    pot = -np.log(rmat)#/2/np.pi
    pot[0,0] = 4*pot[1,0]-pot[2,0]-pot[1,-1]-pot[1,1]
    pot = pot/pot[0,0]
    #pot = pot-pot[0,0]+1
    
    return pot

### PART B

# We use conjugate gradient to solve Ax=b where A is G*, the convolution with our Green's
# function, x is the charge rho, and b is the boundary potential.

# This function will serve as "A", performing the convolution to get the potential.
def rho2pot(rho,kernelft):
    (n,m)=rho.shape
    rho_padded = np.pad(rho,(0,n))
    rhoft = np.fft.rfft2(rho_padded)
    pot_padded = np.fft.irfft2(rhoft*kernelft)
    pot = pot_padded[:n,:m]
    return pot

def rho2pot_masked(rho,mask,kernelft):
    rhomat = np.zeros(mask.shape)
    rhomat[mask] = rho
    potmat = rho2pot(rhomat,kernelft)
    return potmat[mask]

# This function will solve for x, given A and b.
def conjgrad(x0,b,mask,kernelft,niter=1,fun=rho2pot_masked):
    # Based on Jon's conjugate gradient code
    r = b-fun(x0,mask,kernelft)
    p = r.copy()
    x = x0.copy()
    rr = np.sum(r*r)
    for i in range(niter):
        Ap = fun(p,mask,kernelft)
        pAp = np.sum(p*Ap)
        alpha = rr/pAp
        x = x+alpha*p
        r = r-alpha*Ap
        rr_new = np.sum(r*r)
        beta = rr_new/rr
        p = r+beta*p
        rr = rr_new
    return x

def gen_square_bc(n):
    bc = np.zeros((n,n))
    mask = np.zeros((n,n),dtype='bool')
    bc[n//4,n//4:3*n//4] = 1
    mask[n//4,n//4:3*n//4] = 1
    bc[3*n//4,n//4:3*n//4] = 1
    mask[3*n//4,n//4:3*n//4] = 1
    bc[n//4:3*n//4,n//4] = 1
    mask[n//4:3*n//4,n//4] = 1
    bc[n//4:3*n//4,3*n//4] = 1
    mask[n//4:3*n//4,3*n//4] = 1
    mask[0,:]=1
    mask[-1,:]=1
    mask[:,0]=1
    mask[:,-1]=1
    return bc,mask

# Running everything

n = 128

kernel = point_charge_potential(2*n)
kernelft = np.fft.rfft2(kernel)

bc,mask = gen_square_bc(n)
b = bc[mask]
x0 = 0*b

rho = np.zeros(mask.shape)
rho[mask] = conjgrad(x0,b,mask,kernelft,niter=3*n)

oneside = rho[n//4:3*n//4+1,n//4]

showfigs = True
if showfigs:
    plt.figure(1)
    plt.imshow(rho)
    plt.show()

    plt.figure(2)
    plt.plot(oneside)
    plt.show()


### PART C

# Calculating potential everywhere with our Green's function
pot = rho2pot(rho,kernelft)
interior_pot = pot[n//4+1:3*n//4-1,n//4+1:3*n//4-1]

# The field is the gradient of the potential (ignoring signs). I will use two-point
# derivative to solve, and assume that dx=dy=1.
fx = (np.roll(pot,1,axis=1)-np.roll(pot,-1,axis=1))/2
fy = (np.roll(pot,1,axis=0)-np.roll(pot,-1,axis=0))/2


if showfigs:
    plt.figure(3)
    plt.imshow(pot)
    plt.show()

    plt.figure(4)
    plt.imshow(interior_pot)
    plt.show()

    plt.figure(5)
    plt.quiver(fx,fy)
    plt.show()
