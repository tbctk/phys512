import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt


R = 1

# For this problem I will use the simpson's rule adaptive integration from problem 2.
def integrate_adaptive(fun,a,b,tol,extra=None):
    x = np.linspace(a,b,5)
    dx = x[1]-x[0]
    if extra == None:
        f0 = fun(a)
        f4 = fun(b)
        f2 = fun(x[2])
    else:
        (f0,f4,f2) = extra
    sol0 = 2*(f0+4*f2+f4)*dx/3
    
    f1,f3 = fun(x[1]),fun(x[3])

    sol1 = (f0+4*f1+2*f2+4*f3+f4)*dx/3
    
    if np.abs(sol1-sol0) < tol:
        return sol1
    else:
        return integrate_adaptive(fun,x[0],x[2],tol/2,extra=(f0,f2,f1))+integrate_adaptive(fun,x[2],x[4],tol/2,extra=(f2,f4,f3))

def my_integrate(fun,a,b):
    return integrate_adaptive(fun,a,b,1e-9)


def dE(z,R,u):
    top = (R**2)*(z-R*u)
    bot = (R**2+z**2-2*z*R*u)**(3/2)
    return top/bot
    
def my_E(z):
    global R
    def tmp(u):
        return dE(z,R,u)
    return my_integrate(tmp,-1,1)

def quad_E(z):
    global R
    def tmp(u):
        return dE(z,R,u)
    (E,r) = integrate.quad(tmp,-1,1)
    return E

# This integral has a singularity at z = R. My  integrator was unable to handle it, whereas quad
# had no problem. From looking at the results, it appears that quad handles it by just linearly
# interpolating between points before and after. In order to avoid infinite recursion when graphing,
# I decided to just set the value to 10 at these problematic points; that way they will be highlighted 
# on the plot but will not cause problems.

z = np.linspace(0,3,2**6)

E1 = z*0
for i in range(len(E1)):
    if np.abs(z[i]-R) < 0.02:
        E1[i] = 10
    else:
        E1[i] = my_E(z[i])

E2 = z*0
for i in range(len(E2)):
    E2[i] = quad_E(z[i])

plt.figure()
plt.plot(z,E1)
plt.plot(z,E2)
plt.show()
