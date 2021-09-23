import numpy as np

z,R = 5,6

def my_integrate(fun,a,b):
    return integrate_adaptive(fun,a,b,1e-9)

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


def E(th,z,R):
    top = np.sin(th)*(z-R*np.cos(th))
    bot = (R**2+z**2-2*z*R*np.cos(th))**(3/2)
    return top/bot

def fun(th):
    global z,R
    return E(th,z,R)

print(integrate_adaptive(fun,0,np.pi,1e-11))
