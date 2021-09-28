import numpy as np

# Takes two-parameter fun (fun(x,y))
def rk4_step(fun,x,y,h):
    b = h/2
    k1 = h*fun(x,y)
    k2 = h*fun(x+b,y+k1/2)
    k3 = h*fun(x+b,y+k2/2)
    k4 = h*fun(x+h,y+k3)
    dy = (k1+2*k2+2*k3+k4)/6
    return y+dy

# fun = function; step = step function; a,b = limits; y0 = initial value (y(a)); n = number of steps
def rk4_eval(fun,step,a,b,y0,n):
    x = np.linspace(a,b,n)
    h = x[1]-x[0]
    y = x*0
    y[0] = y0
    for i in range(1,n):
        y[i] = step(fun,x[i-1],y[i-1],h)
    return x,y

def fun(x,y):
    return y/(1+x**2)

x,y = rk4_eval(fun,rk4_step,-20,20,1,200)
