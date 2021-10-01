import numpy as np

# fun = fun(x,y)
def rk4_step(fun,x,y,h):
    b = h/2
    k1 = h*fun(x,y)
    k2 = h*fun(x+b,y+k1/2)
    k3 = h*fun(x+b,y+k2/2)
    k4 = h*fun(x+h,y+k3)
    dy = (k1+2*k2+2*k3+k4)/6
    return y+dy

def rk4_stepd(fun,x,y,h):
    # One step h
    b = h/2
    k1 = h*fun(x,y)
    k2 = h*fun(x+b,y+k1/2)
    k3 = h*fun(x+b,y+k2/2)
    k4 = h*fun(x+h,y+k3)
    y1 = y + (k1+2*k2+2*k3+k4)/6

    # Two steps h/2
    # Step 1
    c = b/2
    l1 = k1/2
    l2 = b*fun(x+c,y+l1/2)
    l3 = b*fun(x+c,y+l2/2)
    l4 = b*fun(x+b,y+l3)
    y_mid = y + (l1+2*l2+2*l3+l4)/6
    # Step 2
    m1 = b*fun(x,y_mid)
    m2 = b*fun(x+c,y_mid+m1/2)
    m3 = b*fun(x+c,y+m2/2)
    m4 = b*fun(x+b,y+m3)
    y2 = y_mid + (m1+2*m2+2*m3+m4)/6

    delta = y2 - y1
    return y1 + delta/15


# fun = function; step = step function; a,b = limits; y0 = initial value (y(a)); n = number of steps
def rk4_eval(fun,step,a,b,y0,n):
    x = np.linspace(a,b,n+1)
    h = x[1]-x[0]
    y = x*0
    y[0] = y0
    for i in range(1,n):
        y[i] = step(fun,x[i-1],y[i-1],h)
    return x,y

def fun(x,y):
    return y/(1+x**2)


x,y = rk4_eval(fun,rk4_step,-20,20,1,200)
xd,yd = rk4_eval(fun,rk4_stepd,-20,20,1,73)

def real_fun(x):
    k = np.exp(np.arctan(20))
    return k*np.exp(np.arctan(x))

y_true = real_fun(x)
yd_true = real_fun(xd)

err = np.mean((y-y_true)/y_true)
errd = np.mean((yd-yd_true)/yd_true)

print("Average relative error for original RK4 with 200 steps: ",err)
print("Average relative error for modified RK4 with 73 steps: ",errd)
print("Ratio: ",errd/err)

