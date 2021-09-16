import numpy as np

x = 42
eps = 2**-52

def deriv(fun,x,dx):
    dx = (x+dx)-x
    f1 = fun(x+dx)
    f2 = fun(x-dx)
    f3 = fun(x+2*dx)
    f4 = fun(x-2*dx)
    return (8*f1-8*f2-f3+f4)/(12*dx)

opt = eps**(1/5)

print('Relative error with optimal dx: ',(deriv(np.exp,x,opt)-np.exp(x))/np.exp(x))
print('Relative error with 10 times larger dx: ',(deriv(np.exp,x,opt*10)-np.exp(x))/np.exp(x))
print('Relative error with 10 times smaller dx: ',(deriv(np.exp,x,opt/10)-np.exp(x))/np.exp(x))
