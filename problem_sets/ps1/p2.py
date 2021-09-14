import numpy as np

def ndiff(fun,x,full=False):
    eps = np.finfo(float).eps
    dx = eps**(1/3)
    f1 = fun(x+dx)
    f2 = fun(x-dx)
    deriv = (f1-f2)/(2*dx)
    if full:
        err = dx**2
        return deriv,dx,err
    else:
        return deriv

x = 1
diff = ndiff(np.exp,x,True)

print('Expected: ',np.exp(x),' Numerical: ',diff)
