import numpy as np

def ndiff(fun,x,full=False):
    # Use numpy arrays to make element-wise arithmetic easy
    x = np.array(x)
    
    # Calculate optimal dx
    eps = 2**(-52)
    dx = eps**(1/3)
    dx = (x+dx)-x
    
    # Evaluate derivative
    f1 = fun(x+dx)
    f2 = fun(x-dx)
    deriv = (f1-f2)/(2*dx)
    
    if full:
        # Leading order error is (f'''(x)/3)dx^2. We drop constant terms to get an approximate error dx^2.
        err = dx**2
        return deriv,dx,err
    else:
        return deriv

x = 42
diff = ndiff(np.exp,x,False)

print('Output for f=exp, x=42, full=True: ',ndiff(np.exp,x,True))
print('Output for f=exp, x=42, full=False: ',diff)
print('Relative Error: ',(np.exp(x)-diff)/np.exp(x))
