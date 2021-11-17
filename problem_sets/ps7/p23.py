import numpy as np
from matplotlib import pyplot as plt

### PROBLEM 2

# Say we want exponential deviate with PDF = alpha*exp(-alpha*t), where alpha > 0.
# We can use the positive part of a Lorentzian centered at 0 for our rejection.
# Lorentzian CDF would be arctan(a*t)*(2/pi)

def expdev_rejection(n,alpha=1,makeplots=False):
    assert alpha > 0
    # Generate n random points between 0 and alpha
    q = np.random.rand(n)*alpha
    # Use inverse Lorentzian CDF to get t values of the random points
    t = np.tan(q*np.pi/2)/alpha
    # Generate n points distributed under Lorentzian
    y = alpha/(1+(alpha*t)**2)*np.random.rand(n)
    # Accept points that fall under exponential curve
    lim = alpha*np.exp(-alpha*t)
    accept = y<lim
    t_use = t[accept]
    
    # Plotting
    if makeplots:
        plt.figure()
        bins = np.linspace(0,20,501)
        aa,bb = np.histogram(t_use,bins)
        aa = aa/aa.sum()
        cents = 0.5*(bins[1:]+bins[:-1])
        pred = alpha*np.exp(-alpha*cents)
        pred = pred/pred.sum()
        plt.plot(cents,aa,'*')
        plt.plot(cents,pred,'r')
    
    return t_use

### PROBLEM 3

def get_vmax(alpha):
    u = np.linspace(0,1,2001)[1:] 
    v = u*np.log(alpha/u**2)/alpha
    return v.max()

def expdev_ratio(n,alpha=1,makeplots=False):
    assert alpha > 0
    # Generate n random points (u,v) in bounding rectangular region
    u = np.random.rand(n)
    vmax = get_vmax(alpha)
    v = np.random.rand(n)*vmax
    # Convert back to (t,y) space and accept points where y < p(t)
    t = v/u
    y = u**2
    lim = alpha*np.exp(-alpha*t)
    accept = y < lim
    t_use = t[accept]
    
    # Plotting
    if makeplots:
        plt.figure()
        bins = np.linspace(0,20,501)
        aa,bb = np.histogram(t_use,bins)
        aa = aa/aa.sum()
        cents = 0.5*(bins[1:]+bins[:-1])
        pred = alpha*np.exp(-alpha*cents)
        pred = pred/pred.sum()
        plt.plot(cents,aa,'*')
        plt.plot(cents,pred,'r')

    return t
