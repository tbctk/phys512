import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt

# Load data from lakeshore.txt
data = np.loadtxt('lakeshore.txt')
data = np.transpose(data)
data = np.flip(data,axis=1)

def lakeshore(V,data,showerror=False):
    x = data[1]
    y = data[0]
    
    # Interpolate temperature using spline interpolation
    spln = interpolate.splrep(x,y)
    T = interpolate.splev(V,spln)
    
    # Error approximation is based on Rigel's implementation of the bootstrap method
    if showerror:
        rand = np.random.default_rng(seed=12345)
        resample_size = 100
        n_resamples = 10
        gen_pts = []
        for i in range(n_resamples):
            # Generate a random set of resample_size data points and sort them
            indices = list(range(x.size))
            B = rand.choice(indices,size=resample_size,replace=False)
            B.sort()
            # Get the points that will be used to generate the interpolation
            sample_x = x[B]
            sample_y = y[B]
            # Interpolate the temperature at V
            spln1 = interpolate.splrep(sample_x,sample_y)
            T1 = interpolate.splev(V,spln1)
            # Store results in gen_pts
            gen_pts.append(T1) 
        # Get error as the standard deviation of the results
        gen_pts = np.array(gen_pts)
        err = np.std(gen_pts,axis=0)
        
        return T,err

    else:
        return T

V = [0.15,0.45,0.61,1.3,1.4,1.6]
(T,err) = lakeshore(V,data,True)
print('Interpolated temperatures: ',T)
print('Approximate errors: ',err)

if False:
    V = np.linspace(data[1][0],data[1][-1],1001)
    T = lakeshore(V,data)

    # Plot for sanity
    plt.figure()
    plt.plot(data[1],data[0],'r.')
    plt.plot(V,T)
    plt.show()


