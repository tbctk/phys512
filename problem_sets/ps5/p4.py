import numpy as np

## PROBLEM 4
def conv_safe(arr1,arr2):
    n1 = len(arr1)
    n2 = len(arr2)
    n_safe = n1+n2
    arr1_safe = np.zeros(n_safe)
    arr2_safe = np.zeros(n_safe)
    arr1_safe = np.concatenate((np.zeros(n2),arr1))
    arr2_safe = np.concatenate((np.zeros(n1),arr2))
    return convolve(arr1_safe,arr2_safe)

pars = [0,1,50,10]
x = np.arange(100)
y = gauss(pars,x)
pars2 = [0,1,20,4]
x2 = np.arange(60)
y2 = gauss(pars2,x2)
conv = conv_safe(y,y2)
plt.figure()
plt.plot(conv)
#plt.savefig("safe_convolution.png")
print(len(conv))
