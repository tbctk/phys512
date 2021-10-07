import numpy as np

def paraboloid(m,x,y):
    z = m[0]*(x**2+y**2)-m[1]*x-m[2]*y+m[3]
    return z

def get_pars(m):
    a = m[0]
    x_0 = 0.5*m[1]/a
    y_0 = 0.5*m[2]/a
    z_0 = m[3]-a*(x_0**2+y_0**2)
    return a,x_0,y_0,z_0

# data[0] = x, data[1] = y, data[2] = z
data = np.loadtxt('dish_zenith.txt').T
x = data[0]
y = data[1]
z = data[2]

nd = len(data[0])
nm = 4
A = np.zeros([nd,nm])
A[:,0] = x**2+y**2
A[:,1] = -x
A[:,2] = -y
A[:,3] = 1

# Assuming no noise!!
lhs = A.T@A
rhs = A.T@z
fitp = np.linalg.inv(lhs)@rhs

p = get_pars(fitp)
print("Modfied parameters: a=",fitp[0],", b=",fitp[1],", c=",fitp[2],", d=",fitp[3])
print("Original parameters: a=",p[0]," x_0=",p[1],", y_0=",p[2],", z_0=",p[3])

# Now find noise estimate based on this model

z_fit = paraboloid(fitp,x,y)
noise = np.std(z_fit-z)
N = np.eye(len(z))*noise**2
Ninv = np.eye(len(z))*noise**-2
tmp = np.linalg.inv(A.T@Ninv@A)
p_errs = np.sqrt(np.diag(tmp))
print("Error on a: ",p_errs[0])

# Get error in f
dfda = (4*fitp[0]**2)**-1
f_err = dfda*p_errs[0]
print("Focal length: ",(4*fitp[0])**-1,"Error on focal length: ",f_err)


