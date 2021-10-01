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

realp = get_pars(fitp)

print("Modfied parameters: a=",fitp[0],", b=",fitp[1],", c=",fitp[2],", d=",fitp[3])
print("Original parameters: a=",realp[0]," x_0=",realp[1],", y_0=",realp[2],", z_0=",realp[3])
