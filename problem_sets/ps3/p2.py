import numpy as np
from scipy import integrate

## PART A

def half_life_array_gen():
    day2yr = 1/365
    hr2day = 1/24
    min2hr = 1/60
    sec2min = 1/60

    hl = [
        4.468e9,
        24.1*day2yr,
        6.7*hr2day*day2yr,
        245500,
        75380,
        1600,
        3.8235*day2yr,
        3.1*min2hr*hr2day*day2yr,
        26.81*min2hr*hr2day*day2yr,
        19.91*min2hr*hr2day*day2yr,
        164.3e-6*sec2min*min2hr*hr2day*day2yr,
        22.3,
        5.015,
        138.376*day2yr
    ]
    return np.asarray(hl)

half_life = half_life_array_gen()
# I am normalizing the array to the max half-life to make life easier later.
# If we choose not to normalize it, then we will have a proper time-scale (in
# years).
if True:
    half_life = half_life/np.max(half_life)

def fun(t,y):
    global half_life
    n = len(half_life)
    rates = y[0:n]/half_life
    dydt = np.zeros(n+1)
    dydt[0:n] = -rates
    dydt[1:n+1] = dydt[1:n+1]+rates
    return dydt

y0 = np.zeros(len(half_life)+1); y0[0] = 1
t0,t1 = 0,1

ans_a = integrate.solve_ivp(fun,(t0,t1),y0,method='Radau')
#ans2 = integrate.solve_ivp(fun,(t0,t1),y0)

## PART B

from matplotlib import pyplot as plt

t0,t1 = 0,0.001
npt = 1000

t = np.linspace(t0,t1,npt)

ans_b = integrate.solve_ivp(fun,(t0,t1),y0,method='Radau',t_eval=t)

rat1 = ans_b.y[-1]/ans_b.y[0]
rat2 = ans_b.y[4]/ans_b.y[3]

#plt.figure()
#plt.plot(t,rat1)
#plt.savefig('p2_fig1.png',dpi=150)
#plt.clf()
#plt.plot(t,rat2)
#plt.show()
#plt.savefig('p2_fig2.png',dpi=150)
