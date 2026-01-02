# From: https://xuwd11.github.io/am207/wiki/stratlab.html

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec          # For unequal plot boxes
import scipy.optimize
import scipy.stats

r = np.random
r.seed(42)


plt.figure(figsize=[14,8])
Y = lambda x: x/(x**2+1.0);
intY = lambda x: np.log(x**2 + 1.0)/2.0;



N = 10000
Nrep = 1000
Ntry = 1000
Ns = 10   # number of strata 

xmin=0
xmax =10
step = (xmax - xmin)/Ns
# analytic solution 
Ic = intY(xmax)-intY(xmin)


Imc = np.zeros(Nrep)
Is = np.zeros(Nrep)
Is2 = np.zeros(Nrep)

## Ploting the original functions 
plt.subplot(1,2,1)
x = np.linspace(0,10,100)
plt.plot(x, Y(x), label=u'$x/(x**2+1)$')
for j in range(Ns+1):
    plt.axvline(xmin + j*step, 0, 1, color='r', alpha=0.2)
    
sigmas = np.zeros(Ns)
Utry = np.random.uniform(low=xmin, high=xmax, size=Ntry)
Ytry = Y(Utry)
Umin = 0 
Umax = step
for reg in np.arange(0,Ns):
    localmask = (Utry >= Umin) & (Utry < Umax)
    sigmas[reg] = np.std(Ytry[localmask])
    Umin = Umin + step
    Umax = Umin + step
nums = np.ceil(N*sigmas/np.sum(sigmas)).astype(int)
print(sigmas, nums, np.sum(nums))
    
for k in np.arange(0,Nrep):
    # First lets do it with mean MC method 
    U = np.random.uniform(low=xmin, high=xmax, size=N)
    Imc[k] = (xmax-xmin)* np.mean(Y(U))

    #stratified it in Ns regions
    Umin = 0 
    Umax = step
    Ii = 0
    I2i = 0
    for reg in np.arange(0,Ns):
        x = np.random.uniform(low=Umin, high=Umax, size=N//Ns);
        Ii = Ii + (Umax-Umin)*np.mean(Y(x))
        x2 = np.random.uniform(low=Umin, high=Umax, size=nums[reg]);
        I2i = I2i + (Umax-Umin)*np.mean(Y(x2))
        Umin = Umin + step
        Umax = Umin + step


    Is[k] = Ii
    Is2[k] = I2i

plt.subplot(1,2,2)
plt.hist(Imc,30, histtype='stepfilled', label=u'Normal MC', alpha=0.1)
plt.hist(Is, 30, histtype='stepfilled', label=u'Stratified', alpha=0.3)
plt.hist(Is2, 30, histtype='stepfilled', label=u'Stratified (sigmaprop)', alpha=0.5)


plt.legend()

print(np.std(Imc), np.std(Is), np.std(Is2))
