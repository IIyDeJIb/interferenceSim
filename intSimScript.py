import numpy as np
import pandas as pd
from scipy import special as sp
from matplotlib import pyplot as plt

# Given
# Reservoir and Well Data
phi = 0.1
k = 10  # md
h = 200  # ft
ct = 1e-6  # psi^-1
mu = 1  # cp
B = 1.1  # rb/stb
egrad = 0.4  # psi/ft emulsion gradient
# Flow rate (simulates flow from formation with a sucker rod pump)
maxq = 100.  # stb/d
dq = 1.  # stb/d

# Distance to the point of observation
r = 600.  # ft

# Pump cycle
tStart = 0  # hrs
tEnd = 20 # hrs

# Time of observations after test start
tObserve = np.linspace(0, 50, 500)  # hrs

# Generate a rate and time change series
# Scenario 1: Cyclical SRP operation
# q = np.tile(np.concatenate((np.linspace(0,maxq,int(maxq/dq+1)), np.linspace(maxq-dq,0,int(maxq/dq)))),100)

# Scenario 2: Keep pumped-off
q = np.concatenate((np.linspace(0,maxq,int(maxq/dq+1)), np.repeat(maxq,10*int(maxq/dq+1)),np.linspace(maxq-dq,0,int(maxq/dq))))

tSched = np.linspace(tStart, tEnd, len(q) - 1)  # hrs

# Calculate difference arrays for q and t for superposition calculations
qDiff = np.diff(q)

# Pressure response calculation
dp = pd.Series(index=tObserve)

for timeObs in dp.index: # todo widen the window of observations
    m = 70.6 * qDiff * B * mu / (k * h)
    tSupp = timeObs - tSched
    m[tSupp<0]=0
    tSupp[tSupp<=0]=1
    ei = sp.expi(-948 * phi * ct * r ** 2 / (k * tSupp))
    dp.loc[timeObs] = np.sum(m * ei)

# Visualization
plt.figure(1)
plt.subplot(211)
plt.plot(tSched,q[:-1], color='red')
plt.title('Testing Well')
plt.xlabel('Time, hrs')
plt.ylabel('Flow Rate, stb/d')
plt.subplot(212)
plt.plot(dp/egrad)
plt.title('Observation Well')
plt.xlabel('Time, hrs')
plt.ylabel('Fluid level change, ft')
plt.show()
