import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import math

x = np.array([0, 0.4,0.45,0.6,0.61,0.62,0.63,0.9,1.0])
y = np.sin(3*x)

sample_increase = 4

xs = np.empty((sample_increase*(x.size-1),1))
ys = np.empty((sample_increase*(x.size-1),1))
for i in range(x.size-1):
    xs[i*sample_increase:i*sample_increase+sample_increase] = np.linspace(x[i],x[i+1],sample_increase).reshape(sample_increase,1)
    ys[i*sample_increase:i*sample_increase+sample_increase] = np.linspace(y[i],y[i+1],sample_increase).reshape(sample_increase,1)
    
spl = UnivariateSpline(xs, ys)
spl.set_smoothing_factor(0.5)

plt.plot(xs, spl(xs), 'g', lw=3)
plt.plot(x,y)
plt.show()
