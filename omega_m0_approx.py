import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import scipy as sp

from src.lscdm import *


z_dag_vals = np.arange(1.50, 11.51, 0.01)
Om0_vals_lscdm_true = np.array([Om0_finder_lscdm(z_dag_i) for z_dag_i in z_dag_vals])

# Fit Function
def fit_func(x, c0, c1, c2, c3):
	return c0 + c1/x + c2/x**2 + c3/x**3

# Fit the function to the data
params, covariance = sp.optimize.curve_fit(fit_func, z_dag_vals, Om0_vals_lscdm_true)

# Extract the fitted parameters
c0, c1, c2, c3 = params
params_rounded = np.array([round(i, 4) for i in params])
print(r'(c0,c1,c2,c3)=({0},{1},{2},{3})'.format(*params_rounded))
