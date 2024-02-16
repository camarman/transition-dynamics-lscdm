import numpy as np
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import scipy as sp

from src.lscdm import *
from src.lcdm import *


# ==================== NUMERICAL ANALYSIS ====================
z_dag_vals = np.arange(1.50, 4.51, 0.01)
Om0_lcdm = Om0_finder_lcdm()
Om0_lscdm = np.array([Om0_finder_lscdm(z_dag_i) for z_dag_i in z_dag_vals])

z_ta = 2
a_ta = 1 / (1 + z_ta)

y_dag_points = np.array([0.60, 0.70, 0.80, 1.01, 1.05, 1.08])
z_dag_points = ((1 + z_ta) / y_dag_points) - 1
Om0_points = np.array([Om0_finder_lscdm(z_dag_i) for z_dag_i in z_dag_points])


# ==================== PLOT ====================
# LaTeX rendering text fonts
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Adjusting size of the figure
params = {
            'legend.fontsize': '22.5',
            'axes.labelsize': '30',
            'figure.figsize': (15, 10),
            'xtick.labelsize': '25',
            'ytick.labelsize': '25'
        }

pylab.rcParams.update(params)

fig, ax0 = plt.subplots()

ax0.axhline(y=Om0_lcdm, color='#000000', ls=(0, (5, 1)),
            lw=3.0, label=r'$\Lambda$CDM')

ax0.plot(z_dag_vals, Om0_lscdm, color='#000085', ls='-',
        alpha=0.3, lw=1.5, label=r'$\Lambda_{\rm s}$CDM')

ax0.scatter(z_dag_points[0], Om0_points[0], facecolor='#FFA500', edgecolors='black',
        linewidth=1.0, s=100, label=r'$y_{\dagger}=0.60$')
ax0.scatter(z_dag_points[1], Om0_points[1], facecolor='#Ff0000', edgecolors='black',
        linewidth=1.0, s=100, label=r'$y_{\dagger}=0.70$')
ax0.scatter(z_dag_points[2], Om0_points[2], facecolor='#800000', edgecolors='black',
        linewidth=1.0, s=100, label=r'$y_{\dagger}=0.80$')

ax0.scatter(z_dag_points[3], Om0_points[3], facecolor='#000080', edgecolors='black',
        linewidth=1.0, s=100, label=r'$y_{\dagger}=1.01$')
ax0.scatter(z_dag_points[4], Om0_points[4], facecolor='#0000ff', edgecolors='black',
        linewidth=1.0, s=100, label=r'$y_{\dagger}=1.05$')
ax0.scatter(z_dag_points[5], Om0_points[5], facecolor='#00ffff', edgecolors='black',
        linewidth=1.0, s=100, label=r'$y_{\dagger}=1.08$')

# Setting labels
ax0.set_xlabel(r'$z_{\dagger}$')
ax0.set_ylabel(r'$\Omega_{\rm m0}$')

# Setting limits
ax0.set_xlim(1.50, 4.50)
ax0.set_ylim(0.255, 0.315)

# Tick options
# Set the desired ticks on the x/y-axis
ax0.set_xticks([1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0,
                3.25, 3.5, 3.75, 4.0, 4.25, 4.5])
ax0.set_yticks([0.255, 0.260, 0.265, 0.270, 0.275, 0.280,
                0.285, 0.290, 0.295, 0.300, 0.305, 0.310, 0.315])

ax0.xaxis.set_ticks_position('both')
ax0.yaxis.set_ticks_position('both')

ax0.xaxis.set_tick_params(which='major', width=1.5, size=13.0, direction='in')
ax0.xaxis.set_tick_params(which='minor', width=1.0, size=6.50, direction='in')
ax0.yaxis.set_tick_params(which='major', width=1.5, size=13.0, direction='in')
ax0.yaxis.set_tick_params(which='minor', width=1.0, size=6.50, direction='in')
ax0.minorticks_on()

# Other settings
ax0.legend()
plt.tight_layout()

# Saving the figure
plt.savefig(r'log\omegam0.pdf', format='pdf', dpi=400)
plt.show()
