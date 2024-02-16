import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np

from src.cal_delta_ta import *
from src.eta_solver import *
from src.lscdm import *
from src.lcdm import *


# ==================== PARAMETERS ====================
# --------------------- LCDM
Om0_lcdm = Om0_finder_lcdm()
omega_lcdm = (1 - Om0_lcdm) / Om0_lcdm

# --------------------- Range of the Turnaround
z_ta_range = np.arange(1.70, 4.71, 0.01)
a_ta_range = 1 / (1 + z_ta_range)


# ==================== NUMERICAL ANALYSIS ====================
def find_delta_ta_lcdm(a_ta_i, omega_i):
    delta_ta_lcdm = cal_delta_ta_LCDM(a_ta_i, omega_i)
    return delta_ta_lcdm


def find_delta_ta_lscdm(a_ta_i, y_dag):
    a_dag_i = a_ta_i * y_dag
    z_dag_i = 1 / a_dag_i - 1

    Om0_lscdm = Om0_finder_lscdm(z_dag_i)
    omega_lscdm = (1 - Om0_lscdm) / Om0_lscdm

    if 0 < y_dag < 1:
        delta_ta_lscdm = cal_pre_delta_ta_LsCDM(a_ta_i, omega_lscdm, y_dag)
    else:
        delta_ta_lscdm = cal_post_delta_ta_LsCDM(a_ta_i, omega_lscdm)
    return delta_ta_lscdm

# --------------------- LCDM
delta_ta_lcdm = np.array([1 + find_delta_ta_lcdm(a_ta_i, omega_lcdm) for a_ta_i in a_ta_range])

# --------------------- LsCDM (Pre-Turnaround)
delta_ta_lscdm_1 = np.array([1 + find_delta_ta_lscdm(a_ta_i, y_dag=0.60) for a_ta_i in a_ta_range])
delta_ta_lscdm_2 = np.array([1 + find_delta_ta_lscdm(a_ta_i, y_dag=0.70) for a_ta_i in a_ta_range])
delta_ta_lscdm_3 = np.array([1 + find_delta_ta_lscdm(a_ta_i, y_dag=0.80) for a_ta_i in a_ta_range])

# --------------------- LsCDM (Post-Turnaround)
delta_ta_lscdm_4 = np.array([1 + find_delta_ta_lscdm(a_ta_i, y_dag=1.01) for a_ta_i in a_ta_range])
delta_ta_lscdm_5 = np.array([1 + find_delta_ta_lscdm(a_ta_i, y_dag=1.05) for a_ta_i in a_ta_range])
delta_ta_lscdm_6 = np.array([1 + find_delta_ta_lscdm(a_ta_i, y_dag=1.08) for a_ta_i in a_ta_range])


# ==================== PLOT ====================
# LaTeX rendering text fonts
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Adjusting the size of the figure
params = {
            'legend.fontsize': '22.5',
            'axes.labelsize': '30',
            'figure.figsize': (15, 10),
            'xtick.labelsize': '25',
            'ytick.labelsize': '25'
        }

pylab.rcParams.update(params)

fig, ax0 = plt.subplots()

ax0.plot(z_ta_range, delta_ta_lcdm, color='#000000', ls=(0, (5, 1)),
        lw=3.0, label=r'$\Lambda$CDM ($\Omega_{\rm m}=0.3101$)')

ax0.plot(z_ta_range, delta_ta_lscdm_1, color='#FFA500', ls='-', lw=3.0,
        label=r'$y_{\dagger}=0.60~(\rho_{\Lambda_{\rm s}}(a_{\rm ta}) > 0)$')
ax0.plot(z_ta_range, delta_ta_lscdm_2, color='#FF0000', ls='-', lw=3.0,
        label=r'$y_{\dagger}=0.70~(\rho_{\Lambda_{\rm s}}(a_{\rm ta}) > 0)$')
ax0.plot(z_ta_range, delta_ta_lscdm_3, color='#800000', ls='-', lw=3.0,
        label=r'$y_{\dagger}=0.80~(\rho_{\Lambda_{\rm s}}(a_{\rm ta}) > 0)$')

ax0.plot(z_ta_range, delta_ta_lscdm_4, color='#000080', ls='-', lw=3.0,
        label=r'$y_{\dagger}=1.01~(\rho_{\Lambda_{\rm s}}(a_{\rm ta}) < 0)$')
ax0.plot(z_ta_range, delta_ta_lscdm_5, color='#0000ff', ls='-', lw=3.0,
        label=r'$y_{\dagger}=1.05~(\rho_{\Lambda_{\rm s}}(a_{\rm ta}) < 0)$')
ax0.plot(z_ta_range, delta_ta_lscdm_6, color='#00ffff', ls='-', lw=3.0,
        label=r'$y_{\dagger}=1.08~(\rho_{\Lambda_{\rm s}}(a_{\rm ta}) < 0)$')

# Setting labels
ax0.set_xlabel(r'$z_{\rm ta}$')
ax0.set_ylabel(r'$1 + \delta_{\rm ta}$')

# Setting limits
ax0.set_xlim(1.70, 4.70)
ax0.set_ylim(5.00, 6.50)

# Tick options
ax0.set_xticks([1.70, 2.0, 2.30, 2.60, 2.90, 3.20,
                3.50, 3.80, 4.10, 4.40, 4.70])
ax0.set_yticks([5.00, 5.25, 5.50, 5.75, 6.00, 6.25, 6.50])

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
plt.savefig(r'log\deltaTurnaround.pdf', format='pdf', dpi=400)
plt.show()
