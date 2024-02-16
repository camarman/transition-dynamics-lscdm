import numpy as np
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt

from src.cal_delta_ta import *
from src.lscdm import *
from src.lcdm import *


# ==================== PARAMETERS ====================
# --------------------- LCDM
Om0_lcdm = Om0_finder_lcdm()
omega_lcdm = (1 - Om0_lcdm) / Om0_lcdm

# --------------------- Range of Turnaround
z_ta_range = np.arange(1.7, 4.71, 0.01)
a_ta_range = 1 / (1 + z_ta_range)


# ==================== NUMERICAL ANALYSIS ====================
def cal_R_ratio(a_ta_i, y_dag):
    a_dag_i = a_ta_i * y_dag
    z_dag_i = 1 / a_dag_i - 1

    delta_ta_lcdm = cal_delta_ta_LCDM(a_ta_i, omega_lcdm)

    Om0_lscdm = Om0_finder_lscdm(z_dag_i)
    omega_lscdm = (1 - Om0_lscdm) / Om0_lscdm

    if 0 < y_dag < 1:
        delta_ta_lscdm = cal_pre_delta_ta_LsCDM(a_ta_i, omega_lscdm, y_dag)
    else:
        delta_ta_lscdm = cal_post_delta_ta_LsCDM(a_ta_i, omega_lscdm)
    R_ratio = (1 + delta_ta_lscdm)**(-1/3) / (1 + delta_ta_lcdm)**(-1/3)
    return R_ratio

# --------------------- LsCDM (Pre-Turnaround)
R_ratio_lscdm_1 = np.array([cal_R_ratio(a_ta_i, y_dag=0.60) for a_ta_i in a_ta_range])
R_ratio_lscdm_2 = np.array([cal_R_ratio(a_ta_i, y_dag=0.70) for a_ta_i in a_ta_range])
R_ratio_lscdm_3 = np.array([cal_R_ratio(a_ta_i, y_dag=0.80) for a_ta_i in a_ta_range])

# --------------------- LsCDM (Post-Turnaround)
R_ratio_lscdm_4 = np.array([cal_R_ratio(a_ta_i, y_dag=1.01) for a_ta_i in a_ta_range])
R_ratio_lscdm_5 = np.array([cal_R_ratio(a_ta_i, y_dag=1.05) for a_ta_i in a_ta_range])
R_ratio_lscdm_6 = np.array([cal_R_ratio(a_ta_i, y_dag=1.08) for a_ta_i in a_ta_range])


# ==================== PLOTTING ====================
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

ax0.axhline(y=1, color='#000000', ls=(0, (5, 1)), lw=3.0,
            label=r'$\Lambda$CDM ($\Omega_{\rm m}=0.3101$)')

ax0.plot(z_ta_range, R_ratio_lscdm_1, color='#FFA500', ls='-', lw=3.0,
        label=r'$y_{\dagger}=0.60~(\rho_{\Lambda_{\rm s}}(a_{\rm ta}) > 0)$')
ax0.plot(z_ta_range, R_ratio_lscdm_2, color='#FF0000', ls='-', lw=3.0,
        label=r'$y_{\dagger}=0.70~(\rho_{\Lambda_{\rm s}}(a_{\rm ta}) > 0)$')
ax0.plot(z_ta_range, R_ratio_lscdm_3, color='#800000', ls='-', lw=3.0,
        label=r'$y_{\dagger}=0.80~(\rho_{\Lambda_{\rm s}}(a_{\rm ta}) > 0)$')

ax0.plot(z_ta_range, R_ratio_lscdm_4, color='#000080', ls='-', lw=3.0,
        label=r'$y_{\dagger}=1.01~(\rho_{\Lambda_{\rm s}}(a_{\rm ta}) < 0)$')
ax0.plot(z_ta_range, R_ratio_lscdm_5, color='#0000ff', ls='-', lw=3.0,
        label=r'$y_{\dagger}=1.05~(\rho_{\Lambda_{\rm s}}(a_{\rm ta}) < 0)$')
ax0.plot(z_ta_range, R_ratio_lscdm_6, color='#00ffff', ls='-', lw=3.0,
        label=r'$y_{\dagger}=1.08~(\rho_{\Lambda_{\rm s}}(a_{\rm ta}) < 0)$')

# Setting labels
ax0.set_xlabel(r'$z_{\rm ta}$')
ax0.set_ylabel(r'$R_{\rm p,ta}^{\Lambda_{\rm s}} / R_{\rm p,ta}^{\Lambda}$')

# Setting limits
ax0.set_xlim(1.700, 4.700)
ax0.set_ylim(0.970, 1.060)

# Tick options
ax0.set_xticks([1.7, 2.0, 2.3, 2.6, 2.9, 3.2, 3.5, 3.8, 4.1, 4.4, 4.7])
ax0.set_yticks([0.970, 0.980, 0.990, 1.00, 1.010, 1.020, 1.030, 1.040, 1.050, 1.060])

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
plt.savefig(r'log\RTurnaroundRatio.pdf', format='pdf', dpi=400)
plt.show()
