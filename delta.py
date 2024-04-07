import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np

from src.cal_delta_ta import *
from src.eta_solver import *
from src.lscdm import *


# ==================== PARAMETERS ====================
# --------------------- Range of Turnaround
z_ta_range = np.arange(1.70, 4.71, 0.01)
a_ta_range = 1 / (1 + z_ta_range)


# ==================== NUMERICAL ANALYSIS ====================
def cal_delta(a_ta_i, y_dag):
    a_dag_i = a_ta_i * y_dag
    z_dag_i = 1 / a_dag_i - 1

    H0_lscdm = 100 * h0_finder_LsCDM(z_dag_i)
    Om0_lscdm = Om0_finder_LsCDM(z_dag_i)
    omega_lscdm = (1 - Om0_lscdm) / Om0_lscdm

    delta_ta_lscdm = cal_post_delta_ta_LsCDM(a_ta_i, omega_lscdm)

    H_pos = H0_lscdm * np.sqrt(Om0_lscdm * a_dag_i**(-3) + (1 - Om0_lscdm))
    H_neg = H0_lscdm * np.sqrt(Om0_lscdm * a_dag_i**(-3) - (1 - Om0_lscdm))

    u_dag = find_u_dag(a_ta_i, omega_lscdm, y_dag, delta_ta_lscdm)

    b1 = a_ta_i**(-3)*(1 + delta_ta_lscdm)*(1 - u_dag) + omega_lscdm*u_dag*(1 + u_dag**2)
    b2 = a_ta_i**(-3) - omega_lscdm*y_dag**3
    b3 = a_ta_i**(-3)*(1 + delta_ta_lscdm)*(1 - u_dag) + omega_lscdm*u_dag*(1 - u_dag**2)
    b4 = a_ta_i**(-3) + omega_lscdm*y_dag**3
    beta = np.sqrt((b1*b2)/(b3*b4))

    delta = 1 - ((H_pos / H_neg) * beta)
    return delta

# ---------------------
delta_ta_lscdm_1 = np.array([cal_delta(a_ta_i, y_dag=1.01) for a_ta_i in a_ta_range])
delta_ta_lscdm_2 = np.array([cal_delta(a_ta_i, y_dag=1.02) for a_ta_i in a_ta_range])
delta_ta_lscdm_3 = np.array([cal_delta(a_ta_i, y_dag=1.04) for a_ta_i in a_ta_range])
delta_ta_lscdm_4 = np.array([cal_delta(a_ta_i, y_dag=1.06) for a_ta_i in a_ta_range])
delta_ta_lscdm_5 = np.array([cal_delta(a_ta_i, y_dag=1.08) for a_ta_i in a_ta_range])


# ==================== PLOT ====================
# LaTeX rendering text fonts
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Adjusting the size of the figure
params = {'legend.fontsize': '32',
        'axes.labelsize': '44',
        'figure.figsize': (15, 10),
        'xtick.labelsize': '44',
        'ytick.labelsize': '44'}
pylab.rcParams.update(params)

fig, ax0 = plt.subplots()

ax0.plot(z_ta_range, delta_ta_lscdm_1, color='#D22730',
        ls='-', lw=3.5, label=r'$y_{\dagger}=1.01$')
ax0.plot(z_ta_range, delta_ta_lscdm_2, color='#f032e6',
        ls=(0, (3, 5, 1, 5, 1, 5)), lw=3.5, label=r'$y_{\dagger}=1.02$')
ax0.plot(z_ta_range, delta_ta_lscdm_3, color='#006400',
        ls='--', lw=3.5, label=r'$y_{\dagger}=1.04$')
ax0.plot(z_ta_range, delta_ta_lscdm_4, color='#FFAD00',
        ls='-.',  lw=3.5, label=r'$y_{\dagger}=1.06$')
ax0.plot(z_ta_range, delta_ta_lscdm_5, color='#000085',
        ls=(5, (10, 3)), lw=3.5, label=r'$y_{\dagger}=1.08$')

# --------------------- Delta Area
y_start_end = np.linspace(-1, 0, len(z_ta_range))

x_fill_start = 1.7
x_fill_end = 4.7
ax0.axvline(x_fill_start, color='gray', ls='-.', lw=1.5)
ax0.axvline(x_fill_end, color='gray', ls='-.', lw=1.5)
ax0.fill_betweenx(y_start_end, x_fill_start, x_fill_end, color='gray',
                alpha=0.4, label=r'$-1 \leq \Delta < 0$')

# Setting labels
ax0.set_xlabel(r'$z_{\rm ta}$')
ax0.set_ylabel(r'$\Delta$')

# Setting limit
ax0.set_xlim(1.70, 4.70)
ax0.set_ylim(-18.00, 0.00)

# Tick options
ax0.set_xticks([1.7, 2.0, 2.3, 2.6, 2.9, 3.2, 3.5, 3.8, 4.1, 4.4, 4.7])
ax0.set_yticks([0, -2, -4, -6, -8, -10, -12, -14, -16, -18])

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
plt.savefig(r'log\delta.pdf', format='pdf', dpi=400)
plt.show()
