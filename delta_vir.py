import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np

from src.cal_delta_ta import *
from src.cal_a_vir import *
from src.eta_solver import *
from src.lscdm import *
from src.lcdm import *


# ==================== PARAMETERS ====================
# --------------------- LCDM
Om0_lcdm = Om0_finder_lcdm()
omega_lcdm = (1 - Om0_lcdm) / Om0_lcdm

# --------------------- Range of Turnaround
z_ta_range = np.arange(1.70, 4.71, 0.01)
a_ta_range = 1 / (1 + z_ta_range)


# ==================== NUMERICAL ANALYSIS ====================
def find_delta_vir_lcdm(a_ta_i, omega_i):
    delta_ta_lcdm = cal_delta_ta_LCDM(a_ta_i, omega_i)
    epsilon_lcdm = (omega_i*a_ta_i**3)/(1 + delta_ta_lcdm)
    eta_lcdm = solve_eta_lcdm(epsilon_lcdm)

    a_vir_lcdm = cal_a_vir_LCDM(a_ta_i, omega_i)
    y_vir_i = a_vir_lcdm / a_ta_i
    delta_vir_lcdm = ((1 + delta_ta_lcdm) * (y_vir_i / eta_lcdm)**3) - 1
    return delta_vir_lcdm


def find_delta_vir_lscdm(a_ta_i, y_dag):
    a_dag_i = a_ta_i * y_dag
    z_dag_i = 1 / a_dag_i - 1

    H0_lscdm = 100 * hubble_finder_lscdm(z_dag_i)
    Om0_lscdm = Om0_finder_lscdm(z_dag_i)
    omega_lscdm = (1 - Om0_lscdm) / Om0_lscdm

    if 0 < y_dag < 1:
        delta_ta_lscdm = cal_pre_delta_ta_LsCDM(a_ta_i, omega_lscdm, y_dag)
        epsilon_lscdm = (omega_lscdm * a_ta_i**3) / (1 + delta_ta_lscdm)
        eta_lscdm = solve_eta_pre_turn_lscdm(epsilon_lscdm)

        a_vir_lscdm = cal_pre_a_vir_LsCDM(a_ta_i, omega_lscdm, y_dag)
        y_vir_i = a_vir_lscdm / a_ta_i
        delta_vir_lscdm = ((1 + delta_ta_lscdm) * (y_vir_i / eta_lscdm)**3) - 1

    else:
        delta_ta_lscdm = cal_post_delta_ta_LsCDM(a_ta_i, omega_lscdm)
        epsilon_lscdm = (omega_lscdm * a_ta_i**3) / (1 + delta_ta_lscdm)

        H_pos = H0_lscdm * np.sqrt(Om0_lscdm * a_dag_i**(-3) + (1 - Om0_lscdm))
        H_neg = H0_lscdm * np.sqrt(Om0_lscdm * a_dag_i**(-3) - (1 - Om0_lscdm))
        delta_H = H_pos - H_neg

        u_dag = find_u_dag(a_ta_i, omega_lscdm, y_dag, delta_ta_lscdm)

        b1 = a_ta_i**(-3)*(1 + delta_ta_lscdm)*(1 - u_dag) + omega_lscdm*u_dag*(1 + u_dag**2)
        b2 = a_ta_i**(-3) - omega_lscdm*y_dag**3
        b3 = a_ta_i**(-3)*(1 + delta_ta_lscdm)*(1 - u_dag) + omega_lscdm*u_dag*(1 - u_dag**2)
        b4 = a_ta_i**(-3) + omega_lscdm*y_dag**3
        beta = np.sqrt((b1*b2)/(b3*b4))

        delta = 1 - ((H_pos / H_neg) * beta)
        delta0 = delta*(2 + delta)

        eta_lscdm = solve_eta_post_turn_lscdm(epsilon_lscdm, u_dag, delta0)

        a_vir_lscdm = cal_pre_a_vir_LsCDM(a_ta_i, omega_lscdm, y_dag)
        y_vir_i = a_vir_lscdm / a_ta_i
        delta_vir_lscdm = ((1 + delta_ta_lscdm) * (y_vir_i / eta_lscdm)**3) - 1
    return delta_vir_lscdm

# --------------------- LCDM
delta_vir_lcdm = np.array([1 + find_delta_vir_lcdm(a_ta_i, omega_lcdm) for a_ta_i in a_ta_range])

# --------------------- LsCDM (Pre-Turnaround)
delta_vir_lscdm_1 = np.array([1 + find_delta_vir_lscdm(a_ta_i, y_dag=0.60) for a_ta_i in a_ta_range])
delta_vir_lscdm_2 = np.array([1 + find_delta_vir_lscdm(a_ta_i, y_dag=0.70) for a_ta_i in a_ta_range])
delta_vir_lscdm_3 = np.array([1 + find_delta_vir_lscdm(a_ta_i, y_dag=0.80) for a_ta_i in a_ta_range])

# --------------------- LsCDM (Post-Turnaround)
delta_vir_lscdm_4 = np.array([1 + find_delta_vir_lscdm(a_ta_i, y_dag=1.01) for a_ta_i in a_ta_range])
delta_vir_lscdm_5 = np.array([1 + find_delta_vir_lscdm(a_ta_i, y_dag=1.05) for a_ta_i in a_ta_range])
delta_vir_lscdm_6 = np.array([1 + find_delta_vir_lscdm(a_ta_i, y_dag=1.08) for a_ta_i in a_ta_range])


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

ax0.plot(z_ta_range, delta_vir_lcdm, color='#000000', ls=(0, (5, 1)),
        lw=3.0, label=r'$\Lambda$CDM ($\Omega_{\rm m}=0.3101$)')

ax0.plot(z_ta_range, delta_vir_lscdm_1, color='#FFA500', ls='-', lw=3.0,
        label=r'$y_{\dagger}=0.60~(\rho_{\Lambda_{\rm s}}(a_{\rm ta}) > 0)$')
ax0.plot(z_ta_range, delta_vir_lscdm_2, color='#FF0000', ls='-', lw=3.0,
        label=r'$y_{\dagger}=0.70~(\rho_{\Lambda_{\rm s}}(a_{\rm ta}) > 0)$')
ax0.plot(z_ta_range, delta_vir_lscdm_3, color='#800000', ls='-', lw=3.0,
        label=r'$y_{\dagger}=0.80~(\rho_{\Lambda_{\rm s}}(a_{\rm ta}) > 0)$')

ax0.plot(z_ta_range, delta_vir_lscdm_4, color='#000080', ls='-', lw=3.0,
        label=r'$y_{\dagger}=1.01~(\rho_{\Lambda_{\rm s}}(a_{\rm ta}) < 0)$')
ax0.plot(z_ta_range, delta_vir_lscdm_5, color='#0000ff', ls='-', lw=3.0,
        label=r'$y_{\dagger}=1.05~(\rho_{\Lambda_{\rm s}}(a_{\rm ta}) < 0)$')
ax0.plot(z_ta_range, delta_vir_lscdm_6, color='#00ffff', ls='-', lw=3.0,
        label=r'$y_{\dagger}=1.08~(\rho_{\Lambda_{\rm s}}(a_{\rm ta}) < 0)$')

# Setting labels
ax0.set_xlabel(r'$z_{\rm ta}$')
ax0.set_ylabel(r'$1 + \delta_{\rm vir}$')

# Setting limits
ax0.set_xlim(1.7, 4.7)
ax0.set_ylim(165, 245)

# Tick options
ax0.set_xticks([1.7, 2.0, 2.3, 2.6, 2.9, 3.2, 3.5, 3.8, 4.1, 4.4, 4.7])
ax0.set_yticks([165, 175, 185, 195, 205, 215, 225, 235, 245])

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
plt.savefig(r'log\deltaVir.pdf', format='pdf', dpi=400)
plt.show()
