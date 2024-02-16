import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np

from src.lscdm import Om0_finder_lscdm


# ==================== PARAMETERS ====================
# --------------------- LsCDM
z_dag = 1.70
a_dag = 1/(1 + z_dag)

Om0_lscdm = Om0_finder_lscdm(z_dag)
Om0_Ode0_ratio = Om0_lscdm / (1 - Om0_lscdm)
rho_tot_a_dag = Om0_Ode0_ratio * a_dag**(-3)

# --------------------- Initial Scale Factor
a_ini = 0.30
a_vals = np.linspace(a_ini, 1, 10000)
z_vals = 1/a_vals - 1


# ==================== NUMERICAL ANALYSIS ====================
def rho_lscdm(a, eta):
    return Om0_Ode0_ratio*a**(-3) + np.tanh(eta*(a - a_dag)) / np.tanh(eta * (1 - a_dag))

rho_lscdm_vals_1 = rho_lscdm(a_vals, 50)
rho_lscdm_vals_2 = rho_lscdm(a_vals, 100)
rho_lscdm_vals_3 = rho_lscdm(a_vals, 200)
rho_lscdm_vals_4 = rho_lscdm(a_vals, 100000)


# ==================== PLOT ====================
# LaTeX rendering text fonts
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# adjusting size of the figure
params = {
            'legend.fontsize': '22.5',
            'axes.labelsize': '30',
            'figure.figsize': (15, 10),
            'xtick.labelsize': '25',
            'ytick.labelsize': '25'
        }

pylab.rcParams.update(params)

fig, ax0 = plt.subplots()

ax0.plot(a_vals, rho_lscdm_vals_1,  color='red', ls='-',
        lw=3.0, label=r'$\sigma=50$')
ax0.plot(a_vals, rho_lscdm_vals_2,  color='blue', ls='-',
        lw=3.0, label=r'$\sigma=100$')
ax0.plot(a_vals, rho_lscdm_vals_3,  color='orange', ls='-',
        lw=3.0, label=r'$\sigma=200$')
ax0.plot(a_vals, rho_lscdm_vals_4,  color='black', ls='-',
        lw=3.0, label=r'$\sigma \rightarrow \infty$')

ax0.axvline(x=a_dag, color='brown', ls='--', lw=2.0, label=r'$a_{\dagger}=0.3704$')
ax0.axhline(y=rho_tot_a_dag, color='green', ls='-.', lw=2.5,
        label=r'$\rho_{\rm tot} \equiv \rho_{\rm m0}a_{\dagger}^{-3}$')

# --------------------- Graph Options
# Setting Limits
ax0.set_xlim(0.275, 0.475)
ax0.set_ylim(0, 12)

# Set Titles
ax0.set_xlabel('$a$')
ax0.set_ylabel(r'$\rho_{\rm tot} / \rho_{\Lambda_{\rm s}0}$')

# Tick options
ax0.set_xticks([0.275, 0.300, 0.325, 0.350, 0.375, 0.400, 0.425, 0.450, 0.475])
ax0.xaxis.set_ticks_position('both')
ax0.yaxis.set_ticks_position('both')

ax0.xaxis.set_tick_params(which='major', width=1.5, size=13.0, direction='in')
ax0.xaxis.set_tick_params(which='minor', width=1.0, size=6.50, direction='in')
ax0.yaxis.set_tick_params(which='major', width=1.5, size=13.0, direction='in')
ax0.yaxis.set_tick_params(which='minor', width=1.0, size=6.50, direction='in')
ax0.minorticks_on()

# Other Options
ax0.legend()
plt.tight_layout()
plt.savefig(r'log\energyDensity.pdf', format='pdf', dpi=400)
plt.show()
