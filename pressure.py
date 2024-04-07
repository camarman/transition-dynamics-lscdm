import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np


# ==================== PARAMETERS ====================
# --------------------- LsCDM
z_dag = 1.70
a_dag = 1/(1 + z_dag)

# --------------------- Initial Scale Factor
a_ini = 0.001
a_vals = np.linspace(a_ini, 1, 10000)
z_vals = 1/a_vals - 1


# ==================== NUMERICAL ANALYSIS ====================
def pressure_LsCDM(a, eta):
    return (1 / np.tanh(eta * (1 - a_dag))) * (-np.tanh(eta * (a - a_dag)) - eta * (a/3) * (np.cosh(eta * (a - a_dag)))**(-2))

pressure_lscdm_vals_1 = pressure_LsCDM(a_vals, 50)
pressure_lscdm_vals_2 = pressure_LsCDM(a_vals, 100)
pressure_lscdm_vals_3 = pressure_LsCDM(a_vals, 200)
pressure_lscdm_vals_4 = pressure_LsCDM(a_vals, 100000)


# ==================== PLOTTING ====================
# LaTeX rendering text fonts
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Adjusting the size of the figure
params = {'legend.fontsize': '30',
        'axes.labelsize': '44',
        'figure.figsize': (15, 10),
        'xtick.labelsize': '44',
        'ytick.labelsize': '44'}
pylab.rcParams.update(params)

fig, ax0 = plt.subplots()

ax0.plot(a_vals, pressure_lscdm_vals_1,  color='red', ls='-', lw=3.5, label=r'$\sigma=50$')
ax0.plot(a_vals, pressure_lscdm_vals_2,  color='blue', ls='-', lw=3.5, label=r'$\sigma=100$')
ax0.plot(a_vals, pressure_lscdm_vals_3,  color='orange', ls='-', lw=3.5, label=r'$\sigma=200$')
ax0.plot(a_vals, pressure_lscdm_vals_4,  color='black', ls='-', lw=3.5, label=r'$\sigma \rightarrow \infty$')

ax0.axvline(x=a_dag, color='brown', ls='--', lw=2.5, label=r'$a_{\dagger}=0.3704$')

ax0.axhline(y=1, color='green', ls='-.', lw=2.0)
# ax0.axhline(y=0, color='green', ls='-.', lw=2.0)
ax0.axhline(y=-1, color='green', ls='-.', lw=2.0)

# --------------------- Graph Options
# Setting Limits
ax0.set_xlim(0.275, 0.475)
ax0.set_ylim(-25, 3)

# Set Titles
ax0.set_xlabel('$a$')
ax0.set_ylabel(r'$P_{\rm tot} / \rho_{\Lambda_{\rm s}0}$')

# Tick options
ax0.set_xticks([0.275, 0.325, 0.375, 0.425, 0.475])
ax0.set_yticks([1, -1, -5, -10, -15, -20, -25])

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

# Saving the figure
plt.savefig(r'log\pressure.pdf', format='pdf', dpi=400)
plt.show()
