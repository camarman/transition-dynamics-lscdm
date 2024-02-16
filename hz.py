import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np

from src.lscdm import *
from src.lcdm import *


# ==================== PARAMETERS ====================
# --------------------- LCDM
h0_lcdm = hubble_finder_lcdm()
Om0_lcdm = Om0_finder_lcdm()

z_ta = 2.0
a_ta = 1 / (1 + z_ta)


# ==================== HUBBLE FUNCTIONS ====================
def hz_lcdm(z):
    H0_lcdm = h0_lcdm * 100
    return H0_lcdm * np.sqrt(Om0_lcdm * (1+z)**3 + (1 - Om0_lcdm))


def hz_lscdm(z, y_dag):
	#y_dagger is given in terms of a_dagger
    a_dag = y_dag * a_ta
    z_dag = 1/a_dag - 1
    h0_lscdm = hubble_finder_lscdm(z_dag)
    Om0_lscdm = Om0_finder_lscdm(z_dag)

    H0_lscdm = h0_lscdm * 100
    return H0_lscdm * np.sqrt(Om0_lscdm * (1+z)**3 + (1 - Om0_lscdm) * np.sign(z_dag - z))


# ==================== NUMERICAL ANALYSIS ====================
z_vals = np.arange(0, 4.255, 0.005)
Hz_vals_lcdm = np.array([hz_lcdm(z)/(1 + z) for z in z_vals])

Hz_vals_lscdm_y_dag_1 = np.array([hz_lscdm(z, 0.60) / (1 + z) for z in z_vals])
Hz_vals_lscdm_y_dag_2 = np.array([hz_lscdm(z, 0.70) / (1 + z) for z in z_vals])
Hz_vals_lscdm_y_dag_3 = np.array([hz_lscdm(z, 0.80) / (1 + z) for z in z_vals])
Hz_vals_lscdm_y_dag_4 = np.array([hz_lscdm(z, 1.01) / (1 + z) for z in z_vals])
Hz_vals_lscdm_y_dag_5 = np.array([hz_lscdm(z, 1.05) / (1 + z) for z in z_vals])
Hz_vals_lscdm_y_dag_6 = np.array([hz_lscdm(z, 1.08) / (1 + z) for z in z_vals])


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

ax0.plot(z_vals, Hz_vals_lcdm, color='black',
        linestyle=(0, (5, 1)), lw=3.0, label=r'$\Lambda$CDM ($\Omega_{\rm m}=0.3101$)')

ax0.plot(z_vals, Hz_vals_lscdm_y_dag_1, color='#FFA500', ls='-',
        lw=3.0, label=r'$y_{\dagger}=0.60~(\rho_{\Lambda_{\rm s}}(a_{\rm ta}) > 0)$')
ax0.plot(z_vals, Hz_vals_lscdm_y_dag_2, color='#Ff0000', ls='-',
        lw=3.0, label=r'$y_{\dagger}=0.70~(\rho_{\Lambda_{\rm s}}(a_{\rm ta}) > 0)$')
ax0.plot(z_vals, Hz_vals_lscdm_y_dag_3, color='#800000', ls='-',
        lw=3.0, label=r'$y_{\dagger}=0.80~(\rho_{\Lambda_{\rm s}}(a_{\rm ta}) > 0)$')

ax0.plot(z_vals, Hz_vals_lscdm_y_dag_4, color='#000080', ls='-',
        lw=3.0, label=r'$y_{\dagger}=1.01~(\rho_{\Lambda_{\rm s}}(a_{\rm ta}) < 0)$')
ax0.plot(z_vals, Hz_vals_lscdm_y_dag_5, color='#0000ff', ls='-',
        lw=3.0, label=r'$y_{\dagger}=1.05~(\rho_{\Lambda_{\rm s}}(a_{\rm ta}) < 0)$')
ax0.plot(z_vals, Hz_vals_lscdm_y_dag_6, color='#00ffff', ls='-',
        lw=3.0, label=r'$y_{\dagger}=1.08~(\rho_{\Lambda_{\rm s}}(a_{\rm ta}) < 0)$')

ax0.axvline(x=2.0, color='green', ls='-', lw=3.0,
        alpha=0.7, label=r'$z_{\rm ta}=2.0$')

# Setting labels
ax0.set_xlabel(r'$z$')
ax0.set_ylabel(r'$H(z) / (1+z)$')

# Setting limits
ax0.set_xlim(0, 4.25)
ax0.set_ylim(58, 88)

# Tick options
ax0.set_xticks([0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0,
                2.25, 2.50, 2.75, 3.0, 3.25, 3.50, 3.75, 4.0, 4.25])

ax0.set_yticks([58, 60, 62, 64, 66, 68, 70, 72,
                74, 76, 78, 80, 82, 84, 86, 88])

ax0.xaxis.set_ticks_position('both')
ax0.yaxis.set_ticks_position('both')

ax0.xaxis.set_tick_params(which='major', width=1.5, size=13.0, direction='in')
ax0.xaxis.set_tick_params(which='minor', width=1.0, size=6.50, direction='in')
ax0.yaxis.set_tick_params(which='major', width=1.5, size=13.0, direction='in')
ax0.yaxis.set_tick_params(which='minor', width=1.0, size=6.50, direction='in')
ax0.minorticks_on()

# Adding text
plt.text(1.7, 83.0, r'$y_{\dagger} > 1$', fontsize=20, color='black')
plt.text(2.1, 83.0, r'$y_{\dagger} < 1$', fontsize=20, color='black')

# Other settings
ax0.legend()
plt.tight_layout()

# Saving the figure
plt.savefig(r'log\hz.pdf', format='pdf', dpi=400)
plt.show()
