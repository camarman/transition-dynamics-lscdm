import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np


# ==================== NUMERICAL ANALYSIS ====================
z_ta_values = np.linspace(1.70, 4.70, 15000, endpoint=True)
y_dag_values = np.linspace(0.48, 1.08, 15000, endpoint=True)

X, Y = np.meshgrid(z_ta_values, y_dag_values)

def z_dag_values(z_ta, y_dag):
	return ((1 + z_ta)/y_dag) - 1

variable_grid_data = list((x, y) for x in z_ta_values for y in y_dag_values)
data_points = np.array([z_dag_values(x, y) for (x, y) in variable_grid_data])
new_points = np.reshape(data_points, (len(z_ta_values), len(y_dag_values)))

Z = np.transpose(new_points)


# ==================== PLOT ====================
# LaTeX rendering text fonts
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Adjusting size of the figure
params = {
            'legend.fontsize': '22.5',
            'axes.labelsize': '40',
            'figure.figsize': (15, 10),
            'xtick.labelsize': '30',
            'ytick.labelsize': '30'
        }

pylab.rcParams.update(params)

fig, ax0 = plt.subplots()

# --------------------- Contour plot settings
# Set custom colormap range
vmin, vmax = 1.5, 11.5

# Set custom tick levels
levels = np.linspace(vmin, vmax, 10)
contour = plt.contourf(X, Y, Z, levels=levels, cmap='plasma', vmin=vmin, vmax=vmax)
plt.axhline(y=1, color='white', ls='-', lw=3.0, label=r'$y_{\dagger}=1$')

# Colorbar options
cbar = plt.colorbar(contour)
cbar.set_ticks(np.linspace(vmin, vmax, 10))

# Add labels
plt.xlabel(r'$z_{\rm ta}$')
plt.ylabel(r'$y_{\dagger}$')
cbar.set_label(r'$z_{\dagger}$')

# --------------------- Tick settings
ax0.set_xticks([1.7, 2.0, 2.3, 2.6, 2.9, 3.2, 3.5, 3.8, 4.1, 4.4, 4.7])
ax0.set_yticks([0.48, 0.53, 0.58, 0.63, 0.68, 0.73, 0.78, 0.83, 0.88, 0.93, 0.98, 1.03, 1.08])

ax0.xaxis.set_ticks_position('both')
ax0.yaxis.set_ticks_position('both')

ax0.xaxis.set_tick_params(which='major', width=1.5, size=13.0, color='white', direction='in')
ax0.xaxis.set_tick_params(which='minor', width=1.0, size=6.50, color='white', direction='in')
ax0.yaxis.set_tick_params(which='major', width=1.5, size=13.0, color='white', direction='in')
ax0.yaxis.set_tick_params(which='minor', width=1.0, size=6.50, color='white', direction='in')
ax0.minorticks_on()

plt.text(1.850, 1.018, r'$\rho_{\Lambda_{\rm s}}(a_{\rm ta}) < 0$', fontsize=20, color='white')
plt.text(1.850, 0.975, r'$\rho_{\Lambda_{\rm s}}(a_{\rm ta}) > 0$', fontsize=20, color='white')

# Other settings
ax0.legend()
plt.tight_layout()
ax0.set_rasterized(True)

# Saving the figure
plt.savefig(r'log\zTurnaroundContour.pdf', format='pdf', dpi=400)
plt.show()
