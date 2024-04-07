import numpy as np
import scipy as sp


# ==================== PARAMETERS ====================
max_iter = int(1e6)   # maximum iteration


# ==================== SCALE FACTOR AT THE VIRIALIZATION ====================
# --------------------- LCDM
def cal_a_vir_LCDM(a_ta, omega):

    def integrand_1(y):
        return np.sqrt(y / (a_ta**(-3) + omega*y**3))

    def integrand_2(y):
        return np.sqrt(y / (a_ta**(-3) + omega*y**3))

    def equation(y_vir):
        result_1, _ = sp.integrate.quad(integrand_1, 0, y_vir)
        result_2, _ = sp.integrate.quad(integrand_2, 0, 1)
        return 2*result_2 - result_1

    y_vir = sp.optimize.ridder(equation, 1, 100, maxiter=max_iter)
    a_vir = y_vir * a_ta
    return a_vir

# --------------------- LsCDM (Pre-Turnaround)
def cal_pre_a_vir_LsCDM(a_ta, omega, y_dag):

    def integrand_1(y):
        return np.sqrt(y / (a_ta**(-3) + omega*y**3))

    def integrand_2(y):
        return np.sqrt(y / (a_ta**(-3) - omega*y**3))

    def integrand_3(y):
        return np.sqrt(y / (a_ta**(-3) + omega*y**3))

    def equation(y_vir):
        result_1, _ = sp.integrate.quad(integrand_1, 1, y_vir)
        result_2, _ = sp.integrate.quad(integrand_2, 0, y_dag)
        result_3, _ = sp.integrate.quad(integrand_3, y_dag, 1)
        return result_1 - result_2 - result_3

    y_vir = sp.optimize.ridder(equation, 1, 100, maxiter=max_iter)
    a_vir = y_vir * a_ta
    return a_vir

# --------------------- LsCDM (Post-Turnaround)
def cal_post_a_vir_LsCDM(a_ta, omega, y_dag):

    def integrand_1(y):
        return np.sqrt(y / (a_ta**(-3) - omega*y**3))

    def integrand_2(y):
        return np.sqrt(y / (a_ta**(-3) + omega*y**3))

    def integrand_3(y):
        return np.sqrt(y / (a_ta**(-3) - omega*y**3))

    def equation(y_vir):
        result_1, _ = sp.integrate.quad(integrand_1, 1, y_dag)
        result_2, _ = sp.integrate.quad(integrand_2, y_dag, y_vir)
        result_3, _ = sp.integrate.quad(integrand_3, 0, 1)
        return result_1 + result_2 - result_3

    y_vir = sp.optimize.ridder(equation, 1, 100, maxiter=max_iter)
    a_vir = y_vir * a_ta
    return a_vir
