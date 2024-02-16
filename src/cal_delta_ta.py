import numpy as np
import scipy as sp


# ==================== PARAMETERS ====================
# --------------------- Limits
min_limit = int(1e-6)
max_limit = int(1e2)
max_iter = int(1e6)


# ==================== DENSITY CONTRAST AT THE TURNAROUND ====================
# --------------------- LCDM
def cal_delta_ta_LCDM(a_ta, omega):
    def integrand_1(u, delta_ta):
        return np.sqrt(u / (a_ta**(-3)*(1 + delta_ta)*(1 - u) - omega*u*(1 - u**2)))

    def integrand_2(y):
        return np.sqrt(y / (a_ta**(-3) + omega*y**3))

    def equation(delta_ta):
        result_1, _ = sp.integrate.quad(integrand_1, 0, 1, args=(delta_ta))
        result_2, _ = sp.integrate.quad(integrand_2, 0, 1)
        return result_1 - result_2

    root = sp.optimize.ridder(equation, min_limit, max_limit, maxiter=max_iter)
    return root

# --------------------- LsCDM (Pre-Turnaround)
def cal_pre_delta_ta_LsCDM(a_ta, omega, y_dag):
    """
    Calculating delta_ta for a_s < a_ta < a_vir (pre-turnaround).
    """
    # Define the integrand for the right-hand side
    def integrand_1(y):
        return np.sqrt(y / (a_ta**(-3) - omega*y**3))

    def integrand_2(u, delta_ta):
        return np.sqrt(u / (a_ta**(-3)*(1 + delta_ta)*(1 - u) - omega*u*(1 + u**2)))

    def integrand_3(u, delta_ta):
        return np.sqrt(u / (a_ta**(-3)*(1 + delta_ta)*(1 - u) - omega*u*(1 - u**2)))

    def integrand_4(y):
        return np.sqrt(y / (a_ta**(-3) + omega*y**3))

    def equation(x):
        result_1, _ = sp.integrate.quad(integrand_1, 0, y_dag)
        result_2, _ = sp.integrate.quad(integrand_2, 0, x[0], args=x[1])
        result_3, _ = sp.integrate.quad(integrand_3, x[0], 1, args=x[1])
        result_4, _ = sp.integrate.quad(integrand_4, y_dag, 1)
        return [result_2 - result_1, result_4 - result_3]

    if y_dag == 0.80:
        initial_guess = [0.90, 6]
    elif y_dag == 0.70:
        initial_guess = [0.85, 6]
    elif y_dag == 0.60:
        initial_guess = [0.70, 5]

    root = sp.optimize.root(equation, initial_guess)
    return root.x[1]

# --------------------- LsCDM (Post-Turnaround)
def cal_post_delta_ta_LsCDM(a_ta, omega):
    """
    Calculating delta_ta for a_s < a_ta < a_vir (post-turnaround).
    """
    def integrand_1(u, delta_ta):
        return np.sqrt(u / (a_ta**(-3)*(1 + delta_ta)*(1 - u) + omega*u*(1 - u**2)))

    def integrand_2(y):
        return np.sqrt(y / (a_ta**(-3) - omega*y**3))

    def equation(delta_ta):
        result_1, _ = sp.integrate.quad(integrand_1, 0, 1, args=(delta_ta))
        result_2, _ = sp.integrate.quad(integrand_2, 0, 1)
        return result_1 - result_2

    root = sp.optimize.ridder(equation, min_limit, max_limit, maxiter=max_iter)
    return root


def find_u_dag(a_ta, omega, y_dag, delta_ta):
    def integrand_1(u):
        return np.sqrt(u / (a_ta**(-3)*(1 + delta_ta)*(1 - u) + omega*u*(1 - u**2)))

    def integrand_2(y):
        return np.sqrt(y / (a_ta**(-3) - omega*y**3))

    def equation(u_dag):
        result_1, _ = sp.integrate.quad(integrand_1, u_dag, 1)
        result_2, _ = sp.integrate.quad(integrand_2, 1, y_dag)
        return result_2 - result_1

    root = sp.optimize.ridder(equation, min_limit, 0.999999, maxiter=max_iter)
    return root
