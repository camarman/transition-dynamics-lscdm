import numpy as np
import scipy as sp


# ==================== PARAMETERS ====================
# --------------------- Limits
min_limit = int(1e-6)
max_iter = int(1e6)


# ==================== CALCULATING THE ETA PARAMETER ====================
# --------------------- LCDM
def solve_eta_lcdm(e_Lambda):
    def equation(eta):
        return 4*e_Lambda*eta**3 - 2*(1 + e_Lambda)*eta + 1

    root = sp.optimize.ridder(equation, min_limit, 0.999999, maxiter=max_iter)
    return root

# --------------------- LsCDM (Pre-Turnaround)
def solve_eta_pre_turn_lscdm(e_Lambda_s):
    def equation(eta):
        return 4*e_Lambda_s*eta**3 - 2*(1 + e_Lambda_s)*eta + 1

    root = sp.optimize.ridder(equation, min_limit, 0.999999, maxiter=max_iter)
    return root

# --------------------- LsCDM (Post-Turnaround)
def solve_eta_post_turn_lscdm(e_Lambda_s, u_dag, delta0):
    def equation(eta):
        return 2*eta*(delta0 * (-1 + u_dag) + u_dag + u_dag*e_Lambda_s*(-1 + 2*eta**2 + delta0*(-1 + u_dag**2))) - u_dag
    root = sp.optimize.ridder(equation, min_limit, 0.999999, maxiter=max_iter)
    return root
