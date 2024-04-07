import numpy as np
from scipy.integrate import quad


# ==================== PARAMETERS ====================
# Parameters are taken from https://arxiv.org/pdf/1807.06209.pdf
# Table I. Upper Panel Plik Best Fit Values

c = 299792.458   # speed of light [km/s]
N_eff = 3.046    # effective neutrino number

Obh2 = 0.022383                                     # physical baryon density parameter
Omh2 = 0.143140                                     # physical matter density parameter

Oph2 = 2.469 * 10**(-5)                             # physical photon density
Onh2 = 2.469 * 10**(-5)*(7/8)*(4/11)**(4/3)*N_eff   # physical neutrino density
Orh2 = Oph2 + Onh2                                  # physical radiation density

theta_true = 0.01041085                             # acoustic scale
hubble_error = 1e-8                                 # the accepted error while calculating the hubble constant
h0_prior = [0.4, 1]                                 # h0 prior range


# ==================== CALCULATING REDSHIFT TO LSS ====================
# See https://arxiv.org/pdf/astro-ph/9510117.pdf for further information
def z_lss_finder(Obh2, Omh2):
    """Calculating the redshift to the Last Scattering Surface (LSS)"""
    g1 = 0.0783*Obh2**(-0.238)*(1+39.5*Obh2**(0.763))**(-1)
    g2 = 0.56*(1+21.1*Obh2**(1.81))**(-1)
    z_lss = 1048*(1+0.00124*Obh2**(-0.738))*(1+g1*Omh2**g2)
    return z_lss

# Redshift to LSS, for a given Obh2 and Omh2
z_lss = z_lss_finder(Obh2, Omh2)


# ==================== CALCULATING THE COMOVING SOUND HORIZON AT THE LSS ====================
def r_s_finder_LCDM(h0):
    """Calculating the comoving sound horizon at the LSS (r_s)"""
    def integrand(z):
        R = (3*Obh2) / (4*Oph2*(1+z))
        c_s = c / np.sqrt(3*(1+R))
        return c_s / (100 * np.sqrt(Omh2*(1+z)**3 + Orh2*(1+z)**4 + (h0**2-Omh2-Orh2)))
    r_s = quad(integrand, z_lss, np.inf)[0]
    return r_s


# ==================== CALCULATING COMOVING ANGULAR DIAMETER DISTANCE AT THE LSS ====================
def d_A_finder_LCDM(h0):
    """Calculating the comoving angular diameter distance to the LSS (d_A(z_*))"""
    def integrand(z):
        return c / (100 * np.sqrt(Omh2*(1+z)**3 + Orh2*(1+z)**4 + (h0**2-Omh2-Orh2)))
    r_s = quad(integrand, 0, z_lss)[0]
    return r_s


# ==================== FINDING HUBBLE CONSTANT & MATTER DENSITY PARAMETER ====================
def h0_finder_LCDM():
    """Finding the Hubble constant"""
    h0_min, h0_max = h0_prior
    for i in range(100):
        h0_test = (h0_min + h0_max) / 2
        r_s_test = r_s_finder_LCDM(h0_test)
        d_A_test = d_A_finder_LCDM(h0_test)
        theta_test = r_s_test / d_A_test
        if abs(theta_true - theta_test) > hubble_error: # adjusting the error
            if theta_true - theta_test > 0:
                h0_min = h0_test
            else:
                h0_max = h0_test
        else:
            break
    return h0_test


def Om0_finder_LCDM():
    """Finding the matter density parameter"""
    h0 = h0_finder_LCDM()
    return Omh2/h0**2