# forces.py
import numpy as np
from configs import G_SI

# Miyamoto-Nagai potential 
def mn_forces(R, z, a, b, M, G=G_SI):
    R, z = np.atleast_1d(R), np.atleast_1d(z)
    B = np.sqrt(z**2 + b**2)
    D = np.sqrt(R**2 + (a + B)**2)
    Phi = -G * M / D
    FR = -G * M * R / D**3
    Fz = -G * M * (a + B) * z / (D**3 * B)
    return Phi, FR, Fz

# Perturber forces
def perturber_force_cartesian(x, y, z, t, pert):
    rp = perturber_pos(t, pert)  # <-- use perturber_pos here
    rx, ry, rz = x - rp[0], y - rp[1], z - rp[2]
    r2 = rx**2 + ry**2 + rz**2
    denom = (r2 + pert['eps']**2)**1.5
    denom = np.where(denom==0, np.finfo(float).eps, denom)
    fac = -G_SI * pert['Mp'] / denom
    return fac*rx, fac*ry, fac*rz

def cart_to_cyl_FR_Fz(x, y, Fx, Fy, Fz, R):
    with np.errstate(divide='ignore', invalid='ignore'):
        FR = (x*Fx + y*Fy)/R
        FR = np.where(R==0, 0.0, FR)
    return FR, Fz

# Perturber position function
def perturber_pos(t, pert):
    """
    Returns 3D perturber position at time t (scalar or array)
    """
    r0 = np.array(pert['r0'], dtype=float)
    v0 = np.array(pert['v0'], dtype=float)
    
    t = np.atleast_1d(t)
    pos = r0[np.newaxis, :] + np.outer(t, v0)
    
    if pos.shape[0] == 1:
        return pos[0]  # return 1D array if t scalar
    return pos       # return 2D array (n_times, 3) if t array

