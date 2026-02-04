import numpy as np
from configs import *
from forces import mn_forces, perturber_force_cartesian, cart_to_cyl_FR_Fz

def total_forces_with_perturber(R, z, phi, t, a, b, M_host, pert=None):
    Phi_host, FR_host, Fz_host = mn_forces(R, z, a, b, M_host)
    if pert is None:
        return Phi_host, FR_host, Fz_host
    x, y = R*np.cos(phi), R*np.sin(phi)
    Fx_p, Fy_p, Fz_p = perturber_force_cartesian(x, y, z, t, pert)
    FR_p, Fz_p = cart_to_cyl_FR_Fz(x, y, Fx_p, Fy_p, Fz_p, R)
    Phi_p = -G_SI*pert['Mp']/np.sqrt((x-pert['r0'][0])**2+(y-pert['r0'][1])**2+(z-pert['r0'][2])**2+pert['eps']**2)
    return Phi_host + Phi_p, FR_host + FR_p, Fz_host + Fz_p

def leapfrog_cylindrical_step_with_pert(R, z, vR, vz, phi, Lz, dt, a, b, M_host, t, pert=None):
    Phi, FR, Fz = total_forces_with_perturber(R, z, phi, t, a, b, M_host, pert)
    aR, az = Lz**2/R**3 + FR, Fz
    vR_half, vz_half = vR+0.5*dt*aR, vz+0.5*dt*az
    R_new, z_new = R+dt*vR_half, z+dt*vz_half
    R_new = np.where(R_new<=0, 1e-12, R_new)
    phi_new = phi + dt*(Lz/R_new**2)
    Phi_new, FR_new, Fz_new = total_forces_with_perturber(R_new, z_new, phi_new, t+dt, a, b, M_host, pert)
    aR_new, az_new = Lz**2/R_new**3 + FR_new, Fz_new
    vR_new, vz_new = vR_half + 0.5*dt*aR_new, vz_half + 0.5*dt*az_new
    return R_new, z_new, vR_new, vz_new, phi_new

