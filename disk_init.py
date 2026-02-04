import numpy as np
from forces import mn_forces
from configs import kpc

def sigma_R_of_R(R, R0=8*kpc, sigmaR0=35e3, R_sigma=7*kpc):
    return sigmaR0 * np.exp(-(R - R0)/R_sigma)

def sigma_z_of_R(R, R0=8*kpc, sigmaz0=25e3, R_sigma=7*kpc):
    return sigmaz0 * np.exp(-(R - R0)/R_sigma)

def initialize_thin_disk_realistic(N, a, b, M, R_d=2.5*kpc, R_min=2*kpc, R_max=15*kpc, h=0.3*kpc):
    U = np.random.uniform(0,1,size=N)
    norm = np.exp(-R_min/R_d)-np.exp(-R_max/R_d)
    R = -R_d * np.log(np.exp(-R_min/R_d) - U*norm)
    phi = np.random.uniform(0, 2*np.pi, size=N)
    z = np.random.normal(0,h,size=N)
    _, FR, _ = mn_forces(R, np.zeros_like(R), a, b, M)
    Vc = np.sqrt(-FR*R)
    dR = 1e-4*R
    _, FRp, _ = mn_forces(R+dR, np.zeros_like(R), a, b, M)
    _, FRm, _ = mn_forces(R-dR, np.zeros_like(R), a, b, M)
    Omega = Vc/R
    Om2_p, Om2_m = -FRp/(R+dR), -FRm/(R-dR)
    kappa = np.sqrt(R*(Om2_p-Om2_m)/(2*dR) + 4*Omega**2)
    sigma_R, sigma_z = sigma_R_of_R(R), sigma_z_of_R(R)
    sigma_phi = sigma_R*kappa/(2*Omega)
    vR = np.random.normal(0, sigma_R)
    vz = np.random.normal(0, sigma_z)
    vphi = Vc + np.random.normal(0, sigma_phi)
    Lz = R*vphi
    return R, z, vR, vz, vphi, phi, Lz

