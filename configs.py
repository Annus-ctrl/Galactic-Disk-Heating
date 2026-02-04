import numpy as np

# Constants
G_SI = 6.67430e-11
Msun = 1.98847e30
kpc  = 3.085677581491367e19
sec_per_Myr = 1e6 * 365.25 * 24 * 3600

# Galaxy (Miyamoto-Nagai disk)
M_disk = 6e10 * Msun
a_disk = 2.5 * kpc
b_disk = 0.125 * kpc

R_sun = 8*kpc
z_sun = 0

# Simulation
N_stars = 500
dt = 1e12
n_steps = 20000

# Perturber
Mp = 1e10 * Msun
eps = 0.05 * kpc
case_choice = 'inclined'
impact_radius = 3 * kpc
inclination_deg = 30.0
z_start = 50*kpc
v_mag = 200e3
phi_imp = 0.0

