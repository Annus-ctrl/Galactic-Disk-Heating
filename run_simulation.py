import numpy as np
from configs import *
from disk_init import initialize_thin_disk_realistic
from integrator import leapfrog_cylindrical_step_with_pert
from forces import mn_forces
import analysis as anl


# Initialize disk

R, z, vR, vz, vphi, phi, Lz = initialize_thin_disk_realistic(N_stars, a_disk, b_disk, M_disk)


# Define perturber

pert = {
    'Mp': Mp,
    'eps': eps,
    'r0': np.array([impact_radius, 0.0, z_start], dtype=float),
    'v0': np.array([0.0, 0.0, -v_mag], dtype=float)
}


# Allocate history arrays

R_hist = np.zeros((n_steps, N_stars))
z_hist = np.zeros_like(R_hist)
phi_hist = np.zeros_like(R_hist)
vR_hist = np.zeros_like(R_hist)
vz_hist = np.zeros_like(R_hist)
vphi_hist = np.zeros_like(R_hist)
E_hist = np.zeros_like(R_hist)


# Integration loop

t = 0.0
for i in range(n_steps):
    # Host forces only (needed for energy)
    Phi, FR, Fz = mn_forces(R, z, a_disk, b_disk, M_disk)
    
    vphi = Lz / R
    E = 0.5*(vR**2 + vz**2 + vphi**2) + Phi
    
    # Store
    R_hist[i,:] = R
    z_hist[i,:] = z
    phi_hist[i,:] = phi
    vR_hist[i,:] = vR
    vz_hist[i,:] = vz
    vphi_hist[i,:] = vphi
    E_hist[i,:] = E
    
    # Advance one step with perturber
    R, z, vR, vz, phi = leapfrog_cylindrical_step_with_pert(
        R, z, vR, vz, phi, Lz, dt,
        a_disk, b_disk, M_disk,
        t, pert=pert
    )
    
    t += dt


# Save simulation results

np.savez('simulation_results.npz',
         R_hist=R_hist, z_hist=z_hist, phi_hist=phi_hist,
         vR_hist=vR_hist, vz_hist=vz_hist, vphi_hist=vphi_hist,
         E_hist=E_hist
)


# Compute perturber trajectory & t_cross

time_array = np.arange(n_steps) * dt  # seconds
pert_pos = pert['r0'][None,:] + time_array[:,None] * pert['v0'][None,:]  # shape (n_steps,3)
t_cross_Myr = anl.compute_t_cross(pert_pos, time_array)  # returns in Myr
cross_idx = np.argmin(np.abs(pert_pos[:,2]))  # closest approach
snapshot_indices = [0, cross_idx, n_steps-1]
snapshot_labels = ['Initial', 'Perturber crossing', 'Final']



# Convert to Cartesian coordinates

x_hist = R_hist * np.cos(phi_hist)
y_hist = R_hist * np.sin(phi_hist)


# Perturber trajectory & snapshots

time_array = np.arange(n_steps) * dt  # seconds
pert_pos = pert['r0'][None,:] + time_array[:,None] * pert['v0'][None,:]  # shape (n_steps,3)

# Perturber crossing time (Myr)
t_cross_Myr = anl.compute_t_cross(pert_pos, time_array)
cross_idx = np.argmin(np.abs(pert_pos[:,2]))  # closest approach

# Snapshot indices & labels
snapshot_indices = [0, cross_idx, n_steps-1]
snapshot_labels = ['Initial', 'Perturber crossing', 'Final']


# Bound / unbound stars

bound_mask_hist = E_hist < 0
unbound_mask_hist = ~bound_mask_hist

E_bound = np.sum(E_hist * bound_mask_hist, axis=1)
E_unbound = np.sum(E_hist * unbound_mask_hist, axis=1)
E_tot_MN = np.sum(E_hist, axis=1)


# Velocity dispersions

sigma_R_all = np.std(vR_hist, axis=1)
sigma_z_all = np.std(vz_hist, axis=1)

sigma_R_bound = np.array([
    np.std(vR_hist[i,bound_mask_hist[i]]) if np.sum(bound_mask_hist[i])>0 else np.nan
    for i in range(n_steps)
])
sigma_z_bound = np.array([
    np.std(vz_hist[i,bound_mask_hist[i]]) if np.sum(bound_mask_hist[i])>0 else np.nan
    for i in range(n_steps)
])


# Radial-bin velocity dispersions

rbins_kpc = np.linspace(2,15,7)
rbins = rbins_kpc * kpc
bin_idx = np.digitize(R_hist[0,:], rbins) - 1
nbins = len(rbins)-1
sigmaR_bins_t = np.zeros((n_steps, nbins))

for ib in range(nbins):
    mask = (bin_idx == ib)
    if np.sum(mask) > 0:
        sigmaR_bins_t[:,ib] = np.std(vR_hist[:,mask], axis=1)
    else:
        sigmaR_bins_t[:,ib] = np.nan


# Relative energy changes

rel_dE = (E_hist - E_hist[0,:]) / np.abs(E_hist[0,:])

# For energy vs time (use axis=1 → across stars)
rel_dE_max_time = np.max(np.abs(rel_dE), axis=1)
rel_dE_min_time = np.min(np.abs(rel_dE), axis=1)
rel_dE_median_time = np.median(np.abs(rel_dE), axis=1)

# For energy vs Lz (use axis=0 → per star)
rel_dE_max_star = np.max(np.abs(rel_dE), axis=0)
rel_dE_min_star = np.min(np.abs(rel_dE), axis=0)
rel_dE_median_star = np.median(np.abs(rel_dE), axis=0)


# PLOTTING

# Energy vs Lz (per star)
anl.plot_energy_vs_Lz(Lz, rel_dE_max_star, rel_dE_min_star, rel_dE_median_star, kpc=kpc)

# 3D snapshots with moving perturber
anl.plot_3d_snapshots(x_hist, y_hist, z_hist, pert_pos, snapshot_indices, snapshot_labels, kpc=kpc)

# X-Z snapshots
anl.plot_xz_snapshots(x_hist, z_hist, pert_pos, n_steps)

# Global velocity dispersions
anl.plot_global_velocity_dispersions(time_array/sec_per_Myr, vR_hist, vz_hist, t_cross_Myr)

# Surface density snapshots (xy plane)
anl.plot_surface_density_snapshots(R_hist, phi_hist, kpc, snapshot_indices, snapshot_labels)

# z vs vz phase space
anl.plot_z_vz_phase_space(z_hist, vz_hist, time_array/sec_per_Myr, kpc, snapshot_indices)

# Energy evolution
anl.plot_energy_evolution(time_array/sec_per_Myr, E_bound, E_unbound, E_tot_MN)

# σz: all vs bound
anl.plot_sigma_all_vs_bound(time_array/sec_per_Myr, sigma_z_all, sigma_z_bound, t_cross_Myr, 'σz [km/s]')

# Number of unbound stars
anl.plot_unbound_count(time_array/sec_per_Myr, unbound_mask_hist, t_cross_Myr)

# Radial-bin dispersions
anl.plot_radial_bin_dispersion(time_array/sec_per_Myr, sigmaR_bins_t, rbins_kpc, 'σR [km/s]', t_cross_Myr)

# Energy conservation over time
anl.plot_energy_conservation(time_array/sec_per_Myr, rel_dE_max_time, rel_dE_min_time, rel_dE_median_time, t_cross_Myr)

# Final disk snapshot (bound vs unbound)
anl.plot_final_disk_snapshot(x_hist, y_hist, bound_mask_hist, unbound_mask_hist, kpc)

print("✅ Simulation complete. All plots generated successfully.")


