# analysis.py
import numpy as np
import matplotlib.pyplot as plt
from configs import kpc, sec_per_Myr


# Helper conversion

def km_s(x):
    return x / 1e3


# Compute perturber crossing time from perturber trajectory

def compute_t_cross(pert_pos, t_array):
    pert_z = pert_pos[:, 2]
    sign_z = np.sign(pert_z)
    cross_inds = np.where(sign_z[:-1] * sign_z[1:] <= 0)[0]

    if len(cross_inds) > 0:
        idx = cross_inds[0]
        z1, z2 = pert_z[idx], pert_z[idx+1]
        t1, t2 = t_array[idx], t_array[idx+1]
        if (z2 - z1) != 0:
            t_cross = t1 - z1*(t2 - t1)/(z2 - z1)
        else:
            t_cross = t1
    else:
        idx = np.argmin(np.abs(pert_z))
        t_cross = t_array[idx]
    return t_cross / sec_per_Myr  # in Myr


# Global velocity dispersions

def plot_global_velocity_dispersions(time_Myr, vR_hist, vz_hist, t_cross_Myr=None, title=None):
    sigma_R = np.sqrt(np.mean(vR_hist**2, axis=1))
    sigma_z = np.sqrt(np.mean(vz_hist**2, axis=1))

    plt.figure()
    plt.plot(time_Myr, km_s(sigma_R), label=r'$\sigma_R$')
    plt.plot(time_Myr, km_s(sigma_z), label=r'$\sigma_z$')

    if t_cross_Myr is not None:
        plt.axvline(t_cross_Myr, color='k', linestyle='--', label='Perturber crossing')

    plt.xlabel('Time [Myr]')
    plt.ylabel('Velocity dispersion [km/s]')
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# Surface density snapshots (xy plane)

def plot_surface_density_snapshots(R_hist, phi_hist, kpc, indices, labels):
    plt.figure(figsize=(5 * len(indices), 4))
    for i, idx in enumerate(indices):
        plt.subplot(1, len(indices), i+1)
        x = R_hist[idx] * np.cos(phi_hist[idx]) / kpc
        y = R_hist[idx] * np.sin(phi_hist[idx]) / kpc
        plt.scatter(x, y, s=2)
        plt.gca().set_aspect('equal', 'box')
        plt.xlabel('x [kpc]')
        plt.ylabel('y [kpc]')
        plt.title(labels[i])
    plt.tight_layout()
    plt.show()


# z vs vz phase space

def plot_z_vz_phase_space(z_hist, vz_hist, time_Myr, kpc, indices):
    plt.figure(figsize=(9, 3))
    for i, idx in enumerate(indices):
        plt.subplot(1, len(indices), i+1)
        plt.scatter(z_hist[idx]/kpc, vz_hist[idx]/1e3, s=2)
        plt.xlabel('z [kpc]')
        plt.ylabel('v_z [km/s]')
        plt.title(f't = {time_Myr[idx]:.1f} Myr')
    plt.tight_layout()
    plt.show()


# Energy evolution

def plot_energy_evolution(time_Myr, E_bound, E_unbound, E_MN, scale=1e13):
    plt.figure()
    plt.plot(time_Myr, E_bound/scale, label='Bound')
    plt.plot(time_Myr, E_unbound/scale, label='Unbound')
    plt.plot(time_Myr, E_MN/scale, '--', label='MN only')
    plt.xlabel('Time [Myr]')
    plt.ylabel(r'Energy [$10^{13}$ J]')
    plt.legend()
    plt.grid(True)
    plt.show()


# Velocity dispersions: all vs bound

def plot_sigma_all_vs_bound(time_Myr, sigma_all, sigma_bound, t_cross_Myr, ylabel='σ [km/s]', title=None):
    plt.figure()
    plt.plot(time_Myr, km_s(sigma_all), label='All stars')
    plt.plot(time_Myr, km_s(sigma_bound), label='Bound stars')
    plt.axvline(t_cross_Myr, color='k', linestyle='--', label='Perturber crossing')
    plt.xlabel('Time [Myr]')
    plt.ylabel(ylabel)
    if title: plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# Number of unbound stars

def plot_unbound_count(time_Myr, unbound_mask_hist, t_cross_Myr):
    num_unbound = np.sum(unbound_mask_hist, axis=1)
    plt.figure()
    plt.plot(time_Myr, num_unbound, label='Unbound stars')
    plt.axvline(t_cross_Myr, color='r', linestyle='--', label='Perturber crossing')
    plt.xlabel('Time [Myr]')
    plt.ylabel('Number of unbound stars')
    plt.legend()
    plt.grid(True)
    plt.show()


# Radial-bin dispersions

def plot_radial_bin_dispersion(time_Myr, sigma_bins, rbins_kpc, ylabel, t_cross_Myr):
    plt.figure(figsize=(9,5))
    for i in range(sigma_bins.shape[1]):
        label = f'{rbins_kpc[i]:.1f}-{rbins_kpc[i+1]:.1f} kpc'
        plt.plot(time_Myr, km_s(sigma_bins[:,i]), label=label)
    plt.axvline(t_cross_Myr, color='k', linestyle='--')
    plt.xlabel('Time [Myr]')
    plt.ylabel(ylabel)
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Energy conservation vs time

def plot_energy_conservation(time_Myr, rel_dE_max, rel_dE_min, rel_dE_median, t_cross_Myr):
    plt.figure(figsize=(8,5))
    plt.plot(time_Myr, rel_dE_max, color='r', alpha=0.7, label='Max |ΔE/E0|')
    plt.plot(time_Myr, rel_dE_min, color='b', alpha=0.7, label='Min |ΔE/E0|')
    plt.plot(time_Myr, rel_dE_median, color='k', linestyle='--', label='Median ΔE/E0')
    plt.axvline(t_cross_Myr, color='k', linestyle='--', label=f'Perturber crossing at {t_cross_Myr:.2f} Myr')
    plt.xlabel("Time [Myr]")
    plt.ylabel("Relative Energy Change ΔE / |E0|")
    plt.title("Energy conservation")
    plt.grid(True)
    plt.legend()
    plt.show()


#  Final disk snapshot (bound/unbound)

def plot_final_disk_snapshot(x_hist, y_hist, bound_mask_hist, unbound_mask_hist, kpc):
    plt.figure(figsize=(6,6))
    plt.scatter(x_hist[-1,bound_mask_hist[-1,:]]/kpc, y_hist[-1,bound_mask_hist[-1,:]]/kpc, s=2, label='Bound')
    plt.scatter(x_hist[-1,unbound_mask_hist[-1,:]]/kpc, y_hist[-1,unbound_mask_hist[-1,:]]/kpc, s=5, color='r', label='Unbound')
    plt.xlabel('x [kpc]')
    plt.ylabel('y [kpc]')
    plt.title('Final disk snapshot')
    plt.legend()
    plt.gca().set_aspect('equal', 'box')
    plt.grid(True)
    plt.show()

def plot_xz_snapshots(x_hist, z_hist, pert_pos, n_steps, kpc=kpc):
    # Convert to kpc
    x = x_hist / kpc
    z = z_hist / kpc
    pert_x = pert_pos[:,0] / kpc
    pert_z = pert_pos[:,2] / kpc

    # Find perturber crossing index (closest to z=0)
    cross_idx = np.argmin(np.abs(pert_z))

    # Snapshot indices
    snapshot_indices = [0, cross_idx, n_steps-1]
    snapshot_labels = ['Initial', 'Perturber crossing', 'Final']

    # Plot
    plt.figure(figsize=(15,4))
    for i, idx in enumerate(snapshot_indices):
        plt.subplot(1,3,i+1)
        plt.scatter(x[idx,:], z[idx,:], s=2, color='blue', label='Stars')
        plt.scatter(pert_x[idx], pert_z[idx], color='red', marker='*', s=100, label='Perturber')
        plt.xlabel('X [kpc]')
        if i == 0:
            plt.ylabel('Z [kpc]')
        plt.title(snapshot_labels[i])
        plt.axis('equal')
        plt.grid(True)
        if i == 0:
            plt.legend()
    plt.tight_layout()
    plt.show()
    
# 3D Snapshot (Perturber disk interaction)

def plot_3d_snapshots(x_hist, y_hist, z_hist, pert_pos, indices, labels, kpc=kpc):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(18,5))
    for i, idx in enumerate(indices):
        ax = fig.add_subplot(1, len(indices), i+1, projection='3d')
        ax.scatter(x_hist[idx,:]/kpc, y_hist[idx,:]/kpc, z_hist[idx,:]/kpc, s=2, color='blue', label='Stars')
        # moving perturber
        ax.scatter(pert_pos[idx,0]/kpc, pert_pos[idx,1]/kpc, pert_pos[idx,2]/kpc,
                   color='red', marker='*', s=100, label='Perturber')
        ax.set_xlabel('X [kpc]')
        ax.set_ylabel('Y [kpc]')
        ax.set_zlabel('Z [kpc]')
        ax.set_title(labels[i])
        ax.view_init(elev=10, azim=120)
        ax.set_box_aspect([1,1,0.5])
        if i==0:
            ax.legend()
    plt.tight_layout()
    plt.show()

# Energy vs Angular Momentum

def plot_energy_vs_Lz(Lz, rel_dE_max, rel_dE_min, rel_dE_median, kpc=1.0):
    import matplotlib.pyplot as plt
    import numpy as np

    # Ensure all arrays are 1D and same length
    Lz = np.ravel(Lz)
    rel_dE_max = np.ravel(rel_dE_max)
    rel_dE_min = np.ravel(rel_dE_min)
    rel_dE_median = np.ravel(rel_dE_median)

    if not (len(Lz) == len(rel_dE_max) == len(rel_dE_min) == len(rel_dE_median)):
        raise ValueError("Lz and rel_dE arrays must have the same length")

    plt.figure(figsize=(8,5))
    plt.scatter(Lz/(kpc*1e3), rel_dE_max, s=10, color='r', alpha=0.6, label='Max |ΔE/E0|')
    plt.scatter(Lz/(kpc*1e3), rel_dE_min, s=10, color='b', alpha=0.6, label='Min |ΔE/E0|')
    plt.scatter(Lz/(kpc*1e3), rel_dE_median, s=10, color='k', alpha=0.6, label='Median ΔE/E0')

    plt.xlabel("Angular Momentum Lz [kpc * km/s]")
    plt.ylabel("Relative Energy Change ΔE / |E0|")
    plt.title("Energy Conservation vs Angular Momentum")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

