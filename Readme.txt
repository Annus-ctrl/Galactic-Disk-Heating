# Galactic Disk Simulation

Simulates a thin galactic disk interacting with a perturber using a Miyamoto-Nagai potential. Tracks stellar positions, velocities, energies, and produces visualizations.

---

## Project Structure

- `configs.py`           : Constants, disk and perturber parameters, N_stars, dt, etc. 
- `disk_init.py`         : Initializes disk stars (positions, velocities, angular momentum) 
- `forces.py`            : Miyamoto-Nagai forces + perturber forces 
- `integrator.py`        : Leapfrog integration in cylindrical coordinates with perturber 
- `analysis.py`          : Functions to generate all plots (velocity dispersion, energy, snapshots) 
- `run_simulation.py`    : Main simulation script 
- `simulation_results.npz`: Optional saved simulation data 
---

## Requirements

- Python 3.10+ 
- numpy, matplotlib 



---

## Running the Simulation

1. Navigate to the project folder:


2. Run the simulation:


This will:
- Initialize `N_stars` in a thin disk 
- Define a perturber with mass, position, velocity 
- Integrate the disk using leapfrog 
- Compute energies, velocity dispersions, bound/unbound stars 
- Save results to `simulation_results.npz` 
- Automatically generate all analysis plots 
---

## Key Outputs / Plots

1. **Energy vs Angular Momentum** – ΔE/E0 vs Lz per star 
2. **3D Snapshots** – Stars and moving perturber at initial, crossing, final times 
3. **X-Z Snapshots** – Vertical disk structure 
4. **Global Velocity Dispersions** – σR and σz over time
5. **Surface Density Snapshots** – XY plane at key times 
6. **Z-vZ Phase Space** – Vertical motion of stars 
7. **Energy Evolution** – Bound, unbound, total energy 
8. **σz: All vs Bound** – Comparison 
9. **Number of Unbound Stars** – Over time 
10. **Radial-bin Dispersions** – σR in radial bins 
11. **Energy Conservation** – Max, min, median ΔE/E0 over time 
12. **Final Disk Snapshot** – Bound vs unbound stars 

---

## Configuration

All parameters in `configs.py`:

- Disk: `a_disk`, `b_disk`, `M_disk` 
- Stars: `N_stars` 
- Time step: `dt`
- Perturber: `Mp`, `eps`, `impact_radius`, `z_start`, `v_mag` 
- Constants: `G_SI`, `kpc`, `sec_per_Myr` 

Modify values to change disk model, perturber, or simulation duration.

---

## Notes

- Leapfrog integration ensures stable energy evolution 
- Perturber trajectory is computed analytically from initial position & velocity 
- Snapshots include **initial**, **perturber crossing**, **final** times 
- Positions in kpc, velocities in km/s for plotting 
- Energy vs Lz plots are per star; energy conservation plots are disk-wide 

---

## Example Usage

```python
import numpy as np
import analysis as anl

data = np.load('simulation_results.npz')
R_hist, phi_hist, z_hist = data['R_hist'], data['phi_hist'], data['z_hist']
x_hist = R_hist * np.cos(phi_hist)
y_hist = R_hist * np.sin(phi_hist)
pert_pos = ... # computed from perturber initial r0 and v0
snapshot_indices = [0, 2000, 9999]
snapshot_labels = ['Initial', 'Crossing', 'Final']

anl.plot_3d_snapshots(x_hist, y_hist, z_hist, pert_pos, snapshot_indices, snapshot_labels)

