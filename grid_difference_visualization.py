# Script to Compare Coarse and Fine Grid Solutions with Visualization

import numpy as np
import matplotlib.pyplot as plt

# --- Load Solution Data ---
coarse_file = "pde_solution_Nx10_Ny11.npy"
fine_file = "pde_solution_Nx20_Ny21.npy"

try:
    U_coarse = np.load(coarse_file)
    U_fine = np.load(fine_file)
    print(f"Loaded coarse solution from {coarse_file} (Shape: {U_coarse.shape})")
    print(f"Loaded fine solution from {fine_file} (Shape: {U_fine.shape})")
except FileNotFoundError as e:
    print(f"Error loading solution files: {e}")
    print("Please ensure the .npy files from the refinement study exist.")
    exit()

# --- Grid Parameters (must match the runs that generated the .npy files) ---
Nx_coarse, Ny_coarse = 10, 11
Nx_fine, Ny_fine = 20, 21
a = 2.0
b = 1.0

# --- Check Shapes ---
if U_coarse.shape != (Nx_coarse + 1, Ny_coarse) or U_fine.shape != (Nx_fine + 1, Ny_fine):
    print("Error: Loaded array shapes do not match expected grid sizes.")
    exit()

# --- Generate Coordinate Arrays for Coarse Grid ---
x_coarse = np.linspace(0, a, Nx_coarse + 1)
y_coarse = np.linspace(-b, b, Ny_coarse)
X_coarse, Y_coarse = np.meshgrid(x_coarse, y_coarse, indexing='ij') # Corrected indexing

# --- Calculate Difference at Common Grid Points ---
# Create an array to store the difference, same shape as the coarse grid
Difference_Coarse_vs_Fine = np.zeros_like(U_coarse)

differences_list = []

# Iterate over the coarse grid indices
for i_c in range(Nx_coarse + 1):
    for j_c in range(Ny_coarse):
        # Find corresponding indices in the fine grid
        i_f = 2 * i_c
        j_f = 2 * j_c

        # Check if fine grid indices are within bounds
        if i_f < U_fine.shape[0] and j_f < U_fine.shape[1]:
            val_coarse = U_coarse[i_c, j_c]
            val_fine = U_fine[i_f, j_f]
            diff = val_fine - val_coarse # Fine - Coarse
            Difference_Coarse_vs_Fine[i_c, j_c] = diff
            differences_list.append(abs(diff))
        else:
            # Handle boundary points if necessary (though here indices should match)
            Difference_Coarse_vs_Fine[i_c, j_c] = np.nan # Mark as not comparable if needed

# --- Summary Statistics of Differences ---
if differences_list:
    max_diff = np.max(differences_list)
    avg_diff = np.mean(differences_list)
    print(f"\nMaximum absolute difference between grids at common points: {max_diff:.2e}")
    print(f"Average absolute difference between grids at common points: {avg_diff:.2e}")
else:
    print("\nNo common points found or compared.")

# --- Visualize the Difference ---
fig, ax = plt.subplots(figsize=(8, 7))

contour_levels_diff = 15
# Use contourf to plot the difference on the coarse grid coordinates
# Transpose Difference_Coarse_vs_Fine because contourf expects (y, x) indexing
diff_plot = ax.contourf(X_coarse, Y_coarse, Difference_Coarse_vs_Fine, levels=contour_levels_diff, cmap='coolwarm')

ax.set_title(f'Difference (Fine Grid - Coarse Grid) (Max: {max_diff:.2e})') # Corrected title string
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal', adjustable='box') # Corrected aspect string
fig.colorbar(diff_plot, shrink=0.7, aspect=15, label='U_fine - U_coarse')

plt.tight_layout()
plot_filename = "grid_difference_visualization.png"
plt.savefig(plot_filename)
print(f"\nDifference visualization saved to {plot_filename}")