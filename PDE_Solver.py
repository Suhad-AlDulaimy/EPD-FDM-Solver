"""
Master EPD Solver Script
Solves: u_xx - u_yy - (1/y)*u_y = 0
Domain: 0 < x < a, -b < y < b
BCs: u(0,0)=A, du/dx(0,0)=B, lim(y->0)[(1/y)du/dy]=x^2

Features:
- Grid refinement study with 4 grid sizes
- 3D surface plots and error heatmaps for each grid
- Grid difference visualization (Fine vs. Coarse)
- Convergence table
- Solution matrices saved as .npy files
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import time

# --- Parameters ---
a = 2.0  # Domain limit for x
b = 1.0  # Domain limit for y
A = 0.0  # Boundary condition u(0,0)
B = 0.0  # Boundary condition du/dx(0,0)
mu = 1.0 # EPD parameter

def u_analytical_func(x, y):
    """Analytical solution: u(x,y) = x^4/6 + Bx + A + x^2*y^2/2 + y^4/16"""
    return x**4 / 6 + B * x + A + x**2 * y**2 / 2 + y**4 / 16

def solve_epd_master(Nx, Ny):
    """
    Solve the EPD equation on a uniform grid using FDM.
    Returns: max_error, avg_error, U_numerical, U_analytical, x_coords, y_coords
    """
    dx = a / Nx
    dy = 2 * b / (Ny - 1)
    x = np.linspace(0, a, Nx + 1)
    y = np.linspace(-b, b, Ny)
    j_mid = (Ny - 1) // 2
    
    # Unknowns: i from 0 to Nx-1, j from 1 to Ny-2
    num_x_unknowns = Nx
    num_y_unknowns = Ny - 2
    total_unknowns = num_x_unknowns * num_y_unknowns

    rows = []
    cols = []
    data = []
    rhs = np.zeros(total_unknowns)

    def get_k(i, j):
        if not (0 <= i <= Nx - 1 and 1 <= j <= Ny - 2):
            return -1
        return i * num_y_unknowns + (j - 1)

    for i in range(Nx):
        for j in range(1, Ny - 1):
            k = get_k(i, j)
            
            # PDE: u_xx - u_yy - (1/y)*u_y = 0
            coef_x_0 = -2.0 / dx**2
            coef_x_p1 = 1.0 / dx**2
            coef_x_m1 = 1.0 / dx**2
            
            if j == j_mid:
                # Singularity handling: u_{i,j_mid} = u_{i,j_mid+1} - x_i^2 * dy^2 / 2
                data.append(1.0); rows.append(k); cols.append(k)
                nk = get_k(i, j + 1)
                if nk != -1:
                    data.append(-1.0); rows.append(k); cols.append(nk)
                else:
                    rhs[k] += u_analytical_func(x[i], y[j+1])
                rhs[k] -= x[i]**2 * dy**2 / 2
                continue

            # y-derivative coefficients
            coef_y_0 = 2.0 / dy**2
            coef_y_p1 = -1.0 / dy**2 - 1.0 / (y[j] * 2 * dy)
            coef_y_m1 = -1.0 / dy**2 + 1.0 / (y[j] * 2 * dy)

            # Add diagonal term
            data.append(coef_x_0 + coef_y_0); rows.append(k); cols.append(k)
            
            # Add x-neighbors
            if i == 0: # Neumann BC at x=0: du/dx = 0
                data.append(2.0 / dx**2); rows.append(k); cols.append(get_k(i + 1, j))
            else:
                # i-1 neighbor
                nk = get_k(i - 1, j)
                if nk != -1:
                    data.append(coef_x_m1); rows.append(k); cols.append(nk)
                else:
                    rhs[k] -= coef_x_m1 * u_analytical_func(x[i-1], y[j])
                
                # i+1 neighbor
                nk = get_k(i + 1, j)
                if nk != -1:
                    data.append(coef_x_p1); rows.append(k); cols.append(nk)
                else:
                    rhs[k] -= coef_x_p1 * u_analytical_func(x[i+1], y[j])

            # Add y-neighbors
            nk = get_k(i, j + 1)
            if nk != -1:
                data.append(coef_y_p1); rows.append(k); cols.append(nk)
            else:
                rhs[k] -= coef_y_p1 * u_analytical_func(x[i], y[j+1])
            
            nk = get_k(i, j - 1)
            if nk != -1:
                data.append(coef_y_m1); rows.append(k); cols.append(nk)
            else:
                rhs[k] -= coef_y_m1 * u_analytical_func(x[i], y[j-1])

    M = csr_matrix((data, (rows, cols)), shape=(total_unknowns, total_unknowns))
    sol = spsolve(M, rhs)
    
    U = np.zeros((Nx + 1, Ny))
    # Fill Dirichlet boundaries
    for i_f in range(Nx + 1):
        U[i_f, 0] = u_analytical_func(x[i_f], y[0])
        U[i_f, Ny-1] = u_analytical_func(x[i_f], y[Ny-1])
    for j_f in range(Ny):
        U[Nx, j_f] = u_analytical_func(x[Nx], y[j_f])
    
    # Fill unknowns
    for i_s in range(Nx):
        for j_s in range(1, Ny - 1):
            U[i_s, j_s] = sol[get_k(i_s, j_s)]
            
    X, Y = np.meshgrid(x, y, indexing='ij')
    U_an = u_analytical_func(X, Y)
    err = np.abs(U - U_an)
    return np.max(err), np.mean(err), U, U_an, x, y

# --- Grid Refinement Study ---
print("=" * 60)
print("EPD SOLVER - GRID REFINEMENT STUDY")
print("=" * 60)

grids = [(10, 11), (20, 21), (40, 41), (80, 81)]
results = []
solutions = {}  # Store solutions for later use

for nx, ny in grids:
    print(f"\nSolving for {nx}x{ny} grid...")
    start = time.time()
    max_err, avg_err, U_num, U_an, x_coords, y_coords = solve_epd_master(nx, ny)
    elapsed = time.time() - start
    
    results.append({'Nx': nx, 'Ny': ny, 'Max Error': max_err, 'Avg Error': avg_err})
    solutions[f'{nx}x{ny}'] = {'U_num': U_num, 'U_an': U_an, 'x': x_coords, 'y': y_coords}
    
    print(f"  Max Error: {max_err:.4e}, Avg Error: {avg_err:.4e}")
    print(f"  Time: {elapsed:.2f}s")

    # Save solution matrices as .npy files
    np.save(f'U_numerical_{nx}x{ny}.npy', U_num)
    np.save(f'U_analytical_{nx}x{ny}.npy', U_an)
    print(f"  Saved U_numerical_{nx}x{ny}.npy and U_analytical_{nx}x{ny}.npy")

    # Generate 3D comparison plots and error heatmap for each grid
    fig = plt.figure(figsize=(18, 6))
    X_plot, Y_plot = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    # Numerical Solution (3D)
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X_plot, Y_plot, U_num, cmap='viridis', edgecolor='none')
    ax1.set_title(f'Numerical Solution ({nx}x{ny})')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u(x,y)')
    
    # Analytical Solution (3D)
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X_plot, Y_plot, U_an, cmap='viridis', edgecolor='none')
    ax2.set_title('Analytical Solution')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u(x,y)')
    
    # Error Heatmap (2D)
    ax3 = fig.add_subplot(133)
    error_data = np.abs(U_num - U_an)
    error_plot = ax3.imshow(error_data.T, extent=[0, x_coords[-1], -b, b], 
                            origin='lower', aspect='auto', cmap='hot')
    ax3.set_title(f'Absolute Error (Max: {max_err:.2e})')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    plt.colorbar(error_plot, ax=ax3)
    
    plt.tight_layout()
    plot_filename = f'EPD_solution_{nx}x{ny}.png'
    plt.savefig(plot_filename, dpi=150)
    print(f"  Saved {plot_filename}")
    plt.close()

# --- Convergence Table ---
df = pd.DataFrame(results)
df['Max Ratio'] = df['Max Error'].shift(1) / df['Max Error']
df['Avg Ratio'] = df['Avg Error'].shift(1) / df['Avg Error']

print("\n" + "=" * 60)
print("CONVERGENCE TABLE")
print("=" * 60)
print(df.to_string(index=False))
df.to_csv('EPD_convergence_table.csv', index=False)
print("\nConvergence table saved to EPD_convergence_table.csv")

# --- Grid Difference Visualization (Fine vs. Coarse) ---
print("\n" + "=" * 60)
print("GENERATING GRID DIFFERENCE VISUALIZATION")
print("=" * 60)

# Compare 20x21 (fine) with 10x11 (coarse)
U_coarse = solutions['10x11']['U_num']
U_fine = solutions['20x21']['U_num']
x_coarse = solutions['10x11']['x']
y_coarse = solutions['10x11']['y']

Nx_coarse, Ny_coarse = 10, 11
Nx_fine, Ny_fine = 20, 21

# Calculate difference at common grid points
Difference = np.zeros_like(U_coarse)
differences_list = []

for i_c in range(Nx_coarse + 1):
    for j_c in range(Ny_coarse):
        i_f = 2 * i_c
        j_f = 2 * j_c
        
        if i_f < U_fine.shape[0] and j_f < U_fine.shape[1]:
            val_coarse = U_coarse[i_c, j_c]
            val_fine = U_fine[i_f, j_f]
            diff = val_fine - val_coarse
            Difference[i_c, j_c] = diff
            differences_list.append(abs(diff))

if differences_list:
    max_diff = np.max(differences_list)
    avg_diff = np.mean(differences_list)
    print(f"Maximum absolute difference (Fine - Coarse): {max_diff:.2e}")
    print(f"Average absolute difference (Fine - Coarse): {avg_diff:.2e}")

# Visualize the difference
fig, ax = plt.subplots(figsize=(10, 8))
X_coarse, Y_coarse = np.meshgrid(x_coarse, y_coarse, indexing='ij')

diff_plot = ax.contourf(X_coarse, Y_coarse, Difference, levels=20, cmap='coolwarm')
ax.set_title(f'Grid Difference: Fine (20x21) - Coarse (10x11)\n(Max: {max_diff:.2e})', fontsize=12)
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('y', fontsize=11)
ax.set_aspect('equal', adjustable='box')
cbar = fig.colorbar(diff_plot, ax=ax, label='U_fine - U_coarse')

plt.tight_layout()
plot_filename = 'EPD_grid_difference.png'
plt.savefig(plot_filename, dpi=150)
print(f"Saved {plot_filename}")
plt.close()

print("\n" + "=" * 60)
print("MASTER SCRIPT COMPLETED SUCCESSFULLY")
print("=" * 60)
print("\nGenerated Files:")
print("  - EPD_solution_10x11.png")
print("  - EPD_solution_20x21.png")
print("  - EPD_solution_40x41.png")
print("  - EPD_solution_80x81.png")
print("  - EPD_grid_difference.png")
print("  - EPD_convergence_table.csv")
print("  - U_numerical_*.npy (4 files)")
print("  - U_analytical_*.npy (4 files)")
print("\n" + "=" * 60)
