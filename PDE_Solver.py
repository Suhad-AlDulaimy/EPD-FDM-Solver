"""
Master EPD Solver Script 
===================================
Solves the Euler-Poisson-Darboux (EPD) equation:
    u_xx - u_yy - (mu/y)*u_y = 0

Domain: 0 < x < a, -b < y < b
Boundary Conditions:
    - Dirichlet on y=-b, y=b, x=a (from analytical solution)
    - Neumann on x=0: du/dx = B
    - Singularity at y=0: lim(y->0)[(mu/y)*du/dy] = f(x)

Analytical Solution (for verification):
    u(x,y) = x^4/6 + B*x + A + x^2*y^2/2 + y^4/16

Features:
- Input validation (Ny must be odd for y=0 to be a grid line)
- Grid refinement study with 4 grid sizes
- Convergence order calculation: p = log2(Ratio)
- Matrix condition number reporting
- 3D surface plots and error heatmaps for each grid
- Grid difference visualization (Fine vs. Coarse)
- Convergence table with orders
- Solution matrices saved as .npy files

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import norm as sparse_norm
import time
import warnings

# Suppress matplotlib warnings for clean output
warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================
# PARAMETERS
# ============================================================
a = 2.0   # Domain limit for x: [0, a]
b = 1.0   # Domain limit for y: [-b, b]
A = 0.0   # Boundary condition: u(0,0) = A
B = 0.0   # Boundary condition: du/dx(0,0) = B
mu = 1.0  # EPD parameter

def u_analytical_func(x, y):
    """
    Analytical solution: u(x,y) = x^4/6 + B*x + A + x^2*y^2/2 + y^4/16
    
    Verification that this satisfies u_xx - u_yy - (1/y)*u_y = 0:
        u_x  = 2x^3/3 + B + x*y^2
        u_xx = 2x^2 + y^2
        u_y  = x^2*y + y^3/4
        u_yy = x^2 + 3y^2/4
        (1/y)*u_y = x^2 + y^2/4
        
        u_xx - u_yy - (1/y)*u_y = (2x^2 + y^2) - (x^2 + 3y^2/4) - (x^2 + y^2/4) = 0
    """
    return x**4 / 6 + B * x + A + x**2 * y**2 / 2 + y**4 / 16

def f_singularity(x):
    """
    Problem-dependent singularity condition: f(x) = lim(y->0)[(mu/y)*u_y]
    For our analytical solution: f(x) = x^2
    
    This function can be modified for different problem specifications.
    """
    return x**2

# ============================================================
# SOLVER FUNCTION
# ============================================================
def solve_epd_master(Nx, Ny, report_condition=False):
    """
    Solve the EPD equation on a uniform grid using FDM with sparse matrices.
    
    Parameters:
        Nx (int): Number of intervals in x-direction
        Ny (int): Number of grid points in y-direction (MUST be odd)
        report_condition (bool): If True, compute and return the matrix condition number
    
    Returns:
        max_error (float): Maximum absolute error vs analytical solution
        avg_error (float): Average absolute error vs analytical solution
        U_numerical (ndarray): 2D array of numerical solution values
        U_analytical (ndarray): 2D array of analytical solution values
        x_coords (ndarray): x-coordinate array
        y_coords (ndarray): y-coordinate array
        cond_num (float or None): Matrix condition number (if report_condition=True)
    """
    # --- Input Validation ---
    assert Ny % 2 == 1, (
        f"Ny must be odd to ensure y=0 is a grid line. Got Ny={Ny}. "
        f"The singularity handling requires the grid line y=0 to exist exactly."
    )
    assert Nx > 0 and Ny > 2, "Grid must have at least Nx=1 and Ny=3."
    
    dx = a / Nx
    dy = 2 * b / (Ny - 1)
    x = np.linspace(0, a, Nx + 1)
    y = np.linspace(-b, b, Ny)
    j_mid = (Ny - 1) // 2
    
    # Verify that y[j_mid] is indeed 0
    assert abs(y[j_mid]) < 1e-14, (
        f"Grid construction error: y[j_mid] = {y[j_mid]} != 0. "
        f"Check Ny={Ny} and b={b}."
    )
    
    # Unknowns: i from 0 to Nx-1, j from 1 to Ny-2
    num_x_unknowns = Nx
    num_y_unknowns = Ny - 2
    total_unknowns = num_x_unknowns * num_y_unknowns

    rows = []
    cols = []
    data = []
    rhs = np.zeros(total_unknowns)

    def get_k(i, j):
        """Map 2D grid index (i,j) to 1D system index k. Returns -1 if on boundary."""
        if not (0 <= i <= Nx - 1 and 1 <= j <= Ny - 2):
            return -1
        return i * num_y_unknowns + (j - 1)

    # --- Matrix Assembly ---
    for i in range(Nx):
        for j in range(1, Ny - 1):
            k = get_k(i, j)
            
            # Coefficients for u_xx term
            coef_x_0 = -2.0 / dx**2
            coef_x_p1 = 1.0 / dx**2
            coef_x_m1 = 1.0 / dx**2
            
            # --- Singularity handling at y=0 ---
            if j == j_mid:
                # From Taylor expansion and symmetry:
                # u_{i,j_mid} = u_{i,j_mid+1} - f(x_i) * dy^2 / 2
                # Rearranged: u_{i,j_mid} - u_{i,j_mid+1} = -f(x_i) * dy^2 / 2
                data.append(1.0); rows.append(k); cols.append(k)
                nk = get_k(i, j + 1)
                if nk != -1:
                    data.append(-1.0); rows.append(k); cols.append(nk)
                else:
                    rhs[k] += u_analytical_func(x[i], y[j+1])
                rhs[k] -= f_singularity(x[i]) * dy**2 / 2
                continue

            # --- General interior point (j != j_mid) ---
            # Coefficients for -u_yy - (mu/y)*u_y terms
            coef_y_0 = 2.0 / dy**2   # from -u_yy: +2/dy^2
            coef_y_p1 = -1.0 / dy**2 - mu / (y[j] * 2 * dy)  # from -u_yy and -(mu/y)*u_y
            coef_y_m1 = -1.0 / dy**2 + mu / (y[j] * 2 * dy)  # from -u_yy and -(mu/y)*u_y

            # Diagonal term: sum of u_{i,j} coefficients from u_xx and -u_yy
            data.append(coef_x_0 + coef_y_0); rows.append(k); cols.append(k)
            
            # --- x-direction neighbors ---
            if i == 0:
                # Neumann BC at x=0: du/dx = B = 0
                # Ghost point: u_{-1,j} = u_{1,j} - 2*B*dx = u_{1,j} (since B=0)
                # u_xx at i=0: (u_{1,j} - 2*u_{0,j} + u_{1,j}) / dx^2 = 2*(u_{1,j} - u_{0,j}) / dx^2
                # Coefficient for u_{1,j}: 2/dx^2 (replaces coef_x_p1 + coef_x_m1)
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

            # --- y-direction neighbors ---
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

    # --- Assemble and Solve ---
    M = csr_matrix((data, (rows, cols)), shape=(total_unknowns, total_unknowns))
    
    # Condition number reporting 
    cond_num = None
    if report_condition:
        try:
            from scipy.sparse.linalg import svds
            # Estimate condition number using largest and smallest singular values
            s_max = svds(M, k=1, which='LM', return_singular_vectors=False)[0]
            s_min = svds(M, k=1, which='SM', return_singular_vectors=False)[0]
            cond_num = s_max / s_min
        except Exception as e:
            print(f"  Warning: Could not compute condition number: {e}")
            cond_num = None
    
    sol = spsolve(M, rhs)
    
    # --- Reconstruct Full Solution ---
    U = np.zeros((Nx + 1, Ny))
    # Fill Dirichlet boundaries
    for i_f in range(Nx + 1):
        U[i_f, 0] = u_analytical_func(x[i_f], y[0])
        U[i_f, Ny-1] = u_analytical_func(x[i_f], y[Ny-1])
    for j_f in range(Ny):
        U[Nx, j_f] = u_analytical_func(x[Nx], y[j_f])
    
    # Fill unknowns from solution vector
    for i_s in range(Nx):
        for j_s in range(1, Ny - 1):
            U[i_s, j_s] = sol[get_k(i_s, j_s)]
            
    # --- Error Analysis ---
    X, Y = np.meshgrid(x, y, indexing='ij')
    U_an = u_analytical_func(X, Y)
    err = np.abs(U - U_an)
    
    return np.max(err), np.mean(err), U, U_an, x, y, cond_num


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    
    print("=" * 70)
    print("  EPD SOLVER - GRID REFINEMENT STUDY (REVISED)")
    print("  Equation: u_xx - u_yy - (mu/y)*u_y = 0, mu = 1")
    print(f"  Domain: [0, {a}] x [{-b}, {b}]")
    print(f"  Analytical Solution: u = x^4/6 + {B}*x + {A} + x^2*y^2/2 + y^4/16")
    print("=" * 70)

    grids = [(10, 11), (20, 21), (40, 41), (80, 81)]
    results = []
    solutions = {}

    for idx, (nx, ny) in enumerate(grids):
        print(f"\n{'─' * 50}")
        print(f"  Grid {idx+1}/{len(grids)}: Nx={nx}, Ny={ny}")
        print(f"  dx = {a/nx:.4f}, dy = {2*b/(ny-1):.4f}")
        print(f"{'─' * 50}")
        
        start = time.time()
        # Report condition number only for the smallest grid 
        report_cond = (idx == 0)
        max_err, avg_err, U_num, U_an, x_coords, y_coords, cond_num = solve_epd_master(nx, ny, report_condition=report_cond)
        elapsed = time.time() - start
        
        results.append({
            'Nx': nx, 'Ny': ny, 
            'Max Error': max_err, 'Avg Error': avg_err,
            'Time (s)': elapsed
        })
        solutions[f'{nx}x{ny}'] = {'U_num': U_num, 'U_an': U_an, 'x': x_coords, 'y': y_coords}
        
        print(f"  Max Error:  {max_err:.6e}")
        print(f"  Avg Error:  {avg_err:.6e}")
        if cond_num is not None:
            print(f"  Condition Number: {cond_num:.2e}")
        print(f"  Solve Time: {elapsed:.3f}s")

        # Save solution matrices as .npy files
        np.save(f'U_numerical_{nx}x{ny}.npy', U_num)
        np.save(f'U_analytical_{nx}x{ny}.npy', U_an)
        print(f"  Saved: U_numerical_{nx}x{ny}.npy, U_analytical_{nx}x{ny}.npy")

        # Generate 3D comparison plots and error heatmap
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
        print(f"  Saved: {plot_filename}")
        plt.close()

    # ============================================================
    # CONVERGENCE TABLE WITH ORDERS
    # ============================================================
    df = pd.DataFrame(results)
    df['Max Ratio'] = df['Max Error'].shift(1) / df['Max Error']
    df['Avg Ratio'] = df['Avg Error'].shift(1) / df['Avg Error']
    df['Max Order p'] = np.log2(df['Max Ratio'])
    df['Avg Order p'] = np.log2(df['Avg Ratio'])

    print("\n" + "=" * 70)
    print("  CONVERGENCE TABLE")
    print("=" * 70)
    print(df[['Nx', 'Ny', 'Max Error', 'Avg Error', 'Max Ratio', 'Max Order p', 'Avg Ratio', 'Avg Order p']].to_string(index=False))
    df.to_csv('EPD_convergence_table.csv', index=False)
    print("\n  Saved: EPD_convergence_table.csv")
    
    # Print theoretical expectation
    print("\n  Theoretical expectation for 2nd-order scheme:")
    print("    Ratio -> 4.0, Order p -> 2.0")

    # ============================================================
    # GRID DIFFERENCE VISUALIZATION (Fine vs. Coarse)
    # ============================================================
    print("\n" + "=" * 70)
    print("  GRID DIFFERENCE VISUALIZATION")
    print("=" * 70)

    # Compare 20x21 (fine) with 10x11 (coarse)
    U_coarse = solutions['10x11']['U_num']
    U_fine = solutions['20x21']['U_num']
    x_coarse = solutions['10x11']['x']
    y_coarse = solutions['10x11']['y']

    Nx_coarse, Ny_coarse = 10, 11

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
        print(f"  Max |U_fine - U_coarse| at common points: {max_diff:.4e}")
        print(f"  Avg |U_fine - U_coarse| at common points: {avg_diff:.4e}")

    # Visualize the difference
    fig, ax = plt.subplots(figsize=(10, 8))
    X_coarse_mesh, Y_coarse_mesh = np.meshgrid(x_coarse, y_coarse, indexing='ij')

    diff_plot = ax.contourf(X_coarse_mesh, Y_coarse_mesh, Difference, levels=20, cmap='coolwarm')
    ax.set_title(f'Grid Difference: Fine (20x21) - Coarse (10x11)\n(Max: {max_diff:.2e})', fontsize=12)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_aspect('equal', adjustable='box')
    cbar = fig.colorbar(diff_plot, ax=ax, label='U_fine - U_coarse')

    plt.tight_layout()
    plot_filename = 'EPD_grid_difference.png'
    plt.savefig(plot_filename, dpi=150)
    print(f"  Saved: {plot_filename}")
    plt.close()

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("  All the generated files")
    print("=" * 70)
    print("\n  Generated Files:")
    print("    Plots:")
    for nx, ny in grids:
        print(f"      - EPD_solution_{nx}x{ny}.png")
    print(f"      - EPD_grid_difference.png")
    print("    Data:")
    print(f"      - EPD_convergence_table.csv")
    for nx, ny in grids:
        print(f"      - U_numerical_{nx}x{ny}.npy")
        print(f"      - U_analytical_{nx}x{ny}.npy")
    print("\n" + "=" * 70)
