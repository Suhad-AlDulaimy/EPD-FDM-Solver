# PDE Solver using Finite Difference Method
# Solves: u_xx - u_yy - (1/y)*u_y = 0
# Domain: 0 < x < a, -b < y < b
# BCs: u(0,0)=A, du/dx(0,0)=B, lim(y->0)[(1/y)du/dy]=x^2
# Derived BCs for FDM:
#   u(x, -b) = analytical(x, -b)
#   u(x, b) = analytical(x, b)
#   u(a, y) = analytical(a, y)
#   du/dx(0, y) = B (Neumann)
#   Singularity at y=0 handled via: u(x, 0) = u(x, dy) - x^2 * dy^2 / 2

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# --- Parameters ---
a = 2.0  # Domain limit for x
b = 1.0  # Domain limit for y
A = 0.0  # Boundary condition u(0,0)
B = 0.0  # Boundary condition du/dx(0,0)

Nx = 20  # Number of intervals in x -> Nx+1 points
Ny = 21 # Number of points in y (MUST BE ODD for y=0 to be a grid line)

dx = a / Nx
dy = 2 * b / (Ny - 1)

x = np.linspace(0, a, Nx + 1)
y = np.linspace(-b, b, Ny)

# Midpoint index for y=0
j_mid = (Ny - 1) // 2
if not np.isclose(y[j_mid], 0.0):
    print("Error: Ny must be odd for y=0 to be a grid line.")
    exit()

# Total number of unknowns: u_{i,j} for 0 <= i < Nx, 0 <= j < Ny
total_unknowns = Nx * Ny
M = np.zeros((total_unknowns, total_unknowns))
rhs = np.zeros(total_unknowns)

# Mapping from (i, j) to 1D index k
def get_k(i, j):
    # i runs from 0 to Nx-1
    # j runs from 0 to Ny-1
    return i * Ny + j

# Inverse mapping (for debugging)
def get_ij(k):
    i = k // Ny
    j = k % Ny
    return i, j

# Analytical solution function (for boundary conditions and validation)
def u_analytical_func(x_val, y_val, A_val, B_val):
    return x_val**4/6 + B_val*x_val + A_val + x_val**2*y_val**2/2 + y_val**4/16

print(f"Setting up {total_unknowns}x{total_unknowns} matrix...")

# --- Fill the Matrix M and RHS Vector rhs ---
for i in range(Nx): # 0 to Nx-1 (unknown columns)
    for j in range(Ny): # 0 to Ny-1 (all rows)
        k = get_k(i, j) # Current equation row (k-th unknown)

        # --- Boundary Conditions --- (These define the equation for the k-th unknown)
        # Dirichlet y=-b (j=0)
        if j == 0:
            M[k, k] = 1.0
            rhs[k] = u_analytical_func(x[i], y[j], A, B)
        # Dirichlet y=b (j=Ny-1)
        elif j == Ny - 1:
            M[k, k] = 1.0
            rhs[k] = u_analytical_func(x[i], y[j], A, B)
        # Singularity y=0 (j=j_mid)
        elif j == j_mid:
            # Equation derived from ∂²u/∂y²(x,0) = x² using central difference & symmetry
            # u_{i, j_mid} = u_{i, j_mid+1} - xᵢ² * dy² / 2
            k_mid = get_k(i, j_mid)
            k_mid_plus_1 = get_k(i, j_mid + 1)
            M[k, k_mid] = 1.0
            if j_mid + 1 == Ny - 1: # Next point is known boundary y=b
                rhs[k] = u_analytical_func(x[i], y[j_mid+1], A, B) - x[i]**2 * dy**2 / 2
            else: # Next point is unknown u(i, j_mid+1)
                M[k, k_mid_plus_1] = -1.0
                rhs[k] = -x[i]**2 * dy**2 / 2
        # --- Interior points (including i=0 Neumann boundary) ---
        else: # 0 < j < Ny-1 and j != j_mid
            # PDE Discretized: u_xx - u_yy - (1/y)*u_y = 0
            # Coeffs: c_xmm*u_{i-1,j} + c_x0*u_{i,j} + c_xpp*u_{i+1,j} + c_ymm*u_{i,j-1} + c_y0*u_{i,j} + c_ypp*u_{i,j+1} = 0

            coef_x_pp = 1.0 / dx**2
            coef_x_0 = -2.0 / dx**2
            coef_x_mm = 1.0 / dx**2
            coef_y_pp = -1.0 / dy**2 - 1.0 / (y[j] * 2 * dy)
            coef_y_0 = 2.0 / dy**2
            coef_y_mm = -1.0 / dy**2 + 1.0 / (y[j] * 2 * dy)

            # --- Handle x=0 Neumann Boundary (i=0) ---
            if i == 0:
                # Use central difference for Neumann: u_{-1,j} = u_{1,j} - 2*B*dx
                # Substitute into PDE at i=0:
                # coef_x_mm*(u_{1,j} - 2*B*dx) + coef_x_0*u_{0,j} + coef_x_pp*u_{1,j} + ... = 0
                # (coef_x_mm + coef_x_pp)*u_{1,j} + coef_x_0*u_{0,j} + ... = coef_x_mm * 2*B*dx

                k_0_j = get_k(0, j)
                k_1_j = get_k(1, j) # i+1 = 1 < Nx, so this is an unknown

                # u(0, j) term (diagonal)
                M[k, k_0_j] = coef_x_0 + coef_y_0
                # u(1, j) term
                M[k, k_1_j] = coef_x_mm + coef_x_pp
                # RHS term from Neumann BC
                rhs[k] = coef_x_mm * 2 * B * dx

                # y terms (check boundaries)
                k_j_plus_1 = get_k(i, j + 1)
                if j + 1 == Ny - 1: # Boundary y=b
                    rhs[k] -= coef_y_pp * u_analytical_func(x[i], y[j+1], A, B)
                # elif j + 1 == j_mid: # This case is handled separately
                #    pass
                else:
                    M[k, k_j_plus_1] = coef_y_pp

                k_j_minus_1 = get_k(i, j - 1)
                if j - 1 == 0: # Boundary y=-b
                    rhs[k] -= coef_y_mm * u_analytical_func(x[i], y[j-1], A, B)
                # elif j - 1 == j_mid: # This case is handled separately
                #    pass
                else:
                    M[k, k_j_minus_1] = coef_y_mm

            # --- Handle x=a Dirichlet Boundary (i=Nx-1) ---
            # The equation for u(Nx-1, j) involves u(Nx, j) which is known
            elif i == Nx - 1:
                # Base equation: coef_x_mm*u_{i-1,j} + coef_x_0*u_{i,j} + coef_x_pp*u_{i+1,j} + ... = 0
                # coef_x_mm*u_{Nx-2,j} + coef_x_0*u_{Nx-1,j} + coef_x_pp*u_{Nx,j} + ... = 0
                # Move known u_{Nx,j} term to RHS

                k_curr = get_k(i, j)
                k_prev = get_k(i - 1, j)

                # u(i, j) term (diagonal)
                M[k, k_curr] = coef_x_0 + coef_y_0
                # u(i-1, j) term
                M[k, k_prev] = coef_x_mm
                # RHS term from Dirichlet BC at x=a
                u_a_j = u_analytical_func(x[i+1], y[j], A, B)
                rhs[k] = -coef_x_pp * u_a_j

                # y terms (check boundaries)
                k_j_plus_1 = get_k(i, j + 1)
                if j + 1 == Ny - 1: # Boundary y=b
                    rhs[k] -= coef_y_pp * u_analytical_func(x[i], y[j+1], A, B)
                else:
                    M[k, k_j_plus_1] = coef_y_pp

                k_j_minus_1 = get_k(i, j - 1)
                if j - 1 == 0: # Boundary y=-b
                    rhs[k] -= coef_y_mm * u_analytical_func(x[i], y[j-1], A, B)
                else:
                    M[k, k_j_minus_1] = coef_y_mm

            # --- True Interior Points (0 < i < Nx-1) ---
            else:
                k_curr = get_k(i, j)
                k_prev_i = get_k(i - 1, j)
                k_next_i = get_k(i + 1, j)
                k_prev_j = get_k(i, j - 1)
                k_next_j = get_k(i, j + 1)

                # u(i, j) term (diagonal)
                M[k, k_curr] = coef_x_0 + coef_y_0
                # u(i-1, j) term
                M[k, k_prev_i] = coef_x_mm
                # u(i+1, j) term
                M[k, k_next_i] = coef_x_pp

                # y terms (check boundaries)
                if j + 1 == Ny - 1: # Boundary y=b
                    rhs[k] -= coef_y_pp * u_analytical_func(x[i], y[j+1], A, B)
                else:
                    M[k, k_next_j] = coef_y_pp

                if j - 1 == 0: # Boundary y=-b
                    rhs[k] -= coef_y_mm * u_analytical_func(x[i], y[j-1], A, B)
                else:
                    M[k, k_prev_j] = coef_y_mm

# --- Solve the Linear System ---
print("Solving the linear system...")
start_time = time.time()
try:
    solution_vector = np.linalg.solve(M, rhs)
    print(f"System solved in {time.time() - start_time:.2f} seconds.")

    # --- Reshape Solution ---
    U = np.zeros((Nx + 1, Ny))

    # Fill known boundary values first
    # y = -b (j=0) and y = b (j=Ny-1)
    for i_fill in range(Nx + 1):
        U[i_fill, 0] = u_analytical_func(x[i_fill], y[0], A, B)
        U[i_fill, Ny-1] = u_analytical_func(x[i_fill], y[Ny-1], A, B)
    # x = a (i=Nx)
    for j_fill in range(Ny):
        U[Nx, j_fill] = u_analytical_func(x[Nx], y[j_fill], A, B)

    # Fill calculated unknown values (i from 0 to Nx-1, j from 0 to Ny-1)
    for k_sol in range(total_unknowns):
        i_sol, j_sol = get_ij(k_sol) # Indices corresponding to the unknowns
        U[i_sol, j_sol] = solution_vector[k_sol]

    # --- Analytical Solution for Comparison ---
    X, Y = np.meshgrid(x, y, indexing='ij')
    U_analytical = u_analytical_func(X, Y, A, B)

    # --- Calculate Error ---
    # Compare the solved part (0 <= i < Nx, 0 <= j < Ny)
    error = np.abs(U[0:Nx, :] - U_analytical[0:Nx, :])
    max_error = np.max(error)
    avg_error = np.mean(error)
    print(f"Max absolute error (excluding x=a boundary): {max_error:.2e}")
    print(f"Average absolute error (excluding x=a boundary): {avg_error:.2e}")

    # --- Visualization ---
    # Create a 2x3 grid of plots (3D numerical, 3D analytical, 2D numerical, 2D analytical, error)
    fig = plt.figure(figsize=(15, 10))

    # Numerical Solution (3D)
    ax1 = fig.add_subplot(231, projection='3d')
    surf1 = ax1.plot_surface(X, Y, U, cmap='viridis', edgecolor='none')
    ax1.set_title(f'Numerical Solution (FDM {Nx}x{Ny})')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u(x,y)')
    fig.colorbar(surf1, shrink=0.5, aspect=10, ax=ax1)

    # Analytical Solution (3D)
    ax2 = fig.add_subplot(232, projection='3d')
    surf2 = ax2.plot_surface(X, Y, U_analytical, cmap='viridis', edgecolor='none')
    ax2.set_title('Analytical Solution (3D)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u(x,y)')
    fig.colorbar(surf2, shrink=0.5, aspect=10, ax=ax2)

    # Error Plot (3D)
    ax3 = fig.add_subplot(233, projection='3d')
    error_full = np.abs(U - U_analytical)
    surf3 = ax3.plot_surface(X, Y, error_full, cmap='hot', edgecolor='none')
    ax3.set_title(f'Absolute Error (3D) (Max: {np.max(error_full):.2e})')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('Error')
    fig.colorbar(surf3, shrink=0.5, aspect=10, ax=ax3)

    # Numerical Solution (2D Contour)
    ax4 = fig.add_subplot(234)
    contour_levels = 20
    contour_plot = ax4.contourf(X, Y, U, levels=contour_levels, cmap='viridis')
    ax4.set_title('Numerical Solution (2D Contour)')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_aspect('equal', adjustable='box')
    fig.colorbar(contour_plot, shrink=0.5, aspect=10, ax=ax4)

    # Analytical Solution (2D Contour) - NEW
    ax5 = fig.add_subplot(235)
    contour_plot_analytical = ax5.contourf(X, Y, U_analytical, levels=contour_levels, cmap='viridis')
    ax5.set_title('Analytical Solution (2D Contour)')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_aspect('equal', adjustable='box')
    fig.colorbar(contour_plot_analytical, shrink=0.5, aspect=10, ax=ax5)

    # Error Plot (2D Contour)
    ax6 = fig.add_subplot(236)
    contour_levels_err = 15
    error_plot = ax6.contourf(X, Y, error_full, levels=contour_levels_err, cmap='hot')
    ax6.set_title(f'Absolute Error (2D) (Max: {np.max(error_full):.2e})')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_aspect('equal', adjustable='box')
    fig.colorbar(error_plot, shrink=0.5, aspect=10, ax=ax6)

    plt.tight_layout()
    plot_filename = "pde_solution_comparison_AB0_with_analytical_2d.png"
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

except np.linalg.LinAlgError as e:
    print(f"Error solving linear system: {e}")
    print("The matrix might be singular or ill-conditioned. Check boundary conditions and discretization.")
    plot_filename = None # Indicate failure
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    plot_filename = None # Indicate failure