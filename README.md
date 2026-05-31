# Finite Difference Solver for the Euler–Poisson–Darboux Equation

The complete Python implementation used in this study is available in this repository.

The code includes:
- finite difference discretization of the EPD equation,
- matrix assembly procedures,
- singularity handling at y = 0,
- linear system solution routines,
- error analysis,
- grid refinement studies,
- and visualization scripts used to generate the numerical results and figures presented in the paper.
To generate Figure 2, run the solver twice: first with Nx=10, Ny=11, then with Nx=20, Ny=21. The comparison script will then load the saved output files automatically.
