# juliaPF
<img align="right" width="220" height="220" src="/images/cover.PNG">

**A Julia implementation of a phase-field model for fracture.**
<br />

The script `FEM.jl` contains the Julia implementation of the quasi-static **AT1** phase-field model with the volumetric-deviatoric split.
Each time step is solved using the modified Newton scheme presented in

> O. Lampron, D. Therriault, and M. Lévesque, “An efficient and robust monolithic approach to phase-field quasi-static brittle fracture using a modified Newton method,” Computer Methods in Applied Mechanics and Engineering, vol. 386, p. 114091, Dec. 2021, doi: 10.1016/j.cma.2021.114091.

A preprint of the paper is available at https://arxiv.org/abs/2109.05373.

The script `main.jl` contains the pre-processing and settings of the problem. It calls the functions of `FEM.jl`.


## Dependencies and usage
`FEM.jl` relies on the Pardiso and WriteVTK library. You can add them to your environment using, in the Julia session, the command `Pkg.add("Pardiso")` or `Pkg.add("WriteVTK")`.
You might need to build Pardiso after adding the package with the command `Pkg.build("Pardiso")`. For more detail see the `Pardiso.jl` package documentation (https://github.com/JuliaSparse/Pardiso.jl).

To run a simulation, 2 options are available:
1. 	- Open a terminal and place yourself in the directory containing the mesh files, the `FEM.jl` script and `main.jl` script.
	- Open a Julia session (if properly installed, on Linux, simply call `julia`. Otherwise see Julia documentation).
	- To read and compile the `FEM.jl` script, execute: 	`include("FEM.jl")`.
	- To launch the simulation, execute: 			`include("main.jl")`.
2. If you are using Atom (text editor) and Juno (Atom environment for the Julia language), then simply:
	- Open and execute the `FEM.jl` file with Atom+Juno (`Ctrl + Shift + Enter` to execute the file).
	- Open and execute the `main.jl` file with Atom+Juno (`Ctrl + Shift + Enter` to execute the file).

## Validation
The validation files contain the solution obtain for the **SENP-shear** test with the implemented model. The force-displacement solution is compared with results from the literature.

## Geometries
**Geometries** contains the mesh used for multiple geometries or benchmarks:
- Single-edge notched plate submitted to tensile (**SENP-tensile**).
- Single-edge notched plate submitted to shear (**SENP-shear**).
- Three-point bending notched beam (**3Point-n**).
- L-shaped panel (**L-shaped**).
- Notched bilayer (**Bilayer**).

## Notes
This project is licensed under the terms of the MIT license (see `LICENSE.txt`).

Author: Olivier Lampron
