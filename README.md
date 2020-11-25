# juliaPF
Julia implementation(s) of phase-field models (PFMs) for fracture.

The script FEM.jl contains the Julia implementation of the quasi-static AT1 phase-field model.
Each time step is solved using the modified Newton scheme presented in the paper "An efficient and robust monolithic approach to phase-field quasi-static brittle fracture using a modified Newton method".

The script main_tensile.jl contains the pre-processing and settings of the problem. It calls the functions of FEM.jl.

## Validation
The **Validation** folder contains two examples of validation tests: 

1. SENP-tensile from Miehe 2010. ***Attention***: The script FEM.jl used to validate against Miehe's results contains the AT2 model.

2. Uniaxial traction test from Pham et al. 2011 and reused in Wu et al. 2018 (see the reference documents in the validation directory).

## Dependencies and usage
FEM.jl relies on the Pardiso and WriteVTK library. You can add them to your environment using, in the Julia session, the command "Pkg.add("Pardiso")" or "Pkg.add("WriteVTK")".
You might need to build Pardiso after adding the package with the command "Pkg.build("Pardiso")". For more detail see the Pardiso.jl package documentation (https://github.com/JuliaSparse/Pardiso.jl).

To run a simulation, 2 options are available:
1. 	- Open a terminal and place yourself in the directory containing the mesh files, the FEM.jl script and main_tensile.jl script.
	- Open a Julia session (if properly installed, on Linux, simply call "julia". Otherwise see Julia documentation).
	- To read and compile the FEM.jl script, execute: 	include("FEM.jl")
	- To launch the simulation, execute: 				include("main_tensile.jl")
	
2. If you have Atom (text editor) and Juno (Atom environment for the Julia language) installed, then simply:
	- Open and execute the FEM.jl file with Atom+Juno (Ctrl + Shift + Enter to execute the file).
	- Open and execute the main_tensile file with Atom+Juno (Ctrl + Shift + Enter to execute the file).

## Others
The **Others** folder contains new and/or non-validated implementation of the:
- Length-scale insensitive cohesive PFM (Wu and Nguyen 2018)
- AT2 model (Bourdin et al. 2000)

## Geometries
**Geometries** contains the mesh used for multiple geometries benchmarks:
- Single-edge notched plate submitted to tensile (**SENP-tensile**)
- Single-edge notched plate submitted to shear (**SENP-shear**)
- Three-point bending notched (**3Point-n**)
- Three-point bending pre-cracked (**3Point-c**)
- L-shaped panel (**L-shaped**)
- Notched bilayer (**Bilayer**)
