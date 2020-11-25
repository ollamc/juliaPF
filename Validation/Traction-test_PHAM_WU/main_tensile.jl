using SharedArrays
using DelimitedFiles
using .FEM

### Initialization
ndof = 3
ninpt = 4
nstatevar = 10

timestep = 50
file1 = "coord.txt"
file2 = "connec.txt"

cpus = 1

# Material Props
E = 6000
ν = 0.2
Gc = 0.04
lc = 10
xk = 10^-8

# Additional parameters
energySplit = false          # Activate Miehe's Stress Decomposition

# Output parameters
output = [1, 50]            # Output reaction force every output[1] timestep, output solution every output[2] timestep (VTK)

props = SharedArray([E, ν, Gc, lc, xk])

# Construct Dirichlet [Node, DOF, value]
nset1 = readdlm("nset1.txt", ',', Int64)
nset2 = readdlm("nset2.txt", ',', Int64)
nset3 = readdlm("nset3.txt", ',', Int64)

dirichlet = SharedArray(zeros(length(nset1)+length(nset1)+length(nset2)+length(nset3),3))

# Impose 0.0 to DDL 1 (Uy) on nset1
dirichlet[1:length(nset1),1] = nset1
dirichlet[1:length(nset1),2] .= 1
dirichlet[1:length(nset1),3] .= 0.0

# Impose 0.0 to DDL 2 (Ux) on nset1
dirichlet[length(nset1)+1:length(nset1)+length(nset1),1] = nset1
dirichlet[length(nset1)+1:length(nset1)+length(nset1),2] .= 2
dirichlet[length(nset1)+1:length(nset1)+length(nset1),3] .= 0.0

# Impose dispalcement to DDL 2 on nset2
dirichlet[length(nset1)+length(nset1)+1:length(nset1)+length(nset1)+length(nset2),1] = nset2
dirichlet[length(nset1)+length(nset1)+1:length(nset1)+length(nset1)+length(nset2),2] .= 2
dirichlet[length(nset1)+length(nset1)+1:length(nset1)+length(nset1)+length(nset2),3] .= 0.12

# Block damage (DDL3) on nset3
dirichlet[length(nset1)+length(nset1)+length(nset2)+1:end,1] = nset3
dirichlet[length(nset1)+length(nset1)+length(nset2)+1:end,2] .= 3
dirichlet[length(nset1)+length(nset1)+length(nset2)+1:end,3] .= 0.0

# List nodes on which the Reaction Force will be extracted
outputRF = dirichlet[1:11,1]

coord, connec = load(file1, file2)
coord = SharedArray(coord)
connec = SharedArray(connec)

# Initialize statevariable array
statevar = zeros(Float64,size(connec,1),ninpt,nstatevar)

# Assign number (address) to each DOFs
numer = SharedArray(ordering(coord, dirichlet, ndof))

@time U = solveNEWTON(coord, connec, dirichlet, numer, props, timestep, statevar, cpus, energySplit, output, outputRF)
