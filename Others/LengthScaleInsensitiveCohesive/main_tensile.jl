using SharedArrays
using DelimitedFiles
using .FEM

### Initialization
ndof = 3
ninpt = 4
nstatevar = 10

timestep = 100
file1 = "coord.txt"
file2 = "connec.txt"

cpus = 8

# Material Props
E = 20800
ν = 0.3
Gc = 0.54
lc = 0.045
xk = 10^-8
ft = 208.47
p = 2
a2 = -0.5
a3 = 0

# Output parameters
output = [1, 5]            # Output reaction force every output[1] timestep, output solution every output[2] timestep (VTK)

# Construct array for material definition
props = SharedArray([E, ν, Gc, lc, xk, ft, p, a2, a3])

# Construct Dirichlet [Node, DOF, value]
nset1 = readdlm("nset1.txt", ',', Int64)
nset2 = readdlm("nset2.txt", ',', Int64)
nset3 = readdlm("nset3.txt", ',', Int64)
nset4 = readdlm("nset4.txt", ',', Int64)

dirichlet = SharedArray(zeros(length(nset1)+length(nset2)+length(nset3)+length(nset4),3))

# Impose 0.0 to DDL 1 on nset1
dirichlet[1:length(nset1),1] = nset1
dirichlet[1:length(nset1),2] .= 2
dirichlet[1:length(nset1),3] .= 0.0

# Impose 0.0 to DDL 2 on nset1
dirichlet[length(nset1)+1:length(nset1)+length(nset2),1] = nset2
dirichlet[length(nset1)+1:length(nset1)+length(nset2),2] .= 1
dirichlet[length(nset1)+1:length(nset1)+length(nset2),3] .= 0.0

# Impose 0.0 to DDL 2 on nset1
dirichlet[length(nset1)+length(nset2)+1:length(nset1)+length(nset2)+length(nset3),1] = nset3
dirichlet[length(nset1)+length(nset2)+1:length(nset1)+length(nset2)+length(nset3),2] .= 2
dirichlet[length(nset1)+length(nset2)+1:length(nset1)+length(nset2)+length(nset3),3] .= -0.1

# Impose 0.0 to DDL 3 on nset3
dirichlet[length(nset1)+length(nset2)+length(nset3)+1:end,1] = nset4
dirichlet[length(nset1)+length(nset2)+length(nset3)+1:end,2] .= 3
dirichlet[length(nset1)+length(nset2)+length(nset3)+1:end,3] .= 0.0

coord, connec = load(file1, file2)
coord = SharedArray(coord)
connec = SharedArray(connec)

# Initialize statevariable array
statevar = zeros(Float64,size(connec,1),ninpt,nstatevar)

# Assign number (adress) to each DOFs
numer = SharedArray(ordering(coord, dirichlet, ndof))

@time U = solveNEWTON(coord, connec, dirichlet, numer, props, timestep, statevar, cpus, output)
