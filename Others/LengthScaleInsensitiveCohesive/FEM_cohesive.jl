using Distributed
addprocs(7)

@everywhere module FEM

    using WriteVTK
    using Printf
    using DelimitedFiles
    using LinearAlgebra
    using SparseArrays
    using Pardiso
    using Distributed
    using SharedArrays

    export solveNEWTON
    export solveBFGS
    export load
    export ordering

    function matStiffness(E,ν)
            #Calculates material stiffness tensor for Isotropic material
            C = zeros(Float64, 4, 4)
            eg2 = E/(1.0+ν)
            elam = (E/(1.0-2.0*ν)-eg2)/3.0

            for i = 1:3
                    for j = 1:3
                            C[j,i] = elam
                    end
                    C[i,i] = eg2+elam
            end
            C[4,4] = eg2/2.0

            return C
    end

    function shapefcn(kintk,nnode)
            # Linear shape functions for 2D quadtrilateral
            # Return the shape functions and their derivatives evaluated at the i-th (kintk) integration point
            ndim = 2

            dN = zeros(Float64, nnode, 1)
            dNdz = zeros(Float64, ndim, nnode)

            #2D 4-nodes

            gausscoord = 0.577350269

            coord24 = [-1.0 -1.0;
                        1.0 -1.0;
                        1.0 1.0;
                       -1.0 1.0]

            #Integration point coordinates
            g = coord24[kintk, 1]*gausscoord
            h = coord24[kintk, 2]*gausscoord

            #shape functions
            dN[1,1]=(1.0-g)*(1.0-h)/4.0
            dN[2,1]=(1.0+g)*(1.0-h)/4.0
            dN[3,1]=(1.0+g)*(1.0+h)/4.0
            dN[4,1]=(1.0-g)*(1.0+h)/4.0

            #derivative d(Ni)/d(g)
            dNdz[1,1]=-(1.0-h)/4.0
            dNdz[1,2]=(1.0-h)/4.0
            dNdz[1,3]=(1.0+h)/4.0
            dNdz[1,4]=-(1.0+h)/4.0

            #derivative d(Ni)/d(h)
            dNdz[2,1]=-(1.0-g)/4.0
            dNdz[2,2]=-(1.0+g)/4.0
            dNdz[2,3]=(1.0+g)/4.0
            dNdz[2,4]=(1.0-g)/4.0

            return dN, dNdz
    end

    function jacobian(jelem, ndim, nnode, coords, dNdz)
            # Returns the jacobian for an element and
            # the derivatives of the shape functions in the global coordinates
            xjac = zeros(Float64, ndim, ndim)
            xjaci = zeros(Float64, ndim, ndim)
            dNdx = zeros(Float64, ndim, nnode)

            for n = 1:nnode
                    for i = 1:ndim
                            for j = 1:ndim
                                    xjac[j,i] = xjac[j,i]+dNdz[j,n]*coords[i,n]
                            end
                    end
            end
            # Compute determinant of Jacobian (2x2 matrix)
            djac = xjac[1,1]*xjac[2,2]-xjac[1,2]*xjac[2,1]

            # Compute inverse of Jacobian (2x2 matrix)
            if djac > 0.0
                    xjaci[1,1] = xjac[2,2]/djac
                    xjaci[2,2] = xjac[1,1]/djac
                    xjaci[1,2] = -xjac[1,2]/djac
                    xjaci[2,1] = -xjac[2,1]/djac
            else
                    println("WARNING: element ", jelem, " has negative Jacobian")
            end
            # Compute global space derivatives
            dNdx =  xjaci*dNdz

            return dNdx, djac
    end

    function degradation(d,p,a1,a2,a3)

            if d >= 1.0
                    d = 1.0
            end
            A = (1-d)^p
            A_p = -p*(1-d)^(p-1)
            A_pp = (p^2-p)*(1-d)^(p-2)

            B = a1*d+a1*a2*d^2+a1*a3*d^3
            B_p = a1+2*a1*a2*d+3*a1*a3*d^2
            B_pp = 2*a1*a2+6*a1*a3*d

            C = A*A_p
            C_p = A_p*A_p + A*A_pp

            w = A/(A+B)
            w_p = A_p/(A+B) - C/(A+B)^2 - A*B_p/(A+B)^2
            #w_p = (A_p*(A+B) - A*(A_p+B_p))/(A+B)^2
            w_pp = A_pp/(A+B) - (A_p*(A_p+B_p))/(A+B)^2 - C_p/(A+B)^2 + 2*C*(A_p+B_p)/(A+B)^3 - (A_p*B_p+A*B_pp)/(A+B)^2 + 2*A*B_p*(A_p+B_p)/(A+B)^3

            #w = (1-d)^2
            #w_p = -2*(1-d)
            #w_pp = 2

            return w, w_p, w_pp
    end

    function elStiffness(props, jelem, un_el, unp1_el, coords, ndof, statevar, type)
            #Takes in entry :
            # Material properties, element number, displacement / proposed displacement for element, coordinates of the mesh, number of DOF and state variables
            # Returns Residual and Hessian
            nnode = 4
            ndim = 2
            ninpt = 4
            ntens = 4

            # Material properties
            E = props[1]
            ν = props[2]
            Gc = props[3]
            lc = props[4]
            xk = props[5]
            ft = props[6]
            p = props[7]
            a2 = props[8]
            a3 = props[9]

            a1 = 4/(pi*lc)*E*Gc/ft^2
            ρ = Gc/lc*(1/pi^2)*(8/lc-2)/0.0001

            # Initialize element Residual and Hessian
            Ke = zeros(Float64, nnode*ndof, nnode*ndof)
            Re = zeros(Float64, nnode*ndof, 1)
            Ee = 0.0
            Es = 0.0

            # Get material stiffness
            C = matStiffness(E,ν)

            # Sum on all integration point
            for kintk = 1:ninpt
                    dN, dNdz = shapefcn(kintk,nnode)
                    dNdx, djac = jacobian(jelem, ndim, nnode, coords, dNdz)

                    B = zeros(Float64, 4, nnode*ndim)
                    for i = 1:nnode
                            B[1,2*i-1] = dNdx[1,i]
                            B[2,2*i] = dNdx[2,i]
                            B[4,2*i-1] = dNdx[2,i]
                            B[4,2*i] = dNdx[1,i]
                    end

                    #Compute phi from nodal values
                    phi = 0.0
                    phin = 0.0
                    for i = 1:nnode
                            phin += dN[i,1]*un_el[ndim*nnode+i]
                            phi += dN[i,1]*unp1_el[ndim*nnode+i]
                    end

                    if phin < 0.0
                            phin = 0.0
                    end
                    if phin > 1.0
                            phin = 1.0
                    end

                    # Get int. pt. variables from previous iterations
                    #stress = statevar[kintk, 1:ntens]
                    #strain = statevar[kintk, ntens+1:2*ntens]
                    #phin = statevar[kintk, 2*ntens+1]
                    Hn = statevar[kintk, 2*ntens+2]

                    # Compute strain and stress from previous time
                    strain = B*unp1_el[1:ndim*nnode,1]
                    stress = C*strain

                    # Obtain principal stresses (Plane strain)
                    principalStrain = zeros(4)
                    principalStrain[1] = (strain[1]+strain[2])/2+sqrt(((strain[1]-strain[2])/2)^2+(strain[4]^2))
                    principalStrain[2] = (strain[1]+strain[2])/2-sqrt(((strain[1]-strain[2])/2)^2+(strain[4]^2))
                    principalStress = C*principalStrain

                    # Compute Rankine criterion
                    sigmaEQ = max(0,principalStress[1],principalStress[2])

                    # Compute crack driving force
                    Y = 0.5*sigmaEQ^2/E
                    #Y = 0.5*dot(strain,stress)
                    H = Y
                    #if Y > Hn
                #        H = Y
                    #else
                        #    H = Hn
                    #end
                    #if H <= 0.5*ft^2/E
                    #        H = 0.5*ft^2/E
                    #end
                    #if H <= 3/16*Gc/lc
                    #        H = 3/16*Gc/lc
                    #end

                    #Update statevar
                    statevar[kintk, 1:ntens] = stress
                    statevar[kintk, ntens+1:2*ntens] = strain
                    statevar[kintk, 2*ntens+1] = phi
                    statevar[kintk, 2*ntens+2] = H

                    α = 2*phi - phi^2
                    α_p = 2-2*phi
                    α_pp = -2

                    w, w_p, w_pp = degradation(phi,p,a1,a2,a3)

                    if type
                            Re[1:8,1] += w*B'*stress*djac
                            Ke[1:8,1:8] += w*B'*C*B*djac

                    else
                            Re[9:12,1] += (2/pi*Gc*lc*dNdx'*dNdx*unp1_el[9:12,1]+dN[:,1]*(w_p*H+Gc/(pi*lc)*α_p)+ρ*min(phi-phin,0.0)*dN)*djac
                            #Re[9:12,1] += (2*3/8*Gc*lc*dNdx'*dNdx*unp1_el[9:12,1]+dN[:,1]*(3/8*Gc/lc*α_p+w_p*H)+γ*min(phi-phin,0.0)*dN)*djac
                            Ke[9:12,9:12] += (2/pi*Gc*lc*dNdx'*dNdx+dN*dN'*(w_pp*H+Gc/(pi*lc)*α_pp)+ρ*((phi-phin)<=0.0)*dN*dN')*djac
                            #Ke[9:12,9:12] += (2*3/8*dNdx'*dNdx*Gc*lc+dN*dN'*(3/8*Gc/lc*α_pp+w_pp*H)+γ*((phi-phin)<=0.0)*dN*dN')*djac
                    end
            end
            return Ke, Re, statevar
    end

    function ordering(coord, dirichlet, ndof)
            #Orders DOFS so that essential BCs are isolated in the system

            n = size(coord,1)
            n_dirichlet = size(dirichlet,1)
            numer=zeros(Int64,n,ndof)

            iddl = 1
            # Loop on all nodes
            for i = 1:n
                    # Loop on all 3 DOFs of each node
                    for j = 1:ndof
                            # IF node not even in dirichlet, gives num to all DOFS
                            if sum(in.(i,dirichlet[:,1])) == 0
                                    numer[i,j] = iddl
                                    iddl += 1
                            # If node is in dirichlet, check if it is the DOF that I have indexed
                            elseif sum(in.(i,dirichlet[:,1])) == 1
                                    index = findfirst(isequal(i), dirichlet[:,1])
                                    if dirichlet[index,2] != j
                                            numer[i,j] = iddl
                                            iddl += 1
                                    end
                            # If node is 2 times in Dirichlet, means that only the Phase Field DOF is free
                            elseif sum(in.(i,dirichlet[:,1])) == 2
                                    index = findall(isequal(i), dirichlet[:,1])
                                    if (dirichlet[index[1][1],2] != j) & (dirichlet[index[2][1],2] != j)
                                            numer[i,j] = iddl
                                            iddl += 1
                                    end
                            end
                    end
            end
            # Give number to DOFs with applied Dirichlet
            for i =1:n_dirichlet
                    numer[convert(Int64,dirichlet[i,1]),convert(Int64,dirichlet[i,2])] = iddl
                    iddl += 1
            end
            return numer
    end

    function assemble(Un, Unp1, coord, connec, dirichlet, numer, props, statevar, type)
              # Assembles the global system (Global Residual and Hessian) using the connectivity table (connec)

              nel, ndk = size(connec)
              nnode, dim = size(coord)

              nddl_t = size(numer,1)*size(numer,2)

              # Initialize global objective, residual and hessian (Sparse assembly for Hessian)
              I = SharedArray{Int,1}(12*12*nel)
              J = SharedArray{Int,1}(12*12*nel)
              V = SharedArray{Float64,1}(12*12*nel)
              X = SharedArray{Int,1}(12*nel)
              Y = SharedArray{Float64,1}(12*nel)
              O = SharedArray{Float64,1}(nel)
              K = 0.0
              R = 0.0

              Hessian = true
              Residual = true

              @sync @inbounds @distributed for el = 1:nel
                      # For phase field (3DOFs), do the same but with only 2 firsts columns of numer, then add third (containing Phase Field)
                      Ladress = hcat(reshape(numer[connec[el,:],[1, 2]]',(1,:)),numer[connec[el,:],3]')
                      coords = coord[connec[el,:],:]'
                      unp1_el = Unp1[Ladress,1]'
                      un_el = Un[Ladress,1]'
                      # Knowing the coordinates and proposed displacement associated to the i-th (el) element, compute element's Residual and Hessian
                      Ke, Re, statevar[el,:,:] = elStiffness(props, el, un_el, unp1_el, coords, size(numer,2), statevar[el,:,:], type)
                      # Add to the global system
                      if Hessian
                              ct_k = el*144-143
                              for i= 1:12
                                      for j = 1:12
                                              I[ct_k] = Ladress[i]
                                              J[ct_k] = Ladress[j]
                                              V[ct_k] = Ke[i,j]
                                              ct_k += 1
                                      end
                              end
                      end
                      if Residual
                              ct_r = el*12-11
                              for i= 1:12
                                      X[ct_r] = Ladress[1,i]
                                      Y[ct_r] = Re[i,1]
                                      ct_r += 1
                              end
                      end
              end
              if Residual
                      R = Vector(sparsevec(X,Y,nddl_t,+))
              end
              if Hessian
                      K = sparse(I,J,V, nddl_t, nddl_t, +)
              end

              return K, R, statevar

     end

    function solveNEWTON(coord, connec, dirichlet, numer, props, timestep, statevar, cpus, output)
            # Set Newton's parameters
            #tol = 10^-5
            maxRes = 1.0E-4
            maxit = 10000

            # Initialize call array to store number of calls per increment
            call = []

            # Initialize vectors
            n = size(dirichlet,1)                   # Number of DOF in dirichlet set
            s = size(numer,1)*size(numer,2)         # Number of DOF in problem
            Ud = zeros(n,1)                         # Total displacement imposed on dirichlet set
            Un = zeros(s,1)                         # Solution at increment n
            Unp1 = zeros(s,1)                       # Solution at increment n plus one
            δu = zeros(s-n,1)                       # Correction on u
            δu_d = zeros(n,1)                       # Fraction of displacement applied to dirichlet set
            statevar_temp = statevar

            adress_phi = vec(numer[:,3]')
            adress_u = vcat(numer[:,1],numer[:,2])
            adress_d = collect(s-n+1:s)
            adress_u = setdiff(adress_u, adress_d)

            # Initialize Pardiso Solver
            ps = MKLPardisoSolver()
            set_nprocs!(ps, cpus)
            set_matrixtype!(ps, 11)

            # Get total imposed displacement from boundary conditions (Dirichlet)
            for i = 1:n
                    index = numer[convert(Int64,dirichlet[i,1]),convert(Int64,dirichlet[i,2])]-(s-n)
                    Ud[index] = dirichlet[i,3]
            end

            # Write initial configuration to outfile
            postprocess(coord,connec,numer,Unp1,0,statevar)

            # Start adding increment of displacement
            for inc = 1:timestep

                    print("Increment: ", inc, "\n")
                    print("Iter \t ||∇f|| \t type  \n")

                    # Prediction: same displacement as previous increment, update imposed displacements
                    δu_d = Ud*inc/timestep
                    Un[:] = Unp1
                    Unp1[s-n+1:end,1] = δu_d

                    # Get Residual and Hessian from prediction
                    K, R, statevar_temp = assemble(Un, Unp1, coord, connec, dirichlet, numer, props, statevar, true)
                    call_inc = 0

                    @printf("  %.0f \t %.4e \t u\n", 0, norm(R[1:s-n,1]))

                    # Apply Newton's method to minimize residual (find equilibrium for imposed displacement)
                    # Double loop as proposed by Gerasimov and De Lorenzis
                    it = 1
                    while (norm(R[adress_u,1],Inf) > maxRes)

                            if it > maxit
                                    println("ERROR: NEWTON NEEDED MORE THAN MAXIT ITERATIONS\n")
                                    @goto escape_label
                            end

                            while (norm(R[adress_u,1],Inf) > 10^-5)
                                δu = solve(ps, -K[adress_u,adress_u], R[adress_u])

                                # Apply correction and recalculate Residual and Hessian
                                Unp1[adress_u,1] += δu
                                call_inc += 1
                                K, R, statevar_temp = assemble(Un, Unp1, coord, connec, dirichlet, numer, props, statevar, true)

                                @printf("  %.0f \t %.4e \t u \n", call_inc,norm(R[adress_u,1]))

                            end

                            K, R, statevar_temp = assemble(Un, Unp1, coord, connec, dirichlet, numer, props, statevar,false)

                            while (norm(R[adress_phi,1],Inf) > 10^-5)
                                    # Solve Newton's linear system for correction
                                    #δu = -K[1:s-n,1:s-n] \ R[1:s-n,1]
                                    δϕ = solve(ps, -K[adress_phi,adress_phi], R[adress_phi])

                                    # Apply correction and recalculate Residual and Hessian
                                    Unp1[adress_phi,1] += δϕ
                                    call_inc += 1
                                    K, R, O = assemble(Un, Unp1, coord, connec, dirichlet, numer, props, statevar, false)

                                    @printf("  %.0f \t %.4e \t ϕ \n", call_inc, norm(R[adress_phi,1]))
                            end
                            K, R, statevar_temp = assemble(Un, Unp1, coord, connec, dirichlet, numer, props, statevar, true)

                    end
                    # Store number of calls for increment
                    push!(call,call_inc)
                    print("\n")
                    statevar = statevar_temp

                    if (rem(inc,output[2]) == 0)
                            postprocess(coord,connec,numer,Unp1,inc,statevar)
                    end

                    # Write results
                    if (rem(inc,output[1]) == 0)
                            # Get dirichlet nodes
                            d = convert.(Int,dirichlet[1:2,1])
                            K, R, statevar = assemble(Un, Unp1, coord, connec, dirichlet, numer, props, statevar, true)

                            # Obtain reaction forces
                            s = size(numer,1)*size(numer,2)
                            n = size(dirichlet,1)
                            RF = zeros(s,1)
                            RF[s-n+1:end,1] = K[s-n+1:end,:]*Unp1

                            #Associate RF from DDL to node
                            RF1 = zeros(size(coord,1),1)
                            RF2 = zeros(size(coord,1),1)
                            RF1 = RF[numer[:,1]]
                            RF2 = RF[numer[:,2]]

                            # Extract RF on Dirichlet nodes
                            RF1_d = RF1[d]
                            RF2_d = RF2[d]
                            RF1_sum = sum(RF1_d)
                            RF2_sum = sum(RF2_d)

                            # Write to file
                            file = open("RF.txt", "a")
                            writedlm(file, [inc/timestep RF1_sum RF2_sum])
                            close(file)
                    end
            end
            @label escape_label
            print("Call \n")
            print(call)
            print("\n")
            return Unp1
    end

    function armijo(δu, Un, Unp1, Obj, R, s, n, coord, connec, mat, dirichlet, numer, props, statevar, energySplit, type, adress)
            #slope = dot(δu,R[1:s-n,1])
            # Perform normal armijo
            t = 1.0
            u_temp = zeros(size(Unp1))
            u_temp += Unp1
            u_temp[adress,1] += δu*t
            K_temp, R_temp, obj_temp = assemble(Un, u_temp, coord, connec, mat, dirichlet, numer, props, statevar, true, true, false, energySplit, type)
            it = 1
            while (norm(R_temp[adress,1]) > norm(R[adress,1])) & (it <= 8) #+ 0.0 * t * slope)) #& (obj_temp > 0.0001)
                    t /= 2.0
                    u_temp = zeros(size(Unp1))
                    u_temp += Unp1
                    u_temp[adress,1] += δu*t
                    K_temp, R_temp, obj_temp = assemble(Un, u_temp, coord, connec, mat, dirichlet, numer, props, statevar, true, true, false, energySplit, type)
                    it += 1
            end
            if it == 9
                    t = 0.0
            end
            return t
    end

    function postprocess(coord,connec,numer,U,timestep,statevar)

            #Prepare nodal values for export
            Ux = zeros(size(coord,1),1)
            Uy = zeros(size(coord,1),1)
            phi = zeros(size(coord,1),1)
            if timestep != 0
                    Ux = U[numer[:,1]]
                    Uy = U[numer[:,2]]
                    phi = U[numer[:,3]]
            end

            # Push connectivity array to VTK file
            cells = MeshCell[]
            for i = 1:size(connec,1)
                    inds = Array{Int32}(undef, 4)
                    inds = connec[i,:]

                    c = MeshCell(VTKCellTypes.VTK_QUAD, inds)
                    push!(cells, c)
            end

            # Write nodal scalars to VTK file
            name = string("Results_",timestep)
            vtkfile = vtk_grid(name, coord', cells)

            vtkfile["Ux", VTKPointData()] = Ux
            vtkfile["Uy", VTKPointData()] = Uy
            vtkfile["phi", VTKPointData()] = phi

            vtk_save(vtkfile)

            #Prepare for int. point values export
            coord_int = zeros(size(connec,1)*4,2)
            strain_y = zeros(size(connec,1)*4,1)

            # For each element, get the coordinates of the 4 integration point
            for i = 1:size(connec,1)
                    coord_int[i*4-3:i*4,:] = getIntCoord(coord[connec[i,:],:])
                    strain_y[i*4-3:i*4,1] = statevar[i,:,6]
            end

            # Push virtual connectivity table for integration points
            cells = MeshCell[]
            for i = 1:size(connec,1)
                    inds = Array{Int32}(undef, 4)
                    inds = collect(i*4-3:i*4)

                    c = MeshCell(VTKCellTypes.VTK_POLY_VERTEX, inds)
                    push!(cells, c)
            end

            # Write int. point scalars to VTK file
            name = string("IntPt_",timestep)
            vtkfile = vtk_grid(name, coord_int', cells)
            vtkfile["Strain_y", VTKPointData()] = strain_y
            vtk_save(vtkfile)
    end

    function getIntCoord(coord)
            # Returns the coordinates of each integration point of an element using the nodes of the element
            coord_int = 0.577350269* [-1.0 -1.0;
                                    1.0 -1.0;
                                    1.0 1.0;
                                    -1.0 1.0]

            int_c = zeros(4,2)

            for i = 1:4
                    dN = zeros(4,1)

                    g = coord_int[i,1]
                    h = coord_int[i,2]

                    dN[1,1]=(1.0-g)*(1.0-h)/4.0
                    dN[2,1]=(1.0+g)*(1.0-h)/4.0
                    dN[3,1]=(1.0+g)*(1.0+h)/4.0
                    dN[4,1]=(1.0-g)*(1.0+h)/4.0

                    int_c[i,:] = transpose(dN)*coord
            end
            return(int_c)
    end

    function load(fileNode,fileEl)
            # Read files containing the mesh definition and the connectivity table

            coord = readdlm(fileNode, ',', Float64)
            connec = readdlm(fileEl, ',', Int64)

            return coord[:,2:end], connec[:,2:end]
    end
end
