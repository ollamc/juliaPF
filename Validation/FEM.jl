
"""This code is licensed under the MIT License.

Copyright (c) 2021 Olivier Lampron

---

Functions shapefcn(), jacobian(), elstiffness() were modified from

[1] E. Martínez-Pañeda, A. Golahmar, C.F. Niordson. A phase field formulation for hydrogen assisted cracking. Computer Methods in Applied Mechanics and Engineering, 342: 742-761 (2018)

distributed under the BSD-style license:

Copyright (c) 2018, Emilio Martínez Pañeda
All rights reserved.

See LICENSE.txt file for details."""

using Distributed
addprocs(5)

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
    export load
    export ordering

    function tangentMod(trE,k,μ,phi,xk)
            # Calculates material stiffness tensor for Isotropic material
            C = zeros(Float64, 4, 4)

            for i = 1:3
                    for j = 1:3
                            C[j,i] = (trE<0)*k+((1-phi)^2+xk)*((trE>=0)*k-μ)
                    end
                    C[i,i] = (trE<0)*k+((1-phi)^2+xk)*((trE>=0)*k+μ)
            end
            C[4,4] = ((1-phi)^2+xk)*μ

            return C
    end

    function shapefcn(kintk,nnode)
            # Modified from [1]
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
            # Modified from [1]
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
            end
            # Compute global space derivatives
            dNdx =  xjaci*dNdz

            return dNdx, djac
    end

    function elStiffness(props, jelem, un_el, unp1_el, coords, ndof, statevar, Objective, Residual, Hessian)
            # Modified from [1]
            # Takes in entry :
            # Material properties, element number, displacement / proposed displacement for element, coordinates of the mesh, number of DOF and state variables (optional)
            # Returns Objective, Residual and Hessian

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

            # Initialize element Residual and Hessian
            Ke = zeros(Float64, nnode*ndof, nnode*ndof)
            Re = zeros(Float64, nnode*ndof, 1)
            Oe = 0.0

            # Get material stiffness
            μ = E/(2.0*(1.0+ν))
            λ = E*ν/((1.0+ν)*(1.0-2.0*ν))
            k =  λ + 2.0*μ/2

            # Penalty coefficient
            TOL = 0.01
            γ = Gc/lc*(27/64)/TOL^2
            Id = [1 1 1 0]'

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

                    #Compute phi (damage) from nodal values
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

                    # Compute strain and stress from previous time
                    strain = B*unp1_el[1:ndim*nnode,1]
                    strainDevS = [0.5*(strain[1]-strain[2]); 0.5*(strain[2]-strain[1]); 0; strain[4]]
                    strainDev = [0.5*(strain[1]-strain[2]); 0.5*(strain[2]-strain[1]); 0; strain[4]/2]
                    trE = strain[1]+strain[2]

                    # Compute strain energy density and stress from the current increment
                    PsiComp = k/2*trE^2*(trE<0)
                    PsiTens = k/2*trE^2*(trE>=0)+μ*dot(strainDev,strainDevS)
                    stress = k*trE*(trE<0)*Id + ((1-phi)^2+xk)*((k*trE*(trE>=0))*Id+2*μ*strainDev)

                    #Update statevar
                    #statevar[kintk, 1:ntens] = stress
                    #statevar[kintk, ntens+1:2*ntens] = strain
                    #statevar[kintk, 2*ntens+1] = phi

                    # Compute Residual
                    if Residual
                            Re[1:8,1] += B'*stress*djac
                            Re[9:12,1] += (0.75*Gc*lc*dNdx'*dNdx*unp1_el[9:12,1]+dN[:,1]*(0.375*Gc/lc+2.0*PsiTens*phi-2.0*PsiTens)+γ*min(phi-phin,0.0)*dN)*djac
                    end

                    # Compute Hessian
                    if Hessian
                            C = tangentMod(trE,k,μ,phi,xk)
                            Ke[1:8,1:8] += B'*C*B*djac
                            Ke[9:12,9:12] += (0.75*dNdx'*dNdx*Gc*lc+2.0*dN*dN'*PsiTens+γ*((phi-phin)<=0.0)*dN*dN')*djac
                            Ke[1:8,9:12] += -2*(1-phi)*B'*((k*trE*(trE>=0))*Id+2*μ*strainDev)*dN'*djac
                            Ke[9:12,1:8] += -2*(1-phi)*dN*((k*trE*(trE>=0))*Id+2*μ*strainDev)'*B*djac
                    end

                    if Objective
                            Oe += (((1-phi)^2+xk)*PsiTens+PsiComp+0.375*Gc*(phi/lc + lc*(dNdx*unp1_el[9:12,1])'*(dNdx*unp1_el[9:12,1]))+0.5*γ*min(phi-phin,0.0)^2)*djac
                    end

            end
            return Ke, Re, Oe, statevar
    end

    function ordering(coord, dirichlet, ndof)
            #Order DOFS so that essential (Dirichlet) BCs are isolated in the system

            n = size(coord,1)
            n_dirichlet = size(dirichlet,1)
            numer=zeros(Int64,n,ndof)

            iddl = 1
            # Loop on all nodes
            for i = 1:n
                    # Count how many times node i appears in Dirichlet
                    sum_i = sum(in.(i,dirichlet[:,1]))
                    # Loop on all 3 DOFs of each node
                    for j = 1:ndof
                            # IF node not even in dirichlet, gives num to all DOFS
                            if sum_i == 0
                                    numer[i,j] = iddl
                                    iddl += 1
                            # If node is in dirichlet[] once, check if it is the DOF that I have indexed
                            elseif sum_i == 1
                                    index = findfirst(isequal(i), dirichlet[:,1])
                                    if dirichlet[index,2] != j
                                            numer[i,j] = iddl
                                            iddl += 1
                                    end
                            # If node is 2 times in Dirichlet, check that DOF is not the one in dirichlet[]
                            elseif sum_i == 2
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

    function assemble(Un, Unp1, coord, connec, dirichlet, numer, props, statevar, Objective, Residual, Hessian)
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

              @inbounds @sync @distributed for el = 1:nel
                      Laddress = hcat(reshape(numer[connec[el,:],[1, 2]]',(1,:)),numer[connec[el,:],3]')        # Get addresses of all DOFs of the element
                      coords = coord[connec[el,:],:]'                                                           # Get coordinates of all nodes of the element
                      unp1_el = Unp1[Laddress,1]'                                                               # Get nodal values at Unp1
                      un_el = Un[Laddress,1]'                                                                   # Get nodal values at Un
                      # Knowing the coordinates and proposed displacement associated to the i-th (el) element, compute element's Hessian, Residual and Objective
                      Ke, Re, Oe, statevar[el,:,:] = elStiffness(props, el, un_el, unp1_el, coords, size(numer,2), statevar[el,:,:], Objective, Residual, Hessian)
                      # Add to the global system
                      if Hessian
                              ct_k = el*144-143
                              for i= 1:12
                                      for j = 1:12
                                              I[ct_k] = Laddress[i]
                                              J[ct_k] = Laddress[j]
                                              V[ct_k] = Ke[i,j]
                                              ct_k += 1
                                      end
                              end
                      end
                      if Residual
                              ct_r = el*12-11
                              for i= 1:12
                                      X[ct_r] = Laddress[1,i]
                                      Y[ct_r] = Re[i,1]
                                      ct_r += 1
                              end
                      end
                      if Objective
                              O[el] = Oe
                      end
              end
              if Residual
                      R = Vector(sparsevec(X,Y,nddl_t,+))
              end
              if Hessian
                      K = sparse(I,J,V, nddl_t, nddl_t, +)
              end

              return K, R, sum(O)

     end

    function solveNEWTON(coord, connec, dirichlet, numer, props, timestep, statevar, cpus, output, outputRF)
            # Set Newton's parameters
            #tol = 10^-5
            maxRes = 1.0E-4
            maxit = 10000
            Objective = true
            Residual = true
            Hessian = true

            # Initialize call array to store number of calls per increment
            call = []

            # Initialize vectors
            n = size(dirichlet,1)                   # Number of DOF in dirichlet set
            s = size(numer,1)*size(numer,2)         # Number of DOF in problem
            Ud = zeros(n,1)                         # Total displacement imposed on dirichlet set
            Un = zeros(s,1)                         # Solution at increment n
            Unp1 = zeros(s,1)                       # Solution at increment n plus one
            δu = zeros(s-n)                         # Correction
            δu_d = zeros(n,1)                       # Fraction of displacement applied to dirichlet set

            # Initialize Pardiso Solver
            ps = MKLPardisoSolver()
            set_nprocs!(ps, cpus)
            set_matrixtype!(ps, -2)                 # Set for Symmetric Indefinite matrices

            adress_phi = vec(numer[:,3]')
            adress_u_t = vcat(numer[:,1],numer[:,2])
            adress_d = collect(s-n+1:s)
            adress_u = setdiff(adress_u_t, adress_d)

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
                    print("Iter \t Objective \t ||∇f|| \t log(δ_w) \t t \n")

                    # Prediction: same displacement as previous increment, update imposed displacements
                    δu_d = Ud*inc/timestep
                    Un[:] = Unp1
                    Unp1[s-n+1:end,1] = δu_d

                    # Get Residual and Hessian from prediction
                    K, R, O = assemble(Un, Unp1, coord, connec, dirichlet, numer, props, statevar, Objective, Residual, Hessian)
                    call_inc = 0

                    @printf("  %.0f \t %.4e \t %.4e \t - \t - \n", 0, O, norm(R[1:s-n,1]))

                    # Initialize correction
                    δw_last = 0

                    # Apply Newton's method to minimize residual (find equilibrium for imposed displacement)
                    it = 1
                    while (norm(R[1:s-n,1],Inf) > maxRes)

                            if it > maxit
                                    println("ERROR: NEWTON NEEDED MORE THAN MAXIT ITERATIONS\n")
                                    @goto escape_label
                            end

                            Kt = tril(K[1:s-n,1:s-n])          # Extract lower triangular block
                            # Solve Newton's linear system for correction
                            pardiso(ps, δu, Kt, -R[1:s-n])
                            eigenP = get_iparm(ps, 22)
                            eigenN = get_iparm(ps, 23)
                            δw = 0

                            if (eigenN != 0)
                                    if δw_last == 0
                                            δw = 1e-4
                                    else
                                            δw = max(1e-20,0.33*δw_last)
                                    end

                                    pardiso(ps, δu, Kt+δw*I, -R[1:s-n])
                                    eigenP = get_iparm(ps, 22)
                                    eigenN = get_iparm(ps, 23)

                                    while (eigenN != 0) & (δw < 1e40)
                                            if δw_last == 0
                                                    δw = 100*δw
                                            else
                                                    δw = 8*δw
                                            end

                                            pardiso(ps, δu, Kt+δw*I, -R[1:s-n])
                                            eigenP = get_iparm(ps, 22)
                                            eigenN = get_iparm(ps, 23)
                                    end

                                    δw_last = δw
                            end

                            # Compute the step size parameter using an Armijo backtracking
                            t = armijo(δu, Un, Unp1, O, R, s, n, coord, connec, dirichlet, numer, props, statevar)
                            # Apply correction and recalculate Residual and Hessian
                            Unp1[1:s-n,1] += δu*t
                            call_inc += 1
                            K, R, O = assemble(Un, Unp1, coord, connec, dirichlet, numer, props, statevar, Objective, Residual, Hessian)

                            @printf("  %.0f \t %.4e \t %.4e \t %.2e \t %.2f \n", it, O, norm(R[1:s-n,1]), log10(δw), t)
                            it += 1
                    end
                    # Store number of calls for increment
                    push!(call,call_inc)
                    print("\n")

                    if (rem(inc,output[2]) == 0)
                            postprocess(coord,connec,numer,Unp1,inc,statevar)
                    end

                    # Write results
                    if (rem(inc,output[1]) == 0)
                            # Get dirichlet nodes
                            d = convert.(Int,outputRF)
                            K, R, O = assemble(Un, Unp1, coord, connec, dirichlet, numer, props, statevar, false, false, Hessian)

                            # Obtain reaction forces
                            s = size(numer,1)*size(numer,2)
                            n = size(dirichlet,1)
                            RF = zeros(s,1)
                            RF[adress_u_t] = K[adress_u_t,adress_u_t]*Unp1[adress_u_t]

                            # Extract RF on Dirichlet nodes
                            RF1_d = RF[numer[d,1]]
                            RF2_d = RF[numer[d,2]]
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

    function armijo(δu, Un, Unp1, Obj, R, s, n, coord, connec, dirichlet, numer, props, statevar)
            # Perform normal armijo
            t = 1.0
            δ = zeros(s)
            δ[1:s-n] = δu
            Ktemp, R_temp, obj_temp = assemble(Un, Unp1+δ*t, coord, connec, dirichlet, numer, props, statevar, true, false, false)
            it = 1
            while (obj_temp > Obj) & (it <= 20) #+ 0.0 * t * slope)) #& (obj_temp > 0.0001)
                    t *= 0.5
                    Ktemp, R_temp, obj_temp = assemble(Un, Unp1+δ*t, coord, connec, dirichlet, numer, props, statevar, true, false, false)
                    it += 1
            end
            if it == 21
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

            # #Prepare for int. point values export
            # coord_int = zeros(size(connec,1)*4,2)
            # strain_y = zeros(size(connec,1)*4,1)
            #
            # # For each element, get the coordinates of the 4 integration point
            # for i = 1:size(connec,1)
            #         coord_int[i*4-3:i*4,:] = getIntCoord(coord[connec[i,:],:])
            #         strain_y[i*4-3:i*4,1] = statevar[i,:,6]
            # end
            #
            # # Push virtual connectivity table for integration points
            # cells = MeshCell[]
            # for i = 1:size(connec,1)
            #         inds = Array{Int32}(undef, 4)
            #         inds = collect(i*4-3:i*4)
            #
            #         c = MeshCell(VTKCellTypes.VTK_POLY_VERTEX, inds)
            #         push!(cells, c)
            # end
            #
            # # Write int. point scalars to VTK file
            # name = string("IntPt_",timestep)
            # vtkfile = vtk_grid(name, coord_int', cells)
            # vtkfile["Strain_y", VTKPointData()] = strain_y
            # vtk_save(vtkfile)
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
