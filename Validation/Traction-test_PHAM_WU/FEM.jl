# This code relies on the implementation presented with two publications:
# [1] E. Martínez-Pañeda, A. Golahmar, and C. F. Niordson, “A phase field formulation for hydrogen assisted cracking,” Computer Methods in Applied Mechanics and Engineering, vol. 342, pp. 742–761, Dec. 2018, doi: 10.1016/j.cma.2018.07.021.
# [2] T. Heister, M. F. Wheeler, and T. Wick, “A primal-dual active set method and predictor-corrector mesh adaptivity for computing fracture propagation using a phase-field approach,” Computer Methods in Applied Mechanics and Engineering, vol. 290, pp. 466–495, Jun. 2015, doi: 10.1016/j.cma.2015.03.009.

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
    export load
    export ordering

    function matStiffness(E,ν)
            # Taken from [1]
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
            # Taken from [1]
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
            # Taken from [1]
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

    function decomposeStrain(strain,strainLin,derivative)
            # Taken from [2]
            # Calculates Strain decomposition and its derivatives (strain_lin) second tensors
            # derivative is a boolean
            λLin = [0.0, 0.0]

            trE = tr(strain)
            trELin = tr(strainLin)

            detE = det(strain)

            # Calculate eigenvalues
            if (abs(strain[1,2]) < 1e-10*abs(strain[1,1])) | (abs(strain[1,2]) < 1e-10*abs(strain[2,2]))
                    # Strain is almost diagonal
                    λ = [strain[1,1], strain[2,2]]
                    v1 = [1.0, 0.0]
                    v2 = [0.0, 1.0]
            else
                    discriminant = sqrt(trE^2/4.0-detE)
                    λ = [0.5*trE + discriminant, 0.5*trE - discriminant]
                    # Calculate eigenvectors
                    coeff = (λ[1]-strain[1,1])/strain[1,2]
                    v1 = 1.0/sqrt(1.0+(coeff)^2)*[1.0, coeff]
                    coeff = (λ[2]-strain[1,1])/strain[1,2]
                    v2 = 1.0/sqrt(1.0+(coeff)^2)*[1.0, coeff]
            end

            P = [v1 v2]
            ΛPlus = Diagonal(max.(λ,0.0))

            if !derivative
                    # Compute strain decomposition
                    strainPlus = P*ΛPlus*P'
                    strainMinus = strain - strainPlus

                    return strainPlus, strainMinus
            else
                    # Compute linearized eigenvalues
                    discriminant = sqrt(trE^2/4.0-detE)
                    coeff = 0.5/discriminant*(strainLin[1,2]*strain[2,1]+strain[1,2]*strainLin[2,1]+0.5*((strain[1,1]-strain[2,2])*(strainLin[1,1]-strainLin[2,2])))
                    λLin[1] = 0.5*trELin + coeff
                    λLin[2] = 0.5*trELin - coeff

                    term1_1 = -1.0/(1.0+((λ[1]-strain[1,1])/strain[1,2])^2)*1.0/(2*sqrt(1.0+((λ[1]-strain[1,1])/strain[1,2])^2))
                    term1_2 = -1.0/(1.0+((λ[2]-strain[1,1])/strain[1,2])^2)*1.0/(2*sqrt(1.0+((λ[2]-strain[1,1])/strain[1,2])^2))

                    term2_1 = 2.0*((λ[1]-strain[1,1])/strain[1,2])*((λLin[1]-strainLin[1,1])*strain[1,2]-(λ[1]-strain[1,1])*strainLin[1,2])/strain[1,2]^2
                    term2_2 = 2.0*((λ[2]-strain[1,1])/strain[1,2])*((λLin[2]-strainLin[1,1])*strain[1,2]-(λ[2]-strain[1,1])*strainLin[1,2])/strain[1,2]^2

                    coeff1_1 = term1_1*term2_1
                    coeff1_2 = term1_2*term2_2

                    norm1 = 1.0/sqrt(1.0+((λ[1]-strain[1,1])/strain[1,2])^2)
                    norm2 = 1.0/sqrt(1.0+((λ[2]-strain[1,1])/strain[1,2])^2)

                    coeff2_1 = [0.0, ((λLin[1]-strainLin[1,1])*strain[1,2]-(λ[1]-strain[1,1])*strainLin[1,2])/strain[1,2]^2]
                    coeff2_2 = [0.0, ((λLin[2]-strainLin[1,1])*strain[1,2]-(λ[2]-strain[1,1])*strainLin[1,2])/strain[1,2]^2]

                    v1Lin = coeff1_1*[1.0,(λ[1]-strain[1,1])/strain[1,2]]+norm1*coeff2_1
                    v2Lin = coeff1_2*[1.0,(λ[2]-strain[1,1])/strain[1,2]]+norm2*coeff2_2

                    # Very important: Set λLin to zero when the corresponding λ is set to zero.
                    if λ[1] < 0.0
                            λLin[1] = 0.0
                    end
                    if λ[2] < 0.0
                            λLin[2] = 0.0
                    end

                    PLin = [v1Lin v2Lin]
                    ΛPlusLin = Diagonal(λLin)

                    strainLinPlus = PLin*ΛPlus*P'+P*ΛPlusLin*P'+P*ΛPlus*PLin'

                    return strainLinPlus
            end
    end


    function elStiffness(props, jelem, un_el, unp1_el, coords, ndof, statevar, Objective, Residual, Hessian, energySplit)
            # Modified, from [1]
            # Takes in entry :
            # Material properties, element number, displacement / proposed displacement for element, coordinates of the mesh, number of DOF and state variables
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
            #C = matStiffness(E,ν)
            μ = E/(2.0*(1.0+ν))
            λ = E*ν/((1.0+ν)*(1.0-2.0*ν))

            # Penalty coefficient
            γ = Gc/lc*(27/64)/0.01^2

            # Sum on all integration point
            for kintk = 1:ninpt
                    dN, dNdz = shapefcn(kintk,nnode)
                    dNdx, djac = jacobian(jelem, ndim, nnode, coords, dNdz)

                    # Construct gradient for each component of vector-shape-function (See shape function of Deal.II here: https://www.dealii.org/current/doxygen/deal.II/step_8.html)
                    dΦdx_t = zeros(nnode*ndim,2,2)
                    dΦdx = zeros(nnode*ndim,4)
                    for i = 1:nnode*ndim
                            if rem(i,2) != 0
                                    dΦdx_t[i,1,:] = dNdx[:,cld(i,2)]'
                            else
                                    dΦdx_t[i,2,:] = dNdx[:,cld(i,2)]'
                            end
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

                    ∇u = zeros(2,2)
                    for i = 1:ninpt
                            ∇u += dNdx[:,i]*unp1_el[2*(i-1)+1:2*i]'
                    end

                    # Compute strain and stress from current time step
                    strain_t = 0.5*(transpose(∇u)+∇u)
                    trE = tr(strain_t)
                    if (norm(unp1_el[1:8]) != 0.0) & energySplit
                            trE_plus = max(trE,0.0)
                            # Decompose strain
                            strainPlus_t, strainMinus_t = decomposeStrain(strain_t,zeros(2,2),false)
                            # Compute decomposed stress
                            stressPlus_t = λ*trE_plus*I+2.0*μ*strainPlus_t
                            stressMinus_t = λ*(trE-trE_plus)*I+2.0*μ*strainMinus_t
                            # Compute strain energy density from the current increment
                            PsiPlus = 0.5*λ*trE_plus^2+μ*tr(strainPlus_t*strainPlus_t)
                            PsiMinus = 0.5*λ*(trE-trE_plus)^2+μ*tr(strainMinus_t*strainMinus_t)
                    else
                            stressPlus_t = λ*trE*I+2.0*μ*strain_t
                            stressMinus_t = zeros(2,2)
                            # Compute strain energy density from the current increment
                            PsiPlus = 0.5*λ*trE^2+μ*tr(strain_t*strain_t)
                            PsiMinus = 0.0
                    end

                    #Update statevar   !!! To use if stress and strain is desired at post-processing !!!
                    #statevar[kintk, 1:ntens] = stress_plus
                    #statevar[kintk, ntens+1:2*ntens] = strainPlus
                    #statevar[kintk, 2*ntens+1] = phi

                    # Compute Residual
                    if Residual
                            # Solid problem
                            @inbounds for i = 1:8
                                    # Assemble residual
                                    Re[i] += (((1-phi)^2+xk)*dot(stressPlus_t,dΦdx_t[i,:,:])+dot(stressMinus_t,dΦdx_t[i,:,:]))*djac
                            end
                            # Phase field problem
                            Re[9:12,1] += (0.75*Gc*lc*dNdx'*dNdx*unp1_el[9:12,1]+dN[:,1]*(0.375*Gc/lc+2.0*PsiPlus*phi-2.0*PsiPlus)+γ*min(phi-phin,0.0)*dN)*djac
                    end

                    # Compute Hessian
                    if Hessian
                            # Kinematic problem
                            @inbounds for i = 1:8
                                    # Assemble Hessian
                                    strainLin_t = 0.5*(transpose(dΦdx_t[i,:,:])+dΦdx_t[i,:,:])
                                    trELin = tr(strainLin_t)
                                    if (norm(unp1_el[1:8]) != 0.0) & energySplit
                                            # If displacement is applied and Energy Split is required
                                            if trE < 0.0
                                                    trEPlusLin = 0.0
                                            else
                                                    trEPlusLin = trELin
                                            end
                                            strainPlusLin_t = decomposeStrain(strain_t,strainLin_t,true)
                                            stressPlusLin_t = λ*trEPlusLin*I+2*μ*strainPlusLin_t
                                            stressMinusLin_t = λ*(trELin-trEPlusLin)*I+2*μ*(strainLin_t-strainPlusLin_t)
                                    else
                                            # If no displacement is applied, do not split strain (otherwise numerical problem)
                                            stressPlusLin_t = λ*trELin*I+2.0*μ*strainLin_t
                                            stressMinusLin_t = zeros(2,2)
                                    end

                                    for j = 1:8
                                            Ke[j,i] += (((1-phi)^2+xk)*dot(stressPlusLin_t,dΦdx_t[j,:,:])+dot(stressMinusLin_t,dΦdx_t[j,:,:]))*djac
                                    end
                                    Ke[9:12,i] += -2.0*(1-phi)*dN*dot(stressPlus_t,dΦdx_t[i,:,:])*djac
                                    Ke[i,9:12] += -2.0*(1-phi)*dN*dot(stressPlus_t,dΦdx_t[i,:,:])*djac
                            end
                            # Damage problem
                            Ke[9:12,9:12] += (0.75*dNdx'*dNdx*Gc*lc+2.0*dN*dN'*PsiPlus+γ*((phi-phin)<=0.0)*dN*dN')*djac
                    end

                    if Objective
                            Oe += (((1-phi)^2+xk)*PsiPlus+PsiMinus+0.375*Gc*(phi/lc + lc*(dNdx*unp1_el[9:12,1])'*(dNdx*unp1_el[9:12,1]))+0.5*γ*min(phi-phin,0.0)^2)*djac
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

    function assemble(Un, Unp1, coord, connec, dirichlet, numer, props, statevar, Objective, Residual, Hessian, energySplit)
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
                      Ke, Re, Oe, statevar[el,:,:] = elStiffness(props, el, un_el, unp1_el, coords, size(numer,2), statevar[el,:,:], Objective, Residual, Hessian, energySplit)
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

    function solveNEWTON(coord, connec, dirichlet, numer, props, timestep, statevar, cpus, energySplit, output, outputRF)
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
                    K, R, O = assemble(Un, Unp1, coord, connec, dirichlet, numer, props, statevar, Objective, Residual, Hessian, energySplit)
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
                            t = armijo(δu, Un, Unp1, O, R, s, n, coord, connec, dirichlet, numer, props, statevar, energySplit)
                            # Apply correction and recalculate Residual and Hessian
                            Unp1[1:s-n,1] += δu*t
                            call_inc += 1
                            K, R, O = assemble(Un, Unp1, coord, connec, dirichlet, numer, props, statevar, Objective, Residual, Hessian, energySplit)

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
                            K, R, O = assemble(Un, Unp1, coord, connec, dirichlet, numer, props, statevar, false, false, Hessian, energySplit)

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

    function armijo(δu, Un, Unp1, Obj, R, s, n, coord, connec, dirichlet, numer, props, statevar, energySplit)
            # Perform normal armijo
            t = 1.0
            δ = zeros(s)
            δ[1:s-n] = δu
            Ktemp, R_temp, obj_temp = assemble(Un, Unp1+δ*t, coord, connec, dirichlet, numer, props, statevar, true, false, false, energySplit)
            it = 1
            while (obj_temp > Obj) & (it <= 20) #+ 0.0 * t * slope)) #& (obj_temp > 0.0001)
                    t *= 0.5
                    Ktemp, R_temp, obj_temp = assemble(Un, Unp1+δ*t, coord, connec, dirichlet, numer, props, statevar, true, false, false, energySplit)
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
