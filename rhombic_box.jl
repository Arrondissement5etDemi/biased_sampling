using LinearAlgebra, Plots, IterTools
include("helper_funcs.jl")

```computes the distance between two particles in a parallelogram simulation box, whose boundaries are defined by 
the vectors in \$basis. 
The particles \$parti1 and \$parti2 are expresse as fractional multiples of the basis vectors```
function dist_pbc(parti1, parti2, basis, basis_images) 
    vec_cartesian = basis*(parti2 - parti1)
    return minimum(broadcast(x -> norm(x + vec_cartesian), basis_images))
end

mutable struct rhombic_box
    ###fields below###
    particles #expressed as fractional multiples of the basis vectors
    basis
    inv_basis
    test_point #expressed as fractional multiples of the basis vectors
    dist_vec #distances from each particle to the test point
    basis_images #image vectors of the basis
    n_cells_per_side
    cell_list #cell list to accelerate MC simulations
    ###functions below###
    dim
    n
    ρ
    ρInv
    equilibrate_biased!
    particle_energy
    move!
    attempt_move!
    move_test_point!
    attempt_move_test_point!
    remove!
    attempt_remove!
    add!
    attempt_add!
    visualize
    
    function rhombic_box(_particles, _basis, _test_point, _n_cells_per_side = 20)
        this = new()
        this.particles = _particles
        this.basis = _basis
        this.inv_basis = inv(_basis)
        this.test_point = _test_point
        dim = size(this.particles, 1)
        this.basis_images = fill(zeros(dim), ntuple(i -> 3, dim))
        for i in product(ntuple(i -> -1:1, dim)...)
            i_vec = collect(i)
            this.basis_images[(i_vec .+ 2)...] = this.basis*i_vec
        end
        this.basis_images = vec(this.basis_images)
        this.dist_vec = vec(mapslices(p -> dist_pbc(this.test_point, p, this.basis, this.basis_images), this.particles; dims = 1))
        this.n_cells_per_side = _n_cells_per_side
        this.cell_list = fill(zeros(0), ntuple(i -> this.n_cells_per_side, dim)) #to store indices of particles in the cells
        for i in 1:size(this.particles, 2)
            cell_inds = Int.(floor.(this.particles[:, i]*this.n_cells_per_side) .+ 1)
            this.cell_list[cell_inds...] = vcat(this.cell_list[cell_inds...], [i])
        end
        
        this.dim = function()
            return size(this.particles, 1)
        end

        this.n = function()
            return size(this.particles, 2)
        end

        this.ρ = function()
            return this.n()/det(this.basis)
        end

        this.ρInv = function()
            return det(this.basis)/this.n()   
        end

        this.equilibrate_biased! = function(weight_func;
                temperature = 1, sweeps = 1000, displace = 0.01, hardcore = 1, λ = 0.1, evd = 3.61e-9)
            hist = zeros(400, 2)
            hist[:, 1] = 0.01:0.01:4.0
            for i = 1:sweeps
                for j = 1:this.n()
                    this.attempt_move!(j, displace, weight_func, hardcore, temperature)
                end
                this.attempt_move_test_point!(displace, weight_func, temperature)
                if i%1000 == 500
                    this.attempt_remove!(λ, hardcore, weight_func, evd, temperature)
                elseif i%1000 == 0
                    this.attempt_add!(λ, hardcore, weight_func, evd, temperature)
                end
                if i % 20 == 0
                    nnd = minimum(this.dist_vec)
                    println(string(i)*" "*string(nnd)*" "*string(this.n()))
                    flush(stdout)
                    dist_test_bin = min(Int(ceil(nnd/0.01)), size(hist, 1))
                    hist[dist_test_bin, 2] += 1
                end
            end
            hist[:, 2] = hist[:, 2]/(sweeps/20)
            return hist
        end
        
        #need to accelerate this using cell list.
        this.particle_energy = function(particle, hardcore)
            nCellsPerSide = size(this.cell_list, 1)
            central_inds = Int.(floor.(particle*this.n_cells_per_side) .+ 1)
            found = false
            surrounding_ind = 1
            all_dists = zeros(0)
            while !found
                #iterate over the cells surrounding the test particle
                for i in product(ntuple(i -> -surrounding_ind:surrounding_ind, this.dim())...)
                    current_cell_ind = mod.(central_inds + collect(i) .- 1, nCellsPerSide) .+ 1
                    current_neighbors = this.cell_list[current_cell_ind...]
                    if size(current_neighbors, 2) > 0
                        dists = broadcast(x -> dist_pbc(this.particles[:, Int(x)], particle, this.basis, this.basis_images), current_neighbors)
                        filter!(x -> x > 0, dists)
                        all_dists = vcat(all_dists, dists)
                        if length(all_dists) > 0
                            found = true
                        end
                    end
                end
                surrounding_ind += 1
            end
            nnd = minimum(all_dists)
            return ifelse(nnd < hardcore, 2*10^6 - 10^6*nnd/hardcore, 0)
        end

        ```moves the \$j-th particle, the displacement is expressed in cartesian coordinates```
        this.move! = function(j, displace, displace_vec = missing)
            if ismissing(displace_vec)
                displace_vec = (rand(this.dim())*2 .- 1)*displace
            end
            cell_inds_old = Int.(floor.(this.particles[:, j]*this.n_cells_per_side) .+ 1)
            this.particles[:, j] = mod.(this.particles[:, j] + displace_vec, 1) #move it
            #update dist_vec
            this.dist_vec[j] = dist_pbc(this.test_point, this.particles[:, j], this.basis, this.basis_images)
            #update cell list
            cell_inds_new = Int.(floor.(this.particles[:, j]*this.n_cells_per_side) .+ 1)
            if cell_inds_old ≠ cell_inds_new
                filter!(x -> Int(x) ≠ j, this.cell_list[cell_inds_old...])
                this.cell_list[cell_inds_new...] = vcat(this.cell_list[cell_inds_new...], [j])
            end
            return displace_vec
        end

        ```proposes to move the \$j-th particle```
        this.attempt_move! = function(j, displace, weight_func, hardcore, temperature)
            old_biase_energy = weight_func(minimum(this.dist_vec))
            old_ej = this.particle_energy(this.particles[:, j], hardcore)
            displace_vec = this.move!(j, displace)
            Δpe = this.particle_energy(this.particles[:, j], hardcore) - old_ej
            Δbiase = weight_func(minimum(this.dist_vec)) - old_biase_energy
            threshold = min(1, exp(-(Δpe + Δbiase)/temperature))
            if rand() > threshold
                this.move!(j, displace, -displace_vec)
            end
        end

        this.move_test_point! = function(displace, displace_vec = missing)
            if ismissing(displace_vec)
                displace_vec = (rand(this.dim())*2.0 .- 1)*displace
            end
            this.test_point = mod.(this.test_point + displace_vec, 1)
            this.dist_vec = vec(mapslices(p -> dist_pbc(this.test_point, p, this.basis, this.basis_images), this.particles; dims = 1))
            return displace_vec
        end

        ```proposes to move the test point```
        this.attempt_move_test_point! = function(displace, weight_func, temperature)
            old_biase_energy = weight_func(minimum(this.dist_vec))
            displace_vec = this.move_test_point!(displace)
            Δbiase = weight_func(minimum(this.dist_vec)) - old_biase_energy
            threshold = min(1, exp(-Δbiase/temperature))
            if rand() > threshold
                this.move_test_point!(displace, -displace_vec)
            end
        end

        ```propose to remove the nearest neighbor from \$test_point if it is within distance \$λ\*\$D from the \$test_point```
        this.remove! = function(λ, hardcore)
            #get the index of the particle to be potentially removed
            min_dist, j = findmin(this.dist_vec)
            #do nothing if the minimum distance is larger than λD
            if min_dist > λ*hardcore
                return zeros(this.dim(), 0)
            end
            #remove the particle at index j from the particle array
            removed_particle = this.particles[:, j] #expressed in this.basis
            this.particles = dropcol(this.particles, j)
            #delete the jth entry in dist_vec
            deleteat!(this.dist_vec, j)
            #update the cell list
            for cell in this.cell_list
                filter!(x -> Int(x) ≠ j, cell)
                for ii in eachindex(cell)
                    if cell[ii] > j
                        cell[ii] -= 1
                    end
                end
            end
            return removed_particle
        end
        
        ```proposes and accepts or rejects a remove```
        this.attempt_remove! = function(λ, hardcore, weight_func, evd, temperature)
            old_biase_energy = weight_func(minimum(this.dist_vec))
            removed_particle = this.remove!(λ, hardcore)
            #do nothing if no particle is within the central sphere
            if size(removed_particle, 2) != 0
                Δpe = -this.particle_energy(removed_particle, hardcore)
                Δbiase = weight_func(minimum(this.dist_vec)) - old_biase_energy
                Δenergy = Δpe + Δbiase
                threshold = min(1, evd/(this.ρ()*v1(λ*hardcore, this.dim()))*exp(-Δenergy/temperature))
                if rand() > threshold
                    this.add!(λ, hardcore, removed_particle)
                end
            end
        end

        ```proposes to add a particle in the ball with distance \$λ\*\$hardcore centered at \$test_point
        then modifies \$this.particles, \$cl and \$dist_vec accordingly.```
        this.add! = function(λ, hardcore, new_particle = missing)
            #generate a new actual particle if it's not already given
            if ismissing(new_particle)
                new_particle = mod.(
                    this.inv_basis*rand_pt_in_sphere(λ*hardcore, this.dim(), this.basis*this.test_point), 1)
            end
            #add the new particle to the list of particles in the box. note that it's added as the last particle
            this.particles = hcat(this.particles, new_particle)
            #update the dist_vec
            push!(this.dist_vec, dist_pbc(this.test_point, new_particle, this.basis, this.basis_images))
            #update the cell list
            cell_inds = Int.(floor.(new_particle*this.n_cells_per_side) .+ 1)
            this.cell_list[cell_inds...] = vcat(this.cell_list[cell_inds...], [this.n()])
            return new_particle
        end

        ```proposes and accepts or rejects an add```
        this.attempt_add! = function(λ, hardcore, weight_func, evd, temperature)
            old_biase_energy = weight_func(minimum(this.dist_vec))
            new_particle = this.add!(λ, hardcore)
            Δpe = this.particle_energy(new_particle, hardcore)
            Δbiase = weight_func(minimum(this.dist_vec)) - old_biase_energy
            Δenergy = Δpe + Δbiase
            threshold = min(1, v1(λ*hardcore, this.dim())*this.ρ()/evd*exp(-Δenergy/temperature))
            if rand() > threshold
                this.remove!(λ, hardcore)
            end
        end

        ```plots the box```
        this.visualize = function()
            particles_cartesian = this.basis*this.particles
            test_point_cartesian = this.basis*this.test_point
            if this.dim() == 2
                e1 = this.basis[:, 1]
                e2 = this.basis[:, 2]
                box_corners = hcat(zeros(2), e1, e1 + e2, e2, zeros(2))
                plot(box_corners[1, :], box_corners[2, :])
                plot!(particles_cartesian[1, :], particles_cartesian[2, :], seriestype = scatter)
                return plot!([test_point_cartesian[1]], [test_point_cartesian[2]], seriestype = scatter, color = "green")
            elseif this.dim() == 1
                plot(particles_cartesian, zeros(this.n()), seriestype = scatter)
                return plot!([test_point_cartesian[1]], [test_point_cartesian[2]], seriestype = scatter, color = "green")
            else
                return nothing
            end
        end
        return this
    end
end

function random_box(dim, n, basis)
    particles = rand(dim, n)
    test_point = rand(dim)
    return rhombic_box(particles, basis, test_point)
end

```creates a triangle lattice in the rhombic simulation box with density \$ρ, and \$n_per_side^2 particles in total ```
function triangle_latt(n_per_side, ρ, test_point = missing)
    n = n_per_side^2
    l = √(2*(n/ρ)/√3) #box side length
    basis = l*[1 1/2; 0 √3/2]
    particles = zeros(2, n)
    for i = 1:n_per_side, j = 1:n_per_side
        particles[:, (i - 1)*n_per_side + j] = [i - 1, j - 1]/n_per_side
    end
    if ismissing(test_point)
        test_point = rand(2)
    end
    return rhombic_box(particles, basis, test_point)
end

b = triangle_latt(20, ϕ2ρ2D(0.73), rand(2))
weight_func(x) = -20x
hist = b.equilibrate_biased!(weight_func; displace = 0.1/√(b.n()), sweeps = 10000)
pl = b.visualize()
display(pl)