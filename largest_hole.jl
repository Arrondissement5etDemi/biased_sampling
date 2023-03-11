using Distributed
@everywhere using LinearAlgebra, IterTools, SharedArrays, Optim
@everywhere include("helper_funcs.jl")

```Distance in periodic boundary conditions```
@everywhere dist_pbc(parti1, parti2, l) = norm(-abs.(mod.(parti1 - parti2, l) .- l/2) .+ l/2)

@everywhere mutable struct Box
    particles::Array{Float64, 2}
    l::Float64
    dim
    n
    ρ
    ρInv
    equilibrate_biased!
    move!
    move_test_point!
    remove!
    add!
    attempt_remove!
    attempt_add!
    
    function Box(_particles, _l)
        this = new()
        this.particles = _particles
        this.l = _l
        this.dim = function()
            return size(this.particles)[1]
        end
        this.n = function()
            return size(this.particles)[2]
        end
        this.ρ = function()
            return this.n()/this.l^this.dim()
        end
        this.ρInv = function()
            return this.l^this.dim()/this.n()   
        end

        this.equilibrate_biased! = function(weight_func, test_point;
             temperature = 1, sweeps = 100, displace = 0.1, hardcore = 1, cells_per_side = 10, λ = 0.1, evd = 3.9e-11)
            dist_vec = vec(mapslices(x -> dist_pbc(test_point, x, this.l), this.particles; dims = 1))
            cl = cell_list(this, cells_per_side)
            cl_update_rate = Int(fld(this.l/cells_per_side, displace*2*sqrt(this.dim())))
            hist = zeros(200, 2)
            hist[:, 1] = 0.01:0.01:2.0
            for i = 1:sweeps
                if i % cl_update_rate == 1
                    cl = cell_list(this, cells_per_side)
                end
                for j = 1:this.n() + 1
                    if j <= this.n() #move the actual particles
                        old_coord = deepcopy(this.particles[:, j])
                        old_dist_vec = deepcopy(dist_vec)
                        old_ej = particle_energy(old_coord, this, cl, hardcore)
                        old_bias = weight_func(minimum(dist_vec))
                        this.move!(j, displace, test_point, dist_vec)
                        new_ej = particle_energy(this.particles[:, j], this, cl, hardcore)
                        new_bias = weight_func(minimum(dist_vec))
                        Δpe = new_ej - old_ej
                        Δbias = new_bias - old_bias
                        if !boltzmann_accept(0, Δpe + Δbias, temperature) #can the new energy be infinity and still be accepted???
                            this.particles[:, j] = old_coord
                            dist_vec = deepcopy(old_dist_vec) 
                        end
                    else #move the test particle
                        old_tp = deepcopy(test_point)
                        old_bias = weight_func(minimum(dist_vec))
                        old_dist_vec = deepcopy(dist_vec)
                        test_point = this.move_test_point!(displace, test_point, dist_vec)
                        new_bias = weight_func(minimum(dist_vec))
                        if !boltzmann_accept(old_bias, new_bias, temperature)
                            test_point = deepcopy(old_tp)
                            dist_vec = deepcopy(old_dist_vec) 
                        end
                    end
                end
                if i%200 == 100
                    this.attempt_remove!(λ, test_point, hardcore, cl, dist_vec, weight_func, evd, temperature)
                elseif i%200 == 0
                    this.attempt_add!(λ, test_point, hardcore, cl, dist_vec, weight_func, evd, temperature)
                end
                if i % 20 == 0
                    nnd = minimum(dist_vec)
                    println(string(i)*" "*string(nnd)*" "*string(this.n()))
                    flush(stdout)
                    if nnd > 0.81 && nnd < 0.87
                        println(test_point)
                        println(this.particles')
                        break
                    end
                    #=dist_test_bin = Int(fld(minimum(dist_vec), 0.01) + 1)
                    hist[dist_test_bin, 2] += 1 =#
                end
            end
            hist[:, 2] = hist[:, 2]/(sweeps/20)
            return hist, test_point
        end

        this.move! = function(j, displace, test_point, dist_vec)
            #move particle at j
            displaceVec = (rand(this.dim())*2.0 .- 1)*displace
            pj = this.particles[:, j]
            pj = mod.(pj + displaceVec, this.l)
            this.particles[:, j] = pj
            #update the vector of distances from pj to the test point
            dist_vec[j] = dist_pbc(test_point, pj, this.l)
        end

        this.move_test_point! = function(displace, test_point, dist_vec)
            displaceVec = (rand(this.dim())*2.0 .- 1)*displace
            new_test_point = mod.(test_point + displaceVec, this.l)
            for j = 1:this.n()
                dist_vec[j] = dist_pbc(new_test_point, this.particles[:, j], this.l)
            end
            return new_test_point
        end

        ```propose to remove the nearest neighbor from \$test_point if it is within distance \$λ\*\$D from the \$test_point```
        this.remove! = function(λ, hardcore, cl, dist_vec)
            #get the index of the particle to be potentially removed
            min_dist, j = findmin(dist_vec)
            #do nothing if the minimum distance is larger than λD
            if min_dist > λ*hardcore
                return zeros(this.dim(), 0)
            end
            #remove the particle at index j from the particle array
            removed_particle = this.particles[:, j]
            this.particles = dropcol(this.particles, j)
            #delete the jth entry in dist_vec
            deleteat!(dist_vec, j)
            #remove the particle from the cell list and update the indices of every remaining particle
            for i in eachindex(cl)
                cl[i] = filter(x -> Int(x) ≠ j, cl[i])
                for ii in eachindex(cl[i])
                    if cl[i][ii] > j
                        cl[i][ii] -= 1
                    end
                end
            end
            return removed_particle
        end

        ```proposes to add a particle in the ball with distance \$λ\*\$hardcore centered at \$test_point
        then modifies \$this.particles, \$cl and \$dist_vec accordingly.```
        this.add! = function(λ, test_point, hardcore, cl, dist_vec, new_particle = missing)
            #generate a new actual particle if it's not already given
            if ismissing(new_particle)
                new_particle = mod.(rand_pt_in_sphere(λ*hardcore, this.dim(), test_point), this.l)
            end
            #add the new particle to the list of particles in the box. note that it's added as the last particle
            this.particles = hcat(this.particles, new_particle)
            #update the cell list
            cell_length = this.l/size(cl, 1)
            cell_inds = Int.(fld.(new_particle, cell_length) .+ 1)
            cl[cell_inds...] = vcat(cl[cell_inds...], [this.n()])
            #update the dist_vec
            push!(dist_vec, dist_pbc(test_point, new_particle, this.l))
            #that's it. We won't care at the moment whether this addition will be rejected. 
            #that's not the business of this function. 
            return new_particle
        end

        ```proposes and accepts or rejects a remove```
        this.attempt_remove! = function(λ, test_point, hardcore, cl, dist_vec, weight_func, evd, temperature)
            old_biase_energy = weight_func(minimum(dist_vec))
            removed_particle = this.remove!(λ, hardcore, cl, dist_vec)
            #do nothing if no particle is within the central sphere
            if size(removed_particle, 2) == 0
                return
            end
            Δpe = -particle_energy(removed_particle, this, cl, hardcore)
            Δbiase = weight_func(minimum(dist_vec)) - old_biase_energy
            Δenergy = Δpe + Δbiase
            threshold = min(1, evd/(this.ρ()*v1(λ*hardcore, this.dim()))*exp(-Δenergy/temperature))
            if rand() > threshold
                this.add!(λ, test_point, hardcore, cl, dist_vec, removed_particle)
            end
        end

        ```proposes and accepts or rejects an add```
        this.attempt_add! = function(λ, test_point, hardcore, cl, dist_vec, weight_func, evd, temperature)
            old_biase_energy = weight_func(minimum(dist_vec))
            new_particle = this.add!(λ, test_point, hardcore, cl, dist_vec)
            Δpe = particle_energy(new_particle, this, cl, hardcore)
            Δbiase = weight_func(minimum(dist_vec)) - old_biase_energy
            Δenergy = Δpe + Δbiase
            threshold = min(1, v1(λ*hardcore, this.dim())*this.ρ()/evd*exp(-Δenergy/temperature))
            if rand() > threshold
                this.remove!(λ, hardcore, cl, dist_vec)
            end
        end
        return this
    end
end

function random_box(dim, n, l)
    particles = rand(dim, n)*l
    return Box(particles, l)
end

function fcc(n, ρ)
    cells_per_side = cbrt(n/4)
    if mod(cells_per_side, 1) != 0
        throw(ArgumentError("n in function fcc is incorrect."))
    else
        cells_per_side = Int(cells_per_side)
        l = (n/ρ)^(1/3)
        lattice_const = l/cells_per_side
        particles = zeros(3, n)
        for i = 1:n ÷ 4
            particles[1, i] = (i - 1) ÷ cells_per_side^2
            particles[2, i] = ((i - 1) % cells_per_side^2) ÷ cells_per_side
            particles[3, i] = ((i - 1) % cells_per_side^2) % cells_per_side
        end
        for i = n ÷ 4 + 1: n ÷ 2
            particles[:, i] = particles[:, i - n ÷ 4] + [1/2, 1/2, 0]
        end
        for i = n ÷ 2 + 1: 3n ÷ 4
            particles[:, i] = particles[:, i - n ÷ 2] + [1/2, 0, 1/2]
        end
        for i = 3n ÷ 4 + 1: n
            particles[:, i] = particles[:, i - 3n ÷ 4] + [0, 1/2, 1/2]
        end
        particles *= lattice_const
        return Box(particles, l)
    end
end

function triangleLatt418(rho)
   particles = zeros(2, 418)
   l = √(418/rho)
   distX = l/19;
   distY = l/22;
   for i = 1:19
       particles[:, i] = [(i - 1)*distX, 0]
   end
   for i = 20:38
       particles[:, i] = [0.5*distX + particles[1, i - 19], distY]
   end
   smart_modulo(x, y) = ifelse(x%y != 0, x%y, y)
   for i = 39:418
       particles[:, i] = [particles[1, smart_modulo(i, 38)], distY*fld(i - 1, 19)]
   end
   return Box(particles, l);
end

function triangleLatt780(rho)
    n, p, q = 780, 26, 30
    particles = zeros(2, n)
    l = √(n/rho)
    distX, distY = l/p, l/q
    for i = 1:p
        particles[:, i] = [(i - 1)*distX, 0]
    end
    for i = p + 1:2*p
        particles[:, i] = [0.5*distX + particles[1, i - p], distY]
    end
    smart_modulo(x, y) = ifelse(x%y != 0, x%y, y)
    for i = 2*p + 1:n
        particles[:, i] = [particles[1, smart_modulo(i, 2*p)], distY*fld(i - 1, p)]
    end
    return Box(particles, l);
 end

function integer_lattice(n, ρ)
    l = n/ρ
    lattice_const = 1/ρ
    particles = [(i - 1) for i = 1:n]'*lattice_const
    return Box(particles, l)
end

@everywhere function largest_hole_Ep(box, dm = missing)
    if ismissing(dm)
        dm = dist_mat(box)
    end
    return maximum(mapslices(v -> minimum(v[v .> 0]), dm; dims = 1))
end

@everywhere function hole_size_Ev(box, test_point, dist_vec = missing)
    if ismissing(dist_vec)
        dist_vec = mapslices(x -> dist_pbc(x, test_point, box.l), box.particles; dims = 1)
    end
    return minimum(dist_vec)
end

@everywhere function dist_mat(box)
    n = box.n()
    particles = box.particles
    l = box.l
    result = SharedArray{Float64}(n, n)
    @sync @distributed for i = 1:n
        @sync for j = 1:n
            if i < j
                result[i, j] = dist_pbc(particles[:, i], particles[:, j], l)
            elseif i == j
                result[i, j] = 0
            else
                result[i, j] = result[j, i]
            end
        end
    end
    return result
end

@everywhere function cell_list(box, nCellsPerSide)
    dim = box.dim()
    result = fill(zeros(0), ntuple(i -> nCellsPerSide, dim)) #to store indices of particles in the cells
    cell_length = box.l/nCellsPerSide
    for i in 1:box.n()
        cell_inds = Int.(fld.(box.particles[:, i], cell_length) .+ 1)
        result[cell_inds...] = vcat(result[cell_inds...], [i])
    end
    return result
end

@everywhere function nearest_neighbor(test_particle, box, cl)
    l = box.l
    nCellsPerSide = size(cl, 1)
    cell_length = l/nCellsPerSide
    test_cell_inds = Int.(fld.(test_particle, cell_length) .+ 1)
    result_dist = l
    found = false
    surrounding_ind = 1
    while !found
        #iterate over the cells surrounding the test particle
        for i in product(ntuple(i -> -surrounding_ind:surrounding_ind, box.dim())...)
            current_cell_ind = mod.(test_cell_inds + collect(i) .- 1, nCellsPerSide) .+ 1
            current_neighbors = cl[current_cell_ind...]
            if size(current_neighbors, 2) > 0
                dists = broadcast(x -> dist_pbc(box.particles[:, Int(x)], test_particle, l), current_neighbors)
                dists = dists[dists .> 0]
                if length(dists) > 0
                    dist_temp = minimum(dists)
                else
                    dist_temp = result_dist
                end
                if dist_temp < result_dist
                    found = true
                    result_dist = dist_temp
                end
            end
        end
        surrounding_ind += 1
    end
    return result_dist
end

```the local maximum of nearest neighbor distance, given a test point as the initial guess
or: the radius of the largest hole that contains the test point```
@everywhere function nnd_localmax(test_point, box, cl)
    opt = optimize(test_p -> -nearest_neighbor(test_p, box, cl), test_point, GradientDescent(), Optim.Options(iterations = 5))
    #println(Optim.minimizer(opt))
    return -Optim.minimum(opt)
end

@everywhere function particle_energy(particle, box, cl, hardcore)
    nnd = nearest_neighbor(particle, box, cl)
    if nnd < hardcore
        return 2*10^6 - 10^6*nnd/hardcore
    else
        return 0.0
    end
end

@everywhere function total_potential_energy(box, cl, hardcore)
    return sum(mapslices(p -> particle_energy(p, box, cl, hardcore), box.particles; dims = 1))
end

@everywhere function biased_energy(total_potential_energy, weight_func, dist_vec; nsmallest = 1)
    smallest_avr = mean(partialsort(dist_vec, 1:nsmallest)) 
    return weight_func(smallest_avr) + total_potential_energy
end

@everywhere function biased_energy2(total_potential_energy, weight_func, test_point, box, cl)
    return weight_func(nnd_localmax(test_point, box, cl)) + total_potential_energy
end

function parse_config(path, arg)
    f = open(path*string(arg)*".out", "r")
    for _ = 1:4103
        readline(f) #dummies
    end
    test_point = [parse(Float64, chop(readline(f); head=1, tail=1))]
    readline(f) #dummy
    config_line1 = readline(f)
    config_line2 = readline(f)
    config_line = config_line1*config_line2
    particles = reshape(parse.(Float64, split(config_line)), 1, 1000)
    println(particles)
    close(f)
    return test_point, Box(particles, 1000/0.95)
end

