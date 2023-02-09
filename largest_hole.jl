using Distributed
@everywhere using LinearAlgebra, IterTools, SharedArrays, Optim
@everywhere include("helper_funcs.jl")

```Distance in periodic boundary conditions```
@everywhere function dist_pbc(parti1, parti2, l)
    d = size(parti1)[1] #dimensionality
    dist_vec = zeros(d)
    for i = 1:d
        central = parti1[i] - parti2[i]
        dist_vec[i] = minimum(abs.([central, central - l, central + l]))
    end
    return norm(dist_vec)
end

@everywhere mutable struct Box
    particles::Array{Float64, 2}
    l::Float64
    dim
    n
    ρ
    ρInv
    equilibrate_biased!
    move!
    
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
        this.equilibrate_biased! = function(weight_func, test_point; temperature = 1, sweeps = 100, displace = 0.2, hardcore = 1, cells_per_side = 10)
            dist_vec = zeros(this.n())
            for j = 1:this.n()
                dist_vec[j] = dist_pbc(test_point, this.particles[:, j], this.l)
            end
            cl = cell_list(this, cells_per_side)
            cl_update_rate = Int(fld(this.l/cells_per_side, displace*2*sqrt(this.dim())))
            tot_pe = total_potential_energy(this, cl, hardcore)
            energy = biased_energy(tot_pe, weight_func, dist_vec)
            hist = zeros(100, 2)
            hist[:, 1] = 0.02:0.02:2.0
            for i = 1:sweeps
                if i % cl_update_rate == 1
                    cl = cell_list(this, cells_per_side)
                end
                for j = 1:this.n()
                    old_coord = deepcopy(this.particles[:, j])
                    old_energy = energy
                    old_dist_vec = deepcopy(dist_vec)
                    old_pe = tot_pe
                    old_ej = particle_energy(old_coord, this, cl, hardcore)
                    this.move!(j, displace, test_point, dist_vec)
                    new_ej = particle_energy(this.particles[:, j], this, cl, hardcore)
                    tot_pe = tot_pe - old_ej + new_ej
                    energy = biased_energy(tot_pe, weight_func, dist_vec)
                    if !boltzmann_accept(old_energy, energy, temperature)
                        this.particles[:, j] = old_coord
                        tot_pe = old_pe
                        energy = old_energy
                        dist_vec = old_dist_vec 
                    end
                end
                if i % 20 == 1
                    println(string(i)*" "*string(hole_size_Ev(this, test_point)))
                    flush(stdout)
                    dist_test_bin = Int(fld(minimum(dist_vec), 0.02) + 1)
                    hist[dist_test_bin, 2] += 1 
                end
            end
            return hist
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
    result = fill(zeros(1, 0), ntuple(i -> nCellsPerSide, dim)) #coordinates of particles in the cells
    cell_length = box.l/nCellsPerSide
    for i in 1:box.n()
        cell_inds = Int.(fld.(box.particles[:, i], cell_length) .+ 1)
        result[cell_inds...] = hcat(result[cell_inds...], [i])
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
                dists = mapslices(x -> dist_pbc(box.particles[:, Int(x[1])], test_particle, l), current_neighbors; dims = 1)
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
    f(test_point) = -nearest_neighbor(test_point, box, cl)
    opt = optimize(f, test_point, BFGS())
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

@everywhere function biased_energy(total_potential_energy, weight_func, dist_vec) 
    return weight_func(minimum(dist_vec)) + total_potential_energy
end

#trial weight functions here###
function weight1(x)
    if x < 0.5
        return -38.089*0.5 + 19.505
    elseif x < 1.5
        return -38.089*x + 19.505
    else
        return Inf
    end
end

function parse_config(path, arg)
    f = open(path*string(arg)*".out", "r")
    for i = 1:4103
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

function maiiin()
    test_point, box1D = parse_config(ARGS[1], ARGS[2])
    #integer_lattice(1000, 0.95)
    #test_point = rand(box1D.dim())*box1D.l
    #box1D.equilibrate_biased!(weight1, test_point; sweeps = 3000, displace = 0.1, cells_per_side = 200)
    hist = box1D.equilibrate_biased!(weight1, test_point; sweeps = 80000, displace = 0.1, cells_per_side = 200)
    pretty_print(hist)
    println(test_point)
    pretty_print(box1D.particles)
    flush(stdout)
end
