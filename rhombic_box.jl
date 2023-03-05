using LinearAlgebra
```computes the distance between two particles in a parallelogram simulation box, whose boundaries are defined by 
the vectors in \$basis. 
The particles \$parti1 and \$parti2 are expresse as fractional multiples of the basis vectors```
function dist_pbc(parti1, parti2, basis) 
    vec_in_basis = parti1 - parti2
    return norm(basis*min.(vec_in_basis, 1 .- vec_in_basis))
end

mutable struct rhombic_box
    ###fields below###
    particles #expressed as fractional multiples of the basis vectors
    basis
    inv_basis
    test_point #expressed as fractional multiples of the basis vectors
    dist_vec #distances from each particle to the test point
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
    
    function rhombic_box(_particles, _basis, _test_point)
        this = new()
        this.particles = _particles
        this.basis = _basis
        this.inv_basis = inv(_basis)
        this.test_point = _test_point
        this.dist_vec = vec(mapslices(p -> dist_pbc(this.test_point, p, this.basis), this.particles; dims = 1))

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
                temperature = 1, sweeps = 100, displace = 0.1, hardcore = 1, λ = 0.1, evd = 3.9e-11)
            hist = zeros(200, 2)
            hist[:, 1] = 0.01:0.01:2.0
            for i = 1:sweeps
                for j = 1:this.n()
                    this.attempt_move!(j, displace)
                end
                this.attempt_move_test_point!(displace)
                if i%1000 == 500
                    this.attempt_remove!(λ, hardcore, weight_func, evd, temperature)
                elseif i%1000 == 0
                    this.attempt_add!(λ, hardcore, weight_func, evd, temperature)
                end
                if i % 20 == 0
                    nnd = minimum(dist_vec)
                    println(string(i)*" "*string(nnd)*" "*string(this.n()))
                    flush(stdout)
                    dist_test_bin = Int(ceil(minimum(dist_vec)/0.01))
                    hist[dist_test_bin, 2] += 1
                end
            end
            hist[:, 2] = hist[:, 2]/(sweeps/20)
            return hist, test_point
        end
        
        this.particle_energy = function(particle, hardcore)
            vec = vec(mapslices(x -> dist_pbc(particle, x, this.basis), this.particles; dims = 1))
            nnd = minimum(vec[vec .> 0])
            return ifelse(nnd < hardcore, 2*10^6 - 10^6*nnd/hardcore, 0)
        end

        ```moves the \$j-th particle, the displacement is still expressed in cartesian coordinates```
        this.move! = function(j, displace, displace_vec_cartesian = missing)
            if ismissing(displace_vec_cartesian)
                displace_vec_cartesian = (rand(this.dim())*2.0 .- 1)*displace
            end
            displace_vec_inbasis = this.inv_basis*displace_vec_cartesian
            this.particles[:, j] = mod.(this.particles[:, j] + displace_vec_inbasis, 1)
            this.dist_vec[j] = dist_pbc(this.test_point, pj, this.basis)
            return displace_vec_cartesian
        end

        this.attempt_move! = function(j, displace)
            old_biase_energy = weight_func(minimum(this.dist_vec))
            old_ej = this.particle_energy(this.particles[:, j], hardcore)
            displace_vec_cartesian = this.move!(j, displace)
            Δpe = this.particle_energy(this.particles[:, j], hardcore) - old_ej
            Δbiase = weight_func(minimum(this.dist_vec)) - old_biase_energy
            Δenergy = Δpe + Δbiase
            threshold = min(1, exp(-Δenergy/temperature))
            if rand() > threshold
                this.move!(j, displace, -displace_vec_cartesian)
            end
        end

        this.move_test_point! = function(displace, displace_vec_cartesian = missing)
            if ismissing(displace_vec_cartesian)
                displace_vec_cartesian = (rand(this.dim())*2.0 .- 1)*displace
            end
            displace_vec_inbasis = this.inv_basis*displace_vec_cartesian
            this.test_point = mod.(this.test_point + displace_vec_inbasis, 1)
            this.dist_vec = vec(mapslices(p -> dist_pbc(this.test_point, p, this.basis), this.particles; dims = 1))
            return displace_vec_cartesian
        end

        this.attempt_move_test_point! = function(displace)
            old_biase_energy = weight_func(minimum(this.dist_vec))
            displace_vec_cartesian = this.move_test_point!(j, displace)
            Δbiase = weight_func(minimum(this.dist_vec)) - old_biase_energy
            threshold = min(1, exp(-Δbiase/temperature))
            if rand() > threshold
                this.move_test_point!(displace, -displace_vec_cartesian)
            end
        end

        ```propose to remove the nearest neighbor from \$test_point if it is within distance \$λ\*\$D from the \$test_point```
        this.remove! = function(λ, hardcore)
            #get the index of the particle to be potentially removed
            min_dist, j = findmin(dist_vec)
            #do nothing if the minimum distance is larger than λD
            if min_dist > λ*hardcore
                return zeros(this.dim(), 0)
            end
            #remove the particle at index j from the particle array
            removed_particle = this.particles[:, j] #expressed in this.basis
            this.particles = dropcol(this.particles, j)
            #delete the jth entry in dist_vec
            deleteat!(this.dist_vec, j)
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
            push!(this.dist_vec, dist_pbc(this.test_point, new_particle, this.basis))
            #that's it. We won't care at the moment whether this addition will be rejected. 
            #that's not the business of this function. 
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

        return this
    end
end