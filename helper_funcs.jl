using LinearAlgebra
using Optim
using BenchmarkTools

function pretty_print(array)
    show(IOContext(stdout, :compact=>false), "text/plain", array)
    println()
end

function bfgs_2(f, params, delta, stepsize, other_args_for_f...)
    n_params = length(params)
    grad = zeros(n_params)
    grad_new = gradient(f, params, delta, other_args_for_f...) 
    h = I(n_params)
    gradnorm = norm(grad_new)
    round = 0
    params_old = deepcopy(params)
    params_new = zeros(n_params)
    while gradnorm > 0.001 && round <= 100
        round += 1
        grad = deepcopy(grad_new)
        direction = h*grad
        dx = zeros(n_params)
        line_search_count = 1
        e_old = f(params_old, other_args_for_f...)
        e_new = e_old
        alphaK = 0.0
        while line_search_count < 100
            params_new = params_old - stepsize*direction
            e_new = f(params_new, other_args_for_f...)
            if e_new < e_old 
                alphaK += stepsize
                e_old, params_old = e_new, params_new
                line_search_count += 1
            else
                params_new = params_old
                if line_search_count > 1
                    break
                elseif stepsize > 1E-8
                    stepsize /= 1.5
                else
                    h = 0.1I(n_params)
                    break
                end
            end
        end
        dx = -alphaK*direction
        if norm(dx) < 0.05
            h = I(n_params)
            continue
        end
        if line_search_count > 10
            stepsize *= 2
        end
        grad_new = gradient(f, params_new, delta, other_args_for_f...)
        gradnorm = norm(grad_new)
        if gradnorm <= 0.001
            break
        end
        y = grad_new - grad
        dx_y = dx⋅y
        y_h_y = y⋅(h*y)
        dxT_dx = dx*dx'
        second_term = (dx_y + y_h_y)/(dx_y*dx_y)*dxT_dx
        h_y_dxT = (h*y)*dx'
        third_term = (h_y_dxT + h_y_dxT')/dx_y
        h += second_term - third_term
    end
    return (params_new, f(params_new, other_args_for_f...))
end

function gradient(f, params, delta, other_args_for_f...) 
    n_params = length(params)
    result = zeros(n_params)
    up = zeros(n_params)
    down = zeros(n_params)
    for i = 1:n_params
        for j = 1:n_params
            if j != i 
                up[j] = params[j]
                down[j] = params[j]
            else 
                up[j] = params[j] + delta/2
                down[j] = params[j] - delta/2
            end
        end
        result[i] = (f(up, other_args_for_f...) -  f(down, other_args_for_f...))/delta;
    end
    return result
end

function test_opt()
    f(x, y) = x[1]^2*y
    @time bfgs_2(f, [100], 0.01, 0.2, 2)
    return bfgs_2(f, [100], 0.01, 0.2, 2)
end

function norm_sq(vec)
    return vec⋅vec
end

function boltzmann_accept(oldE::Real, newE::Real, temperature::Real)::Bool
	if newE < oldE 
		return true
	end
    if isinf(newE) 
        if isinf(oldE)
            return true
        end
		return false
	end
	rand_num = rand()
    prob = exp(-(newE - oldE)/temperature)
    if rand_num <= prob
        return true
    end 
	return false
end

function ϕ2ρ3D(ϕ, diam = 1)
    return 6*ϕ/(π*diam^3)
end

function ϕ2ρ2D(ϕ, diam = 1)
    return 4*ϕ/(π*diam^2)
end

function ϕ2ρ1D(ϕ, diam = 1)
    return ϕ/diam
end

test_opt()
