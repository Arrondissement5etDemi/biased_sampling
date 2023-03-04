include("largest_hole.jl")

function weight1(x)
    if x < 0.5
        return -38.089*0.5 + 19.505
    elseif x < 2
        return -38.089*x + 19.505
    else
        return -38.089*2 + 19.505
    end
end

function main1D()
    #b = integer_lattice(200, 0.95)
    #test_point = b.l*rand(b.dim())
    #b.equilibrate_biased!(weight1, test_point, λ = 0.1, hardcore = 1, cells_per_side = 20, sweeps = 40000, evd = 1.93e-10)
    test_point, b = parse_config(ARGS[1], ARGS[2])
    b.equilibrate_biased!(weight1, test_point, λ = 0.1, hardcore = 1, cells_per_side = 20, sweeps = 40000, evd = 2.47e-10)
    hist = b.equilibrate_biased!(weight1, test_point, λ = 0.1, hardcore = 1, cells_per_side = 20, sweeps = 200000, evd = 2.47e-10)
    pretty_print(hist)
    println(" ")
    println(test_point)
    println(b.n())
    pretty_print(b.particles')
    flush(stdout)
end

function weight2(x)
    if x < 0.5
        return log(2*π*x*0.929465) - log(π*0.929465) - 20
    else
        return -40*x
    end
end

function read_bias()
    f = open("N=418/bias2.pot")
    result = zeros(99, 2)
    for i = 1:99
        result[i, :] = parse.(Float64, split(readline(f),"\t"))
    end
    return result
end

function weight3(x, bias)
    return 20*(x - 1)^2
    #=bin = Int(fld(x, 0.02)) + 1
    if x < 2
        return weight2(x) + bias[min(bin, 99), 2]
    else 
        return Inf 
    end=#
end

function parse_config(path, arg)
    f = open(path*string(arg)*".out", "r")
    for _ = 1:2202
        readline(f) #dummies
    end
    test_point = parse.(Float64, split(strip(readline(f), ['[', ']']), ", "))
    dim = length(test_point)
    readline(f)
    n = parse(Int, readline(f)[1:3]) #number of particles lol
    particles = zeros(dim, n)
    for i = 1:n
        particle_line = readline(f)
        particles[:, i] = reshape(parse.(Float64, split(particle_line)), dim, 1)
    end
    close(f)
    ρ = ϕ2ρ2D(0.73)
    l = √(418/ρ)
    return test_point, Box(particles, l)
end


function main2D()
    b = triangleLatt418(ϕ2ρ2D(0.73))
    test_point = rand(b.dim())*b.l
    b.equilibrate_biased!(weight2, test_point, λ = 0.1, hardcore = 1, cells_per_side = 5, sweeps = 1000, evd = 1.45e-9)
    hist = b.equilibrate_biased!(weight2, test_point, λ = 0.1, hardcore = 1, cells_per_side = 5, sweeps = 4000, evd = 1.45e-9)
    pretty_print(hist)
    println(" ")
    println(test_point)
    println(b.n())
    pretty_print(b.particles')
    flush(stdout)
end

function maiiin()
    test_point, b = parse_config(ARGS[1], ARGS[2])
    bias = read_bias()
    #test_point = b.equilibrate_biased!(weight2, test_point, λ = 0.1, hardcore = 1, cells_per_side = 5, sweeps = 10000, evd = 1.45e-9)[2]
    hist, test_point = b.equilibrate_biased!(weight2, test_point, λ = 0.8, hardcore = 1, cells_per_side = 5, sweeps = 40000, evd = 1.45e-9)
    pretty_print(hist)
    println(" ")
    println(test_point)
    println(b.n())
    pretty_print(b.particles')
    flush(stdout)
end

main2D()
