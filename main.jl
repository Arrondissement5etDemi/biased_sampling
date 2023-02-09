include("largest_hole.jl")

function main1D()
    box1D = integer_lattice(100, 0.95)
    test_point = rand(box1D.dim())*box1D.l
    box1D.equilibrate_biased!(weight1, test_point; sweeps = 2000, displace = 0.1)
    hist = box1D.equilibrate_biased!(weight1, test_point; sweeps = 20000, displace = 0.1)
    pretty_print(hist)
    flush(stdout)
end

function weight2(x)
    if x < 0.5
        return 1.069
    elseif x < 1.5
        return -78.836*x^2 + 77.892*x - 18.168
    else
        return Inf
    end
end 

function main2D()
    box2D = triangleLatt418(ϕ2ρ2D(0.73))
    test_point = rand(box2D.dim())*box2D.l
    box2D.equilibrate_biased!(weight2, test_point; sweeps = 200, displace = 0.1, cells_per_side = 5)
    hist = box2D.equilibrate_biased!(weight2, test_point; sweeps = 2000, displace = 0.1, cells_per_side = 5)
    pretty_print(hist)
    flush(stdout)
end