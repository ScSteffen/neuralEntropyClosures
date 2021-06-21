using LinearAlgebra
using KitBase, KitBase.FastGaussQuadrature, KitBase.ProgressMeter, KitBase.JLD2, KitBase.Plots, KitBase.PyCall
using KitML, KitML.Flux, KitML.DiffEqFlux, KitML.Optim, KitML.CSV

function flux_wall!(
    ff::T1,
    f::T2,
    u::T3,
    dt,
    rot = 1,
) where {
    T1<:AbstractVector{<:AbstractFloat},
    T2<:AbstractVector{<:AbstractFloat},
    T3<:AbstractVector{<:AbstractFloat},
}
    δ = heaviside.(u .* rot)
    fWall = 0.5 .* δ .+ f .* (1.0 .- δ)
    @. ff = u * fWall * dt

    return nothing
end

begin
    # setup
    set = Setup("radiation", "linesource", "1d1f1v", "kfvs", "bgk", 1, 2, "vanleer", "extra", 0.5, 0.7)

    # physical space
    x0 = 0
    x1 = 1
    nx = 100
    nxg = 0
    ps = PSpace1D(x0, x1, nx, nxg)

    # velocity space
    nu = 28
    points, weights = gausslegendre(nu)
    vs = VSpace1D(points[1], points[end], nu, points, ones(nu) .* (points[end] - points[1]) / (nu - 1), weights)

    # material
    σs = ones(Float32, nx)
    σa = zeros(Float32, nx)
    σt = σs + σa
    σq = zeros(Float32, nx)

    # moments
    L = 1
    ne = 2
    m = eval_sphermonomial(points, L)

    # time
    dt = set.cfl * ps.dx[1]
    nt = set.maxTime / dt |> floor |> Int
    global t = 0.0

    # solution
    f0 = 0.0001 * ones(nu)
    phi = zeros(ne, nx)
    for i = 1:nx
        phi[:, i] .= m * f0
    end
    α = zeros(Float32, ne, nx)

    # NN
    αT = zeros(Float32, nx, ne)
    phiT = zeros(Float32, nx, ne)
    phi_old = zeros(Float32, ne, nx)
    phi_temp = deepcopy(phi_old)
    
    opt = optimize_closure(zeros(Float32, 2), m, weights, phi[:, 1], KitBase.maxwell_boltzmann_dual)
    
    global X = zeros(Float32, ne, 1)
    X[:, 1] .= phi[:, 1]
    global Y = zeros(Float32, 1, 1) # h
    Y[1, 1] = kinetic_entropy(opt.minimizer, m, weights)
    #global Y = zeros(Float32, ne, 1) # α
    #Y[:, 1] = opt.minimizer

    cd(@__DIR__)
end

begin
    # initial condition
    f0 = 0.0001 * ones(nu)
    phi = zeros(ne, nx)
    for i = 1:nx
        phi[:, i] .= m * f0
    end
    α = zeros(Float32, ne, nx)
    flux = zeros(Float32, ne, nx + 1)
    fη = zeros(nu)
end

for iter = 1:nt
    println("iteration $iter of $nt")

    # mathematical optimizer
    @inbounds for i = 1:nx
        opt = KitBase.optimize_closure(α[:, i], m, weights, phi[:, i], KitBase.maxwell_boltzmann_dual)
        α[:, i] .= opt.minimizer
        phi[:, i] .= KitBase.realizable_reconstruct(opt.minimizer, m, weights, KitBase.maxwell_boltzmann_dual_prime)
    
        #X = hcat(X, phi[:, i])
        #Y = hcat(Y, kinetic_entropy(α[:, i], m, weights))
    end

    flux_wall!(fη, maxwell_boltzmann_dual.(α[:, 1]' * m)[:], points, dt, 1.0)
    for k in axes(flux, 1)
        flux[k, 1] = sum(m[k, :] .* weights .* fη)
    end

    @inbounds for i = 2:nx
        KitBase.flux_kfvs!(fη, KitBase.maxwell_boltzmann_dual.(α[:, i-1]' * m)[:], KitBase.maxwell_boltzmann_dual.(α[:, i]' * m)[:], points, dt)
        
        for k in axes(flux, 1)
            flux[k, i] = sum(m[k, :] .* weights .* fη)
        end
    end

    @inbounds for i = 1:nx-1
        for q = 1:1
            phi[q, i] =
                phi[q, i] +
                (flux[q, i] - flux[q, i+1]) / ps.dx[i] +
                (σs[i] * phi[q, i] - σt[i] * phi[q, i]) * dt +
                σq[i] * dt
        end

        for q = 2:ne
            phi[q, i] =
                phi[q, i] +
                (flux[q, i] - flux[q, i+1]) / ps.dx[i] +
                (-σt[i] * phi[q, i]) * dt
        end
    end
    phi[:, nx] .=  phi[:, nx-1]

    global t += dt
end
phi0 = deepcopy(phi)
#plot(ps.x, phi[1, :])

cd(@__DIR__)
model = KitML.load_model("tfmodel"; mode = :tf)

function tf_closure(model::PyObject, u, m, weights)
    u0 = phi[1, :]
    u1 = phi[2, :]
    _u = u1 ./ u0

    _h, _alpha = model.predict(_u)
    _h = _h[:]
    _alpha = _alpha[:]

    _alpha0 = zero(_alpha)
    for i in eachindex(_alpha0)
        t1 = _alpha[i] * m[2, :]
        t2 = exp.(t1)
        t3 = sum(t2 .* weights)
        t4 = -log(t3)
        t5 = t4 + log(u0[i])
        _alpha0[i] = t5
    end

    alpha = hcat(_alpha0, _alpha) |> permutedims

    return alpha
end

begin
    # initial condition
    f0 = 0.0001 * ones(nu)
    phi = zeros(ne, nx)
    for i = 1:nx
        phi[:, i] .= m * f0
    end
    α = zeros(Float32, ne, nx)
    flux = zeros(Float32, ne, nx + 1)
    fη = zeros(nu)
end

for iter = 1:nt
    println("iteration $iter of $nt")

    # neural optimizer
    α .= tf_closure(model, phi, m, weights)
    @inbounds for i = 1:nx
        #phi[:, i] .= KitBase.realizable_reconstruct(α[:, i], m, weights, KitBase.maxwell_boltzmann_dual_prime)
    end

    flux_wall!(fη, maxwell_boltzmann_dual.(α[:, 1]' * m)[:], points, dt, 1.0)
    for k in axes(flux, 1)
        flux[k, 1] = sum(m[k, :] .* weights .* fη)
    end

    @inbounds for i = 2:nx
        KitBase.flux_kfvs!(fη, KitBase.maxwell_boltzmann_dual.(α[:, i-1]' * m)[:], KitBase.maxwell_boltzmann_dual.(α[:, i]' * m)[:], points, dt)
        
        for k in axes(flux, 1)
            flux[k, i] = sum(m[k, :] .* weights .* fη)
        end
    end

    @inbounds for i = 1:nx-1
        for q = 1:1
            phi[q, i] =
                phi[q, i] +
                (flux[q, i] - flux[q, i+1]) / ps.dx[i] +
                (σs[i] * phi[q, i] - σt[i] * phi[q, i]) * dt +
                σq[i] * dt
        end

        for q = 2:ne
            phi[q, i] =
                phi[q, i] +
                (flux[q, i] - flux[q, i+1]) / ps.dx[i] +
                (-σt[i] * phi[q, i]) * dt
        end
    end
    phi[:, nx] .=  phi[:, nx-1]

    global t += dt
end

using DataFrames, CSV

df = DataFrame()
df.u0 = phi[1, :]
df.u1 = phi[2, :]
df.u0_ref = phi0[1, :]
df.u1_ref = phi0[2, :]

CSV.write("m1_t0.3.csv", df)
CSV.write("m1_t0.5.csv", df)
CSV.write("m1_t0.7.csv", df)

plot(ps.x, phi[1, :], lw=2, label="u0", xlabel="x", ylabel="u")
plot!(ps.x, phi[2, :], lw=2, label="u1", xlabel="x", ylabel="u")
plot!(ps.x, phi0[1, :], lw=2, line=:dash, color=:black, label="ref")
plot!(ps.x, phi0[2, :], lw=2, line=:dash, color=:black, label=:none)

savefig("1dm1_t0.3.pdf")
savefig("1dm1_t0.5.pdf")
savefig("1dm1_t0.7.pdf")
