using LinearAlgebra
using KitBase, KitBase.FastGaussQuadrature, KitBase.ProgressMeter, KitBase.JLD2, KitBase.Plots
using KitML, KitML.Flux, KitML.DiffEqFlux, KitML.Optim, KitML.CSV

fnn = FastICNN(3, 1, [15, 15, 15, 15], tanh)

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
    set = Setup("radiation", "linesource", "1d1f1v", "kfvs", "bgk", 1, 2, "vanleer", "extra", 0.5, 0.5)

    x0 = 0
    x1 = 1
    nx = 100
    nxg = 0
    ps = PSpace1D(x0, x1, nx, nxg)

    nu = 28
    points, weights = gausslegendre(nu)
    vs = VSpace1D(points[1], points[end], nu, points, ones(nu) .* (points[end] - points[1]) / (nu - 1), weights)

    # material
    σs = ones(Float32, nx)
    σa = zeros(Float32, nx)
    σt = σs + σa
    σq = zeros(Float32, nx)

    L = 2
    ne = 3
    m = eval_sphermonomial(points, L)

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
    
    opt = optimize_closure(zeros(Float32, ne), m, weights, phi[:, 1], KitBase.maxwell_boltzmann_dual)
    
    global X = zeros(Float32, ne, 1)
    X[:, 1] .= phi[:, 1]
    global Y = zeros(Float32, 1, 1) # h
    Y[1, 1] = kinetic_entropy(opt.minimizer, m, weights)

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

phi_ref = deepcopy(phi)
for i = 1:nx
    global X = hcat(X, phi[:, i])
    global Y = hcat(Y, kinetic_entropy(α[:, i], m, weights))
end

res = sci_train(fnn, (X, Y); maxiters=5000, device=gpu)
res = sci_train(fnn, (X, Y), res.u; maxiters=10000, device=cpu)
res = sci_train(fnn, (X, Y), res.u, LBFGS(); maxiters=10000, device=cpu)

X_old = deepcopy(X)
Y_old = deepcopy(Y)

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

X = deepcopy(X_old)
Y = deepcopy(Y_old)

for iter = 1:nt
    println("iteration $iter of $nt")
    phi_old .= phi

    # regularization
    for i = 1:nx
        α[:, i] .= KitML.neural_closure(fnn, res.u, phi_old[:, i])
        phi_temp[:, i] .= KitBase.realizable_reconstruct(opt.minimizer, m, weights, KitBase.maxwell_boltzmann_dual_prime)
    end

    counter = 0
    @inbounds for i = 1:nx
        if norm(phi_temp[:, i] .- phi_old[:, i], 1) / (phi_old[1, i] + 1e-3) > 1e-3
            counter +=1

            opt = KitBase.optimize_closure(α[:, i], m, weights, phi[:, i], KitBase.maxwell_boltzmann_dual)
            α[:, i] .= opt.minimizer
            phi[:, i] .= KitBase.realizable_reconstruct(opt.minimizer, m, weights, KitBase.maxwell_boltzmann_dual_prime)

            X = hcat(X, phi[:, i])
            Y = hcat(Y, kinetic_entropy(α[:, i], m, weights))
        else
            phi[:, i] .= phi_temp[:, i]
        end
    end
    println("newton: $counter of $nx")

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

    if iter%19 == 0 && counter > nx÷2
        global res = KitML.sci_train(fnn, (X, Y), res.u, ADAM(); maxiters=1000)
        global res = KitML.sci_train(fnn, (X, Y), res.u, LBFGS(); maxiters=2000)
    end
end

res = sci_train(fnn, (X, Y), res.u; maxiters=10000, device=cpu)
res = sci_train(fnn, (X, Y), res.u, LBFGS(); maxiters=10000, device=cpu)

plot(ps.x[1:nx], phi_ref[1, :])
plot!(ps.x[1:nx], phi[1, :])

begin
    _α0 = zeros(Float32, ne, nx)
    _α1 = zero(_α0)
    _h0 = zeros(Float32, nx)
    _h1 = zero(_h0)
    for i = 1:nx
        _h1[i] = fnn(phi[:, i], res.u)[1]
        _α1[:, i] .= KitML.neural_closure(fnn, res.u, phi[:, i])

        opt = KitBase.optimize_closure(_α0[:, i], m, weights, phi[:, i], KitBase.maxwell_boltzmann_dual)
        _α0[:, i] .= opt.minimizer
        _h0[i] = kinetic_entropy(_α0[:, i], m, weights)
    end
end

plot(ps.x[1:nx], _h0)
plot!(ps.x[1:nx], _h1, line=:dash)

plot(ps.x[1:nx], _α0')
plot!(ps.x[1:nx], _α1', line=:dash)