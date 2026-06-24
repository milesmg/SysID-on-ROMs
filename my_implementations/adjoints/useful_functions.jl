using LinearAlgebra
using SparseArrays

function lap1d(κ, L, Nx; periodic=false)
    if !periodic
        println("homogenous dirichlet boundary conditions")
    else
        println("periodic boundary conditions")
    end

    Δx = L / Nx
    coeff = κ / Δx^2

    main_diag = fill(-2.0, Nx)
    lower_diag = fill(1.0, Nx - 1)
    upper_diag = fill(1.0, Nx - 1)

    A_diffusion = coeff * spdiagm(
        -1 => lower_diag,
         0 => main_diag,
         1 => upper_diag,
    )

    if periodic
        A_diffusion[1, end] = coeff
        A_diffusion[end, 1] = coeff
    end

    return A_diffusion
end

function step_forward_Euler!(u, t, A)
    u[:, t + 1] .= A * u[:, t]
    return nothing
end

function solve_forward_diffusion_1d(u₀, T, L, Nx, κ, Nsteps; periodic=false)
    A_diffusion = lap1d(κ, L, Nx; periodic=periodic)
    u = zeros(Nx, Nsteps + 1)
    u[:, 1] .= u₀
    Δt = T / Nsteps

    M = sparse(I, Nx, Nx) + Δt * A_diffusion

    for t in 1:Nsteps
        step_forward_Euler!(u, t, M)
    end

    return u
end

function solve_simple_adjoint_diffusion_1d(y, T, L, Nx, κ, Nsteps; periodic=false)
    A_diffusion = lap1d(κ, L, Nx; periodic=periodic)
    v = zeros(Nx, Nsteps + 1)
    v[:, 1] .= y
    Δt = T / Nsteps

    M = (sparse(I, Nx, Nx) + Δt * A_diffusion)'

    for t in 1:Nsteps
        step_forward_Euler!(v, t, M)
    end

    return v
end

function solve_trajectory_adjoint_diffusion_1d(y, T, L, Nx, κ, Nsteps; periodic=false)
    A_diffusion = lap1d(κ, L, Nx; periodic=periodic)
    v = zeros(Nx, Nsteps + 1)
    v[:, 1] .= y[:, end]
    Δt = T / Nsteps

    M = (sparse(I, Nx, Nx) + Δt * A_diffusion)'

    for t in 1:Nsteps
        step_forward_Euler!(v, t, M)
        v[:, t + 1] .+= y[:, end - t]
    end

    return v
end

function step_forward_Euler_AC!(u, t, A, f_c, Δt)
    u[:, t + 1] .= A * u[:, t] .- Δt .* f_c(u[:, t])
    return nothing
end

function solve_forward_AC_1d(u₀, T, L, Nx, κ, Nsteps; periodic=false, f_c=nothing)
    if isnothing(f_c)
        f_c = u -> @. u^3 - u
    end

    A_diffusion = lap1d(κ, L, Nx; periodic=periodic)
    u = zeros(Nx, Nsteps + 1)
    u[:, 1] .= u₀
    Δt = T / Nsteps

    M = sparse(I, Nx, Nx) + Δt * A_diffusion

    for t in 1:Nsteps
        step_forward_Euler_AC!(u, t, M, f_c, Δt)
    end

    return u
end

function solve_simple_adjoint_AC_1d(u, y, T, L, Nx, κ, Nsteps; periodic=false)
    fc_prime_u = @. 3 * u^2 - 1

    A_diffusion = lap1d(κ, L, Nx; periodic=periodic)
    v = zeros(Nx, Nsteps + 1)
    v[:, 1] .= y
    Δt = T / Nsteps

    M = (sparse(I, Nx, Nx) + Δt * A_diffusion)'

    for t in 1:Nsteps
        Jt = M - Δt * spdiagm(0 => fc_prime_u[:, Nsteps + 1 - t])
        step_forward_Euler!(v, t, Jt)
    end

    return v
end
