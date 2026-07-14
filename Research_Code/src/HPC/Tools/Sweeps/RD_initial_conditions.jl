### ADJUSTED: Add named two-species reaction-diffusion sweep initial conditions.
using Random
using Statistics

rd_normalize_initial_condition_name(name) = replace(lowercase(strip(string(name))), r"[-_]+" => " ")

"""Materialize a named stacked `[v1; v2]` RD initial condition."""
function rd_materialize_sweep_initial_condition(name; N, L, dimension, boundary_condition="neumann", seed=1)
    rd_normalize_initial_condition_name(name) == "default" && return nothing
    lowercase(strip(string(boundary_condition))) in ("neumann", "homogeneous_neumann") || error("RD sweep initial conditions require BOUNDARY_CONDITION=neumann")
    dim = Int(dimension)
    dim in (1, 2) || error("RD dimension must be 1 or 2")
    Δx = L / (N - 1)
    x = collect(range(0.0, L; length=N))
    key = rd_normalize_initial_condition_name(name)
    if key in ("random", "random field", "rd random field", "gaussian random field")
        rng = MersenneTwister(seed)
        n = N ^ dim
        smooth_steps = 3
        noise_level = 0.05
        background_state = -cbrt(0.005)

        function smooth_neumann_field(z)
            y = copy(z)
            if dim == 1
                for _ in 1:smooth_steps
                    old = copy(y)
                    y[1] = 0.5 * old[1] + 0.5 * old[2]
                    for i in 2:N-1
                        y[i] = 0.25 * old[i-1] + 0.5 * old[i] + 0.25 * old[i+1]
                    end
                    y[N] = 0.5 * old[N] + 0.5 * old[N-1]
                end
            else
                Y = reshape(y, N, N)
                for _ in 1:smooth_steps
                    old = copy(Y)
                    for j in 1:N, i in 1:N
                        im = i == 1 ? 1 : i - 1
                        ip = i == N ? N : i + 1
                        jm = j == 1 ? 1 : j - 1
                        jp = j == N ? N : j + 1
                        Y[i, j] = 0.5 * old[i, j] +
                                  0.125 * (old[im, j] + old[ip, j] + old[i, jm] + old[i, jp])
                    end
                end
                y = vec(Y)
            end

            y .-= mean(y)
            std_y = std(y)
            std_y > 0 || error("Generated zero-variance RD random field")
            return y ./ std_y
        end

        v1 = background_state .+ noise_level .* smooth_neumann_field(randn(rng, n))
        v2 = background_state .+ noise_level .* smooth_neumann_field(randn(rng, n))
    elseif dim == 1 && key in ("sine", "sinusoidal")
        v1 = 1 .+ 0.05 .* sin.(2π .* x ./ L)
        v2 = 1 .+ 0.05 .* cos.(2π .* x ./ L)
    elseif dim == 2 && key in ("two patches", "patches")
        v1 = [1 + 0.15 * exp(-((x[i] - 0.30L)^2 + (x[j] - 0.50L)^2) / (0.08L)^2) for i in 1:N, j in 1:N]
        v2 = [1 + 0.15 * exp(-((x[i] - 0.70L)^2 + (x[j] - 0.50L)^2) / (0.08L)^2) for i in 1:N, j in 1:N]
    elseif dim == 2 && key in ("sine", "sinusoidal")
        v1 = [1 + 0.05 * sin(2π * x[i] / L) * sin(2π * x[j] / L) for i in 1:N, j in 1:N]
        v2 = [1 + 0.05 * cos(2π * x[i] / L) * cos(2π * x[j] / L) for i in 1:N, j in 1:N]
    else
        error("Unknown RD INITIAL_CONDITION=$name. Available: default, random field, sine, two patches")
    end
    return vcat(vec(v1), vec(v2))
end
