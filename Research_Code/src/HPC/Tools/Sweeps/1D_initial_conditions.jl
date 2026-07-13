const SWEEP_1D_INITIAL_CONDITION_NAMES = [
    "step",
    "low frequency sine",
    "high frequency sine",
    "off-center bump",
]

const SWEEP_1D_INITIAL_CONDITION_FUNCTIONS = Function[
    (x, L, ε2, w) -> x < 0.5L ? -1.0 : 1.0,
    (x, L, ε2, w) -> sin(2π * x / L),
    (x, L, ε2, w) -> sin(8π * x / L),
    (x, L, ε2, w) -> exp(-((x - 0.35L)^2) / (2 * (0.08L)^2)),
]

"""Return the named 1D pointwise initial-condition function."""
function sweep_1d_initial_condition(name)
    key = normalize_initial_condition_name(name)
    index = findfirst(==(key), SWEEP_1D_INITIAL_CONDITION_NAMES)
    isnothing(index) && error("Unknown 1D INITIAL_CONDITION=$name. Available: default, " * join(SWEEP_1D_INITIAL_CONDITION_NAMES, ", "))
    return SWEEP_1D_INITIAL_CONDITION_FUNCTIONS[index]
end

"""Materialize a named 1D sweep initial condition on the active Allen-Cahn grid."""
function materialize_1d_sweep_initial_condition(name; N, L, ε2, dimension, boundary_condition)
    normalize_initial_condition_name(name) == "default" && return nothing
    dim = validate_ac_dimension(dimension)
    dim == 1 || error("Named sweep initial conditions in `1D_initial_conditions.jl` require DIMENSION=1; got DIMENSION=$dimension")
    f = sweep_1d_initial_condition(name)
    grid = ac_grid(N, L, boundary_condition)
    w = sqrt(2ε2)
    return Float64[f(x, L, ε2, w) for x in grid.x]
end

"""Materialize a named 1D or 2D sweep initial condition on the active Allen-Cahn grid."""
function materialize_sweep_initial_condition(name; N, L, ε2, dimension, boundary_condition)
    normalize_initial_condition_name(name) == "default" && return nothing
    dim = validate_ac_dimension(dimension)
    if dim == 1
        return materialize_1d_sweep_initial_condition(name; N, L, ε2, dimension, boundary_condition)
    end
    f = sweep_initial_condition(name)
    grid = ac_grid(N, L, boundary_condition)
    w = sqrt(2ε2)
    return Float64[f(grid.x[i], grid.x[j], L, ε2, w) for i in 1:N, j in 1:N]
end
