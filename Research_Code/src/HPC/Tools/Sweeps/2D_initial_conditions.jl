### ADJUSTED: Keep sweep-selectable initial conditions limited to the current 2D stability notebook options.
normalize_initial_condition_name(name) = replace(lowercase(strip(string(name))), r"[-_]+" => " ")

### ADJUSTED: Store the current 2D initial-condition names and pointwise functions in parallel arrays.
const SWEEP_2D_INITIAL_CONDITION_NAMES = [
    "2d circle drop",
    "2d offcenter drop",
    "2d one direction tanh front",
    "2d annulus",
    "2d soft bleedout patch and static slab",
    "2d sin xy",
    "2d high frequency x sine",
]

const SWEEP_2D_INITIAL_CONDITION_FUNCTIONS = Function[
    (x, y, L, ε2, w) -> tanh((sqrt((x - 0.5L)^2 + (y - 0.5L)^2) - 0.23L) / w),
    (x, y, L, ε2, w) -> tanh((sqrt((x - 0.35L)^2 + (y - 0.58L)^2) - 0.18L) / w),
    (x, y, L, ε2, w) -> tanh((x - 0.5L) / w),
    (x, y, L, ε2, w) -> max(tanh((sqrt((x - 0.5L)^2 + (y - 0.5L)^2) - 0.30L) / w), tanh((0.15L - sqrt((x - 0.5L)^2 + (y - 0.5L)^2)) / w)),
    (x, y, L, ε2, w) -> 0.5 * (1 - tanh((x - 0.50L) / (2.0w))) * (0.55sin(6π * x / L) * sin(4π * y / L) + 0.20sin(10π * x / L + 0.7) * sin(2π * y / L)) + 0.5 * (1 + tanh((x - 0.50L) / (2.0w))) * tanh((x - 0.72L) / (1.2w)),
    (x, y, L, ε2, w) -> sin(2π * x / L) * sin(2π * y / L),
    (x, y, L, ε2, w) -> sin(3π * x / L),
]

"""Return the named 2D pointwise initial-condition function."""
function sweep_initial_condition(name)
    key = normalize_initial_condition_name(name)
    index = findfirst(==(key), SWEEP_2D_INITIAL_CONDITION_NAMES)
    isnothing(index) && error("Unknown 2D INITIAL_CONDITION=$name. Available: default, " * join(SWEEP_2D_INITIAL_CONDITION_NAMES, ", "))
    return SWEEP_2D_INITIAL_CONDITION_FUNCTIONS[index]
end

"""Materialize a named 2D sweep initial condition on the active Allen-Cahn grid."""
function materialize_sweep_initial_condition(name; N, L, ε2, dimension, boundary_condition)
    normalize_initial_condition_name(name) == "default" && return nothing
    dim = validate_ac_dimension(dimension)
    ### ADJUSTED: Report the underscore registry filename in dimension-guard failures.
    dim == 2 || error("Named sweep initial conditions in `2D_initial_conditions.jl` require DIMENSION=2; got DIMENSION=$dimension")
    f = sweep_initial_condition(name)
    grid = ac_grid(N, L, boundary_condition)
    w = sqrt(2ε2)
    return Float64[f(grid.x[i], grid.x[j], L, ε2, w) for i in 1:N, j in 1:N]
end
