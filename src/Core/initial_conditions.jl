
# Functions to build an initial condition.

"""String nonsense for initial condition"""
normalize_initial_condition_name(name) = replace(lowercase(strip(string(name))), r"[-_]+" => " ")

"""
Turn an initial condition into actual function values
- Args:
    - an EquationSpec; see types.jl
    - grid: a Grid; see types.jl
    - name: string with name of initial condition
    - config: a RunConfig; see types.jl
    - supplied: you can just pass in a function or function output directly
"""
function materialize_initial_condition(spec::EquationSpec, grid::Grid, name, config::RunConfig; supplied=nothing)
    fields = spec.fields
    if !isnothing(supplied)
        if supplied isa Function
            # evaluate supplied function 
            if grid.dimension == 1
                values = [supplied(x) for x in grid.x]
            else
                values = [supplied(grid.x[i], grid.x[j]) for i in 1:grid.N, j in 1:grid.N]
            end
            return normalize_state(values, grid; fields)
        end
        return normalize_state(supplied, grid; fields)
    end
    key = normalize_initial_condition_name(name)
    if key == "default"
        values = spec.default_initial_condition(grid, config)
    else
        values = spec.named_initial_condition(key, grid, config)
    end
    normalize_state(values, grid; fields)
end

"""
Makes sure our state vector makes sense; used for initial conditions
Args:
    - u is the state vector
    - grid is a Grid struct; see types.jl
    - fields is the number of vars we're keeping track of (for R-D, = 2)
"""
function normalize_state(u, grid::Grid; fields=1)
    n = spatial_length(grid; fields)
    if fields == 1 && grid.dimension == 2 && ndims(u) == 2 && size(u) == grid.state_shape
        return copy(vec(u))
    elseif fields > 1 && u isa Tuple && length(u) == fields
        return vcat((vec(copy(field)) for field in u)...)
    elseif length(u) == n
        return copy(vec(u))
    end
    error("initial condition has incompatible shape")
end

