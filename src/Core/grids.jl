# Functions for generating and storing the spatial discretization information

"""
Ensure dims 1 or 2
"""
function validate_dimension(dimension)
    dim = Int(dimension)
    dim in (1, 2) || error("dimension must be 1 or 2")
    return dim
end

"""
Make sure N works with a 2D grid if that's what we're working with
"""
state_grid_size(state_length::Integer, dimension) = begin
    dim = validate_dimension(dimension)
    dim == 1 && return state_length
    grid_N = round(Int, sqrt(state_length))
    grid_N^2 == state_length || error("2D states must have square flattened length")
    grid_N
end

"""build a spatial weight for discrete integration"""
spatial_measure(Δx, dimension) = Δx ^ validate_dimension(dimension)
"""build a spatial weight for discrete integration based on a 'grid' object """
spatial_measure(grid::Grid) = grid.Δx ^ grid.dimension

"""
Calculate required length of the state vector;
Args:
    - grid is a Grid struct
    - fields: throughout, fields is the number of state vars we need to keep track of (in the case of RD=2)
"""
spatial_length(grid::Grid; fields=1) = fields * grid.N ^ grid.dimension

"""
This function builds the 'grid' struct; see types.jl
"""
function spatial_grid(N::Integer, L, dimension::Integer, boundary_condition)::Grid
    dimension = validate_dimension(dimension)
    boundary = String(boundary_condition)
    boundary in ("periodic", "dirichlet", "neumann") || error("boundary_condition must be periodic, dirichlet, or neumann")
    # here, we have N points:

    # periodic associates points at start and end; here, we set those to be (0, N+1) so still N points
    # Counting the wrap-around space, we have N spaces between points 

    # dirichlet puts ghost points at the boundaries, here at (0,N+1)
    # thus there are N+1 spaces between points and N points

    # neumann doesn't have the same ghost point structure; it just enforces these derviatives
    # hence, we have N points with N-1 spaces between them
    
    if boundary == "dirichlet"
        Δx = L / (N + 1)
    elseif boundary == "neumann"
        Δx = L / (N - 1)
    elseif boundary == "periodic"
        Δx = L / N
    end
    if boundary == "periodic" || boundary == "neumann"
        x = Δx .* collect(0:N-1)
    elseif boundary == "dirichlet"
        x = Δx .* collect(1:N)
    end
    y, state_shape = nothing, (Int(N),)
    if dimension == 2
        y, state_shape = Float64.(x), (Int(N), Int(N))
    end
    Grid(Int(N), Float64(L), dimension, boundary, Float64(Δx), Float64.(x), y, state_shape)
end
