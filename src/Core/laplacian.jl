# Build the Laplacian as a matrix, or apply it in place.

"""Select which in-place Laplacian to apply based on boundary conditions
Args:
    - du: vector for storing Laplacian
    - u: vector to which Laplacian is applied
    - grid: a Grid struct; see types.jl
    - scale: if you want to multiply the laplacian by something (eg ε²)
"""

function laplacian!(du, u, grid::Grid; scale=one(eltype(u)))
    boundary = grid.boundary_condition
    boundary == "periodic" && return periodic_laplacian!(du, u, grid; scale)
    boundary == "dirichlet" && return dirichlet_laplacian!(du, u, grid; scale)
    boundary == "neumann" && return neumann_laplacian!(du, u, grid; scale)
end

"""Apply a periodic Laplacian in-place."""
function periodic_laplacian!(du, u, grid::Grid; scale=one(eltype(u)))
    N, invΔx2 = grid.N, scale / grid.Δx^2
    if grid.dimension == 1
        @inbounds begin
            du[1] = (u[N] - 2u[1] + u[2]) * invΔx2
            for i in 2:N-1
                du[i] = (u[i - 1] - 2u[i] + u[i + 1]) * invΔx2
            end
            du[N] = (u[N - 1] - 2u[N] + u[1]) * invΔx2
        end
        return nothing
    end
    laplacian_2d_interior!(du, u, N, invΔx2)
    @inbounds begin
        for i in 2:N-1
            bottom, top = i, i + (N - 1) * N
            du[bottom] = (u[bottom - 1] + u[bottom + 1] + u[top] + u[bottom + N] - 4u[bottom]) * invΔx2
            du[top] = (u[top - 1] + u[top + 1] + u[top - N] + u[i] - 4u[top]) * invΔx2
        end
        for j in 2:N-1
            left, right = 1 + (j - 1) * N, j * N
            du[left] = (u[right] + u[left + 1] + u[left - N] + u[left + N] - 4u[left]) * invΔx2
            du[right] = (u[right - 1] + u[left] + u[right - N] + u[right + N] - 4u[right]) * invΔx2
        end
        du[1] = (u[N] + u[2] + u[(N - 1) * N + 1] + u[N + 1] - 4u[1]) * invΔx2
        du[N] = (u[N - 1] + u[1] + u[N * N] + u[2N] - 4u[N]) * invΔx2
        du[(N - 1) * N + 1] = (u[N * N] + u[(N - 1) * N + 2] + u[(N - 2) * N + 1] + u[1] - 4u[(N - 1) * N + 1]) * invΔx2
        du[N * N] = (u[N * N - 1] + u[(N - 1) * N + 1] + u[N * (N - 1)] + u[N] - 4u[N * N]) * invΔx2
    end
    nothing
end

"""Apply a Dirichlet Laplacian in-place."""
function dirichlet_laplacian!(du, u, grid::Grid; scale=one(eltype(u)))
    N, invΔx2 = grid.N, scale / grid.Δx^2
    if grid.dimension == 1
        @inbounds begin
            du[1] = (-2u[1] + u[2]) * invΔx2
            for i in 2:N-1
                du[i] = (u[i - 1] - 2u[i] + u[i + 1]) * invΔx2
            end
            du[N] = (u[N - 1] - 2u[N]) * invΔx2
        end
        return nothing
    end
    laplacian_2d_interior!(du, u, N, invΔx2)
    @inbounds begin
        for i in 2:N-1
            bottom, top = i, i + (N - 1) * N
            du[bottom] = (u[bottom - 1] + u[bottom + 1] + u[bottom + N] - 4u[bottom]) * invΔx2
            du[top] = (u[top - 1] + u[top + 1] + u[top - N] - 4u[top]) * invΔx2
        end
        for j in 2:N-1
            left, right = 1 + (j - 1) * N, j * N
            du[left] = (u[left + 1] + u[left - N] + u[left + N] - 4u[left]) * invΔx2
            du[right] = (u[right - 1] + u[right - N] + u[right + N] - 4u[right]) * invΔx2
        end
        du[1] = (u[2] + u[N + 1] - 4u[1]) * invΔx2
        du[N] = (u[N - 1] + u[2N] - 4u[N]) * invΔx2
        du[(N - 1) * N + 1] = (u[(N - 1) * N + 2] + u[(N - 2) * N + 1] - 4u[(N - 1) * N + 1]) * invΔx2
        du[N * N] = (u[N * N - 1] + u[N * (N - 1)] - 4u[N * N]) * invΔx2
    end
    nothing
end

"""Apply a Neumann Laplacian in-place."""
function neumann_laplacian!(du, u, grid::Grid; scale=one(eltype(u)))
    N, invΔx2 = grid.N, scale / grid.Δx^2
    if grid.dimension == 1
        @inbounds begin
            du[1] = (-u[1] + u[2]) * invΔx2
            for i in 2:N-1
                du[i] = (u[i - 1] - 2u[i] + u[i + 1]) * invΔx2
            end
            du[N] = (u[N - 1] - u[N]) * invΔx2
        end
        return nothing
    end
    laplacian_2d_interior!(du, u, N, invΔx2)
    @inbounds begin
        for i in 2:N-1
            bottom, top = i, i + (N - 1) * N
            du[bottom] = (u[bottom - 1] + u[bottom + 1] + u[bottom + N] - 3u[bottom]) * invΔx2
            du[top] = (u[top - 1] + u[top + 1] + u[top - N] - 3u[top]) * invΔx2
        end
        for j in 2:N-1
            left, right = 1 + (j - 1) * N, j * N
            du[left] = (u[left + 1] + u[left - N] + u[left + N] - 3u[left]) * invΔx2
            du[right] = (u[right - 1] + u[right - N] + u[right + N] - 3u[right]) * invΔx2
        end
        du[1] = (u[2] + u[N + 1] - 2u[1]) * invΔx2
        du[N] = (u[N - 1] + u[2N] - 2u[N]) * invΔx2
        du[(N - 1) * N + 1] = (u[(N - 1) * N + 2] + u[(N - 2) * N + 1] - 2u[(N - 1) * N + 1]) * invΔx2
        du[N * N] = (u[N * N - 1] + u[N * (N - 1)] - 2u[N * N]) * invΔx2
    end
    nothing
end

"""Apply the 2D five-point stencil away from the boundary."""
function laplacian_2d_interior!(du, u, N, invΔx2)
    @inbounds for j in 2:N-1, i in 2:N-1
        index = i + (j - 1) * N
        du[index] = (u[index - 1] + u[index + 1] + u[index - N] + u[index + N] - 4u[index]) * invΔx2
    end
end


function laplacian_matrix(grid::Grid; scale=one(grid.Δx))
    N = grid.N
    diagonal = fill(-2 * scale / grid.Δx^2, N)
    off_diagonal = fill(scale / grid.Δx^2, N - 1)
    L1 = spdiagm(-1 => off_diagonal, 0 => diagonal, 1 => off_diagonal)
    if grid.boundary_condition == "periodic"
        L1[1, N] = scale / grid.Δx^2
        L1[N, 1] = scale / grid.Δx^2
    elseif grid.boundary_condition == "neumann"
        L1[1, 1] = -scale / grid.Δx^2
        L1[N, N] = -scale / grid.Δx^2
    end
    grid.dimension == 1 && return sparse(L1)
    I_N = sparse(I, N, N)
    sparse(kron(I_N, L1) + kron(L1, I_N))
end
