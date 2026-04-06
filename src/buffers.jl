# buffers.jl — Pre-alocación y gestión de buffers para forward pass y backprop

"""
    crear_buffers(::Type{T}, dims_red::Vector{Int}, n_samples::Int) where {T<:AbstractFloat}

Pre-aloca buffers de activación y gradiente para el forward pass y backprop.

Los buffers se dimensionan según la capa más grande de la red (`max(dims_red)`)
y el número de muestras (`n_samples`), con layout column-major `(max_dim, n_samples)`.

Devuelve `Matrix{T}` en CPU. La transferencia a GPU (CuArray{Float32}) se realiza
posteriormente en el módulo de backend.

# Argumentos
- `T`: tipo numérico (`Float64` para CPU, `Float32` para GPU)
- `dims_red`: vector con el número de neuronas por capa (incluyendo entrada y salida)
- `n_samples`: número de muestras en el batch (ya clampeado al tamaño del dataset)

# Retorna
- `NamedTuple` con campos `buffer_activacion::Matrix{T}` y `buffer_gradiente::Matrix{T}`

# Ejemplo
```julia
bufs = crear_buffers(Float64, [4, 10, 3], 32)
bufs.buffer_activacion  # Matrix{Float64} de 10×32
bufs.buffer_gradiente   # Matrix{Float64} de 10×32
```
"""
function crear_buffers(::Type{T}, dims_red::Vector{Int}, n_samples::Int) where {T<:AbstractFloat}
    if isempty(dims_red)
        throw(ArgumentError("dims_red no puede estar vacío"))
    end
    if n_samples < 1
        throw(ArgumentError("n_samples debe ser ≥ 1, recibido: $n_samples"))
    end

    max_dim = maximum(dims_red)

    buffer_activacion = zeros(T, max_dim, n_samples)
    buffer_gradiente  = zeros(T, max_dim, n_samples)

    return (buffer_activacion=buffer_activacion, buffer_gradiente=buffer_gradiente)
end
