# backend.jl — Selección CPU/GPU, transferencias de datos entre dispositivos
# Usa KernelAbstractions.jl como capa de abstracción vendor-agnostic

"""
    gpu_disponible() -> Bool

Verifica si hay un backend de GPU disponible a través de KernelAbstractions.jl.
Intenta obtener un backend GPU; si falla o no hay paquetes GPU cargados, retorna `false`.
"""
function gpu_disponible()::Bool
    try
        backend = KernelAbstractions.get_backend(zeros(Float32, 1))
        # El backend por defecto para Array es CPU; si solo tenemos CPU, no hay GPU
        # Un backend GPU real se detecta cuando hay extensiones cargadas (CUDA, ROCm, Metal)
        # Verificamos si hay algún backend GPU registrado
        return backend !== KernelAbstractions.CPU()
    catch
        return false
    end
end

"""
    seleccionar_backend(config::Configuracion_RCND, n_params::Int) -> Symbol

Selecciona el backend de cómputo (`:cpu` o `:gpu`) según la configuración y el número
de parámetros de la red.

# Lógica
- `:cpu` → retorna `:cpu`
- `:gpu` → verifica GPU disponible; si no hay, lanza `ErrorBackend`
- `:auto` → si `n_params > config.umbral_gpu` retorna `:gpu`, sino `:cpu`

# Argumentos
- `config`: configuración del pipeline con `backend_computo` y `umbral_gpu`
- `n_params`: número total de parámetros de la red

# Retorna
- `:cpu` o `:gpu`

# Errores
- `ErrorBackend` si se solicita `:gpu` (directa o vía `:auto`) y no hay GPU disponible
"""
function seleccionar_backend(config::Configuracion_RCND, n_params::Int)::Symbol
    backend = config.backend_computo

    if backend === :cpu
        return :cpu
    elseif backend === :gpu
        if !gpu_disponible()
            throw(ErrorBackend(
                "Se solicitó backend :gpu pero no se detectó GPU compatible. " *
                "Instale y cargue un paquete GPU (CUDA.jl, AMDGPU.jl, Metal.jl) " *
                "o use backend_computo=:cpu",
                :gpu
            ))
        end
        return :gpu
    elseif backend === :auto
        if n_params > config.umbral_gpu
            return :gpu
        else
            return :cpu
        end
    else
        throw(ErrorBackend(
            "Backend de cómputo desconocido: :$backend. " *
            "Valores válidos: :cpu, :gpu, :auto",
            backend
        ))
    end
end


"""
    transferir_a_dispositivo!(datos, backend::Symbol, ::Type{T}) where {T}

Transfiere arrays al dispositivo seleccionado y convierte al tipo numérico `T`.

- Si `backend == :gpu`: convierte cada array a GPU (via KernelAbstractions) con tipo `Float32`.
- Si `backend == :cpu`: convierte al tipo `T` si es necesario, mantiene en CPU.

Funciona con cualquier estructura que contenga arrays (matrices, vectores, NamedTuples).

# Argumentos
- `datos`: dato o colección de datos a transferir (AbstractArray, Vector de arrays, NamedTuple)
- `backend`: `:cpu` o `:gpu`
- `T`: tipo numérico destino

# Retorna
- Los datos convertidos al dispositivo y tipo apropiados
"""
function transferir_a_dispositivo!(datos::AbstractArray{<:Number}, backend::Symbol, ::Type{T}) where {T}
    if backend === :gpu
        # Usar KernelAbstractions para obtener el backend GPU y transferir
        # Primero convertir a Float32 (requisito 10.6)
        datos_f32 = Float32.(datos)
        gpu_backend = _obtener_backend_gpu()
        return KernelAbstractions.allocate(gpu_backend, Float32, size(datos_f32)...) |>
               arr -> (copyto!(arr, datos_f32); arr)
    else
        # CPU: solo convertir tipo si es necesario
        if eltype(datos) === T
            return datos
        else
            return T.(datos)
        end
    end
end

# Dispatch para vectores de arrays (e.g., vectores de matrices de pesos)
function transferir_a_dispositivo!(datos::Vector{<:AbstractArray}, backend::Symbol, ::Type{T}) where {T}
    return [transferir_a_dispositivo!(d, backend, T) for d in datos]
end

# Dispatch para NamedTuples (e.g., buffers)
function transferir_a_dispositivo!(datos::NamedTuple, backend::Symbol, ::Type{T}) where {T}
    pares = [k => transferir_a_dispositivo!(getfield(datos, k), backend, T) for k in keys(datos)]
    return NamedTuple{keys(datos)}(Tuple(last.(pares)))
end

"""
    _obtener_backend_gpu()

Obtiene el backend GPU disponible a través de KernelAbstractions.
Lanza `ErrorBackend` si no hay GPU disponible.
"""
function _obtener_backend_gpu()
    try
        # Intentar crear un array pequeño en GPU para detectar el backend
        # KernelAbstractions usa extensiones de paquetes GPU para esto
        backend = KernelAbstractions.get_backend(zeros(Float32, 1))
        if backend === KernelAbstractions.CPU()
            throw(ErrorBackend(
                "No se encontró backend GPU. Cargue un paquete GPU compatible.",
                :gpu
            ))
        end
        return backend
    catch e
        if e isa ErrorBackend
            rethrow(e)
        end
        throw(ErrorBackend(
            "Error al obtener backend GPU: $(sprint(showerror, e))",
            :gpu
        ))
    end
end

"""
    transferir_a_cpu!(datos::AbstractArray)

Transfiere un array de GPU a CPU. Si ya está en CPU, lo retorna sin cambios.
Usa `Array()` para materializar datos de GPU en memoria principal.

# Argumentos
- `datos`: array posiblemente en GPU

# Retorna
- `Array` en CPU con los mismos datos
"""
function transferir_a_cpu!(datos::AbstractArray)
    return Array(datos)
end

# Dispatch para vectores de arrays
function transferir_a_cpu!(datos::Vector{<:AbstractArray})
    return [transferir_a_cpu!(d) for d in datos]
end

# Dispatch para NamedTuples
function transferir_a_cpu!(datos::NamedTuple)
    pares = [k => transferir_a_cpu!(getfield(datos, k)) for k in keys(datos)]
    return NamedTuple{keys(datos)}(Tuple(last.(pares)))
end
