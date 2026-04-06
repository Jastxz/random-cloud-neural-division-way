# tipos.jl — Definiciones de tipos: errores, TopologiaOptima, Configuracion_RCND, Informe_RCND

# ─── Tipos de error ───────────────────────────────────────────────────────────

"""
    ErrorConfiguracion <: Exception

Error lanzado cuando la configuración del pipeline contiene valores inválidos.
"""
struct ErrorConfiguracion <: Exception
    mensaje::String
end

"""
    ErrorAdaptador <: Exception

Error lanzado cuando la conversión RedNeuronal → RedBase falla en una capa específica.
"""
struct ErrorAdaptador <: Exception
    mensaje::String
    capa::Int
end

"""
    ErrorBackend <: Exception

Error lanzado cuando hay problemas con el backend de cómputo (CPU/GPU).
"""
struct ErrorBackend <: Exception
    mensaje::String
    backend::Symbol
end

"""
    ErrorDeserializacion <: Exception

Error lanzado cuando la deserialización de un Informe_RCND desde JSON falla.
"""
struct ErrorDeserializacion <: Exception
    mensaje::String
    campo::Union{String, Nothing}
end

# ─── TopologiaOptima ──────────────────────────────────────────────────────────

"""
    TopologiaOptima{T<:AbstractFloat}

Representa la red neuronal con la arquitectura mínima encontrada por la Fase_Nube.

# Campos
- `capas`: neuronas por capa (incluyendo entrada y salida)
- `pesos`: matrices de pesos por capa, cada una de dimensión `(n_out, n_in)`
- `biases`: vectores de biases por capa, cada uno de dimensión `(n_out,)`
- `activaciones`: función de activación por capa (`:sigmoid`, `:relu`, `:identity`)
"""
struct TopologiaOptima{T<:AbstractFloat}
    capas::Vector{Int}
    pesos::Vector{Matrix{T}}
    biases::Vector{Vector{T}}
    activaciones::Vector{Symbol}
end

# ─── Configuracion_RCND ──────────────────────────────────────────────────────

"""
    Configuracion_RCND{T<:AbstractFloat}

Estructura inmutable que agrupa todos los parámetros del pipeline RCND.
Usa un constructor interno con keyword arguments y validación de invariantes.

# Campos — Fase Nube
- `n_redes_por_nube`: número de redes por nube (≥ 1)
- `activaciones`: funciones de activación permitidas (`:sigmoid`, `:relu`, `:identity`)
- `umbral_precision`: precisión objetivo en [0, 1]
- `max_iteraciones_reduccion`: máximo de iteraciones de reducción

# Campos — Fase División
- `umbral_division`: umbral de división en [0, 1]
- `max_epocas`: máximo de épocas de entrenamiento
- `tasa_aprendizaje`: tasa de aprendizaje del optimizador Adam
- `tamano_mini_batch`: tamaño de mini-batch (≥ 1)

# Campos — Global
- `semilla`: semilla aleatoria para reproducibilidad, o `nothing`
- `verbosidad`: nivel de verbosidad (`:silencioso`, `:normal`, `:detallado`)

# Campos — GPU
- `backend_computo`: backend de cómputo (`:cpu`, `:gpu`, `:auto`)
- `umbral_gpu`: número mínimo de parámetros para activar GPU en modo `:auto`

# Campos — Callbacks
- `callback_fase`: función callback entre fases, o `nothing`
"""
struct Configuracion_RCND{T<:AbstractFloat}
    # Fase Nube
    n_redes_por_nube::Int
    activaciones::Vector{Symbol}
    umbral_precision::T
    max_iteraciones_reduccion::Int

    # Fase División
    umbral_division::T
    max_epocas::Int
    tasa_aprendizaje::T
    tamano_mini_batch::Int

    # Global
    semilla::Union{Int, Nothing}
    verbosidad::Symbol

    # GPU
    backend_computo::Symbol
    umbral_gpu::Int

    # Callbacks
    callback_fase::Union{Function, Nothing}

    function Configuracion_RCND{T}(;
        n_redes_por_nube::Int = 100,
        activaciones::Vector{Symbol} = [:sigmoid, :relu, :identity],
        umbral_precision::Real = T(0.95),
        max_iteraciones_reduccion::Int = 10,
        umbral_division::Real = T(0.5),
        max_epocas::Int = 100,
        tasa_aprendizaje::Real = T(0.001),
        tamano_mini_batch::Int = 32,
        semilla::Union{Int, Nothing} = nothing,
        verbosidad::Symbol = :normal,
        backend_computo::Symbol = :auto,
        umbral_gpu::Int = 10_000,
        callback_fase::Union{Function, Nothing} = nothing,
    ) where {T<:AbstractFloat}
        # Validación de invariantes
        up = T(umbral_precision)
        if !(zero(T) <= up <= one(T))
            throw(ArgumentError("umbral_precision debe estar en [0, 1], recibido: $up"))
        end
        if n_redes_por_nube < 1
            throw(ArgumentError("n_redes_por_nube debe ser ≥ 1, recibido: $n_redes_por_nube"))
        end
        if tamano_mini_batch < 1
            throw(ArgumentError("tamano_mini_batch debe ser ≥ 1, recibido: $tamano_mini_batch"))
        end

        new{T}(
            n_redes_por_nube,
            activaciones,
            up,
            max_iteraciones_reduccion,
            T(umbral_division),
            max_epocas,
            T(tasa_aprendizaje),
            tamano_mini_batch,
            semilla,
            verbosidad,
            backend_computo,
            umbral_gpu,
            callback_fase,
        )
    end
end

"""
    Configuracion_RCND(; kwargs...)

Alias de conveniencia que infiere `T=Float64`.
"""
Configuracion_RCND(; kwargs...) = Configuracion_RCND{Float64}(; kwargs...)

# ─── Informe_RCND ─────────────────────────────────────────────────────────────

"""
    Informe_RCND{T<:AbstractFloat}

Informe combinado con los resultados de ambas fases del pipeline RCND.

# Campos — Fase Nube
- `topologia`: topología óptima encontrada por la Fase_Nube
- `precision_topologia`: precisión alcanzada por la topología
- `tiempo_fase_nube`: tiempo de ejecución de la Fase_Nube en segundos

# Campos — Fase División
- `mapa_soluciones`: mapa de soluciones generado, o `nothing` si la fase no se ejecutó/falló
- `precisiones_subredes`: precisión de cada subred, o `nothing`
- `tiempo_fase_division`: tiempo de ejecución de la Fase_División, o `nothing`

# Campos — Global
- `tiempo_total`: tiempo total de ejecución del pipeline en segundos
- `config_utilizada`: configuración usada para la ejecución (para reproducibilidad)
- `umbral_alcanzado`: indica si la Fase_Nube alcanzó el umbral de precisión objetivo
- `error_fase_division`: mensaje de error si la Fase_División falló, o `nothing`
"""
struct Informe_RCND{T<:AbstractFloat}
    # Fase Nube
    topologia::TopologiaOptima{T}
    precision_topologia::T
    tiempo_fase_nube::Float64

    # Fase División
    mapa_soluciones::Union{MapaDeSoluciones{T}, Nothing}
    precisiones_subredes::Union{Vector{T}, Nothing}
    tiempo_fase_division::Union{Float64, Nothing}

    # Global
    tiempo_total::Float64
    config_utilizada::Configuracion_RCND{T}
    umbral_alcanzado::Bool
    error_fase_division::Union{String, Nothing}
end
