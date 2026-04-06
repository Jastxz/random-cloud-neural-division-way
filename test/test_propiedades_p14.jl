# Feature: neural-cloud-division, Property 14: Fase_Division acepta RedBase externa
# **Valida: Requisito 7.3**

using Test
using Random
using Supposition
using Supposition.Data

# ─── Stub de dependencias ─────────────────────────────────────────────────────
module _StubsP14
    using Random

    # Stub mínimo de KernelAbstractions
    module KernelAbstractions
        struct CPU end
        function get_backend(arr)
            return CPU()
        end
        function allocate(backend, T, dims...)
            return zeros(T, dims...)
        end
    end

    # Stub de MapaDeSoluciones (requerido por tipos.jl → Informe_RCND)
    struct MapaDeSoluciones{T<:AbstractFloat}
        data::Vector{T}
    end

    # Stub de RedBase (DivisionNeuronal.jl)
    struct RedBase{T<:AbstractFloat}
        capas::Vector{Int}
        pesos::Vector{Matrix{T}}
        biases::Vector{Vector{T}}
        activaciones::Vector{Symbol}
    end

    # Stub de RedNeuronal (RandomCloud.jl)
    struct RedNeuronal
        capas::Vector{Int}
        pesos::Vector{<:AbstractMatrix}
        biases::Vector{<:AbstractVector}
        activaciones::Vector{Symbol}
        precision::Float64
    end

    # Stub de RandomCloud module
    module RandomCloud
        using Random: AbstractRNG
        import ..RedNeuronal

        function buscar_topologia(datos_x, datos_y;
                n_redes::Int=100,
                activaciones::Vector{Symbol}=[:sigmoid],
                umbral::Real=0.95,
                max_iteraciones::Int=10,
                rng::AbstractRNG=Random.default_rng())
            n_in = size(datos_x, 1)
            n_out = size(datos_y, 1)
            pesos = [randn(rng, n_out, n_in)]
            biases = [randn(rng, n_out)]
            RedNeuronal(
                [n_in, n_out],
                pesos,
                biases,
                [activaciones[1]],
                min(Float64(umbral), 0.98)
            )
        end
    end

    # Stub de DivisionNeuronal module
    module DivisionNeuronal
        using Random: AbstractRNG
        import ..RedBase, ..MapaDeSoluciones

        """Stub de dividir_red que retorna un MapaDeSoluciones simple determinista."""
        function dividir_red(red::RedBase{T}, datos_x, datos_y;
                umbral::Real=0.5,
                max_epocas::Int=100,
                tasa_aprendizaje::Real=0.001,
                tamano_mini_batch::Int=32,
                rng::AbstractRNG=Random.default_rng()) where {T}
            n_out = size(datos_y, 1)
            data = T[randn(rng, T) for _ in 1:n_out]
            MapaDeSoluciones{T}(data)
        end
    end

    include(joinpath(@__DIR__, "..", "src", "tipos.jl"))
    include(joinpath(@__DIR__, "..", "src", "validacion.jl"))
    include(joinpath(@__DIR__, "..", "src", "buffers.jl"))
    include(joinpath(@__DIR__, "..", "src", "backend.jl"))
    include(joinpath(@__DIR__, "..", "src", "adaptador.jl"))
    include(joinpath(@__DIR__, "..", "src", "motor.jl"))
end

const Config_p14 = _StubsP14.Configuracion_RCND
const Motor_p14 = _StubsP14.Motor_RCND
const RedBase_p14 = _StubsP14.RedBase
const MapaDeSoluciones_p14 = _StubsP14.MapaDeSoluciones
const ejecutar_fase_division_p14 = _StubsP14.ejecutar_fase_division

# ─── Generadores ──────────────────────────────────────────────────────────────

# Número de capas de la red (2-5 capas = 1-4 conexiones)
const gen_n_capas_p14 = Data.Integers(2, 5)

# Neuronas por capa (1-20)
const gen_neuronas_p14 = Data.Integers(1, 20)

# Semilla para generar pesos aleatorios
const gen_semilla_p14 = Data.Integers(1, 10_000_000)

# Dimensiones del dataset (deben ser consistentes con la capa de entrada de la red)
const gen_n_samples_p14 = Data.Integers(5, 50)
const gen_n_outputs_p14 = Data.Integers(1, 5)

# Activaciones posibles
const ACTIVACIONES_P14 = [:sigmoid, :relu, :identity]

# ─── Test de propiedad ────────────────────────────────────────────────────────

@testset "P14: Fase_Division acepta RedBase externa" begin

    @check max_examples=100 function p14_fase_division_acepta_redbase_externa(
        n_capas    = gen_n_capas_p14,
        semilla    = gen_semilla_p14,
        n_samples  = gen_n_samples_p14,
        n_outputs  = gen_n_outputs_p14,
    )
        rng = Random.MersenneTwister(semilla)

        # Generar arquitectura aleatoria: n_capas capas con 1-20 neuronas cada una
        capas = [rand(rng, 1:20) for _ in 1:n_capas]

        # Generar pesos y biases aleatorios consistentes con la arquitectura
        n_conexiones = n_capas - 1
        pesos = [randn(rng, Float64, capas[i+1], capas[i]) for i in 1:n_conexiones]
        biases = [randn(rng, Float64, capas[i+1]) for i in 1:n_conexiones]

        # Generar activaciones aleatorias por conexión
        activaciones = [ACTIVACIONES_P14[rand(rng, 1:3)] for _ in 1:n_conexiones]

        # Construir RedBase{Float64} externamente (NO proveniente de Fase_Nube)
        red_externa = RedBase_p14{Float64}(capas, pesos, biases, activaciones)

        # Crear Motor_RCND con datos cuyas dimensiones coincidan con la capa de entrada
        n_features = capas[1]  # La capa de entrada define n_features
        datos_x = randn(rng, Float64, n_features, n_samples)
        datos_y = randn(rng, Float64, n_outputs, n_samples)

        config = Config_p14{Float64}(
            semilla=semilla,
            umbral_precision=0.5,
        )
        motor = Motor_p14(config, datos_x, datos_y)

        # Ejecutar fase_division con la red externa
        resultado = ejecutar_fase_division_p14(motor, red_externa)

        # Verificar que retorna una tupla (MapaDeSoluciones, MetricasFase)
        resultado isa Tuple || return false
        length(resultado) == 2 || return false

        mapa, metricas = resultado

        # Verificar que el mapa es un MapaDeSoluciones válido
        mapa isa MapaDeSoluciones_p14{Float64} || return false

        # Verificar que las métricas tienen campos esperados
        hasfield(typeof(metricas), :precision) || return false
        hasfield(typeof(metricas), :tiempo) || return false
        metricas.tiempo >= 0.0 || return false

        true
    end
end
