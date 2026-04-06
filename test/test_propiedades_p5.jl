# Feature: neural-cloud-division, Property 5: Determinismo con semilla
# **Valida: Requisito 2.4**

using Test
using Random
using Supposition
using Supposition.Data

# ─── Stub de dependencias ─────────────────────────────────────────────────────
module _StubsP5
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

        """Stub de buscar_topologia que retorna una red simple determinista."""
        function buscar_topologia(datos_x, datos_y;
                n_redes::Int=100,
                activaciones::Vector{Symbol}=[:sigmoid],
                umbral::Real=0.95,
                max_iteraciones::Int=10,
                rng::AbstractRNG=Random.default_rng())
            n_in = size(datos_x, 1)
            n_out = size(datos_y, 1)
            # Generar pesos deterministas usando el rng proporcionado
            pesos = [randn(rng, n_out, n_in)]
            biases = [randn(rng, n_out)]
            RedNeuronal(
                [n_in, n_out],
                pesos,
                biases,
                [activaciones[1]],
                min(Float64(umbral), 0.98)  # Simula precisión cercana al umbral
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

const Config_p5 = _StubsP5.Configuracion_RCND
const Motor_p5 = _StubsP5.Motor_RCND
const ejecutar_pipeline_p5 = _StubsP5.ejecutar_pipeline

# ─── Generadores ──────────────────────────────────────────────────────────────

# Semilla: entero positivo
const gen_semilla_p5 = Data.Integers(1, 10_000_000)

# Dimensiones del dataset
const gen_n_features_p5 = Data.Integers(2, 10)
const gen_n_samples_p5 = Data.Integers(5, 50)
const gen_n_outputs_p5 = Data.Integers(1, 3)

# ─── Test de propiedad ────────────────────────────────────────────────────────

@testset "P5: Determinismo con semilla" begin

    @check max_examples=100 function p5_determinismo_con_semilla(
        semilla     = gen_semilla_p5,
        n_features  = gen_n_features_p5,
        n_samples   = gen_n_samples_p5,
        n_outputs   = gen_n_outputs_p5,
    )
        # Generar datos deterministas a partir de la semilla
        data_rng = Random.MersenneTwister(semilla + 999_999)
        datos_x = randn(data_rng, Float64, n_features, n_samples)
        datos_y = randn(data_rng, Float64, n_outputs, n_samples)

        # Config con semilla fija y umbral bajo para que el stub lo alcance
        config = Config_p5{Float64}(
            semilla=semilla,
            umbral_precision=0.5,
        )

        # Ejecución 1
        motor1 = Motor_p5(config, copy(datos_x), copy(datos_y))
        informe1 = ejecutar_pipeline_p5(motor1)

        # Ejecución 2 con misma config y datos
        motor2 = Motor_p5(config, copy(datos_x), copy(datos_y))
        informe2 = ejecutar_pipeline_p5(motor2)

        # Verificar topologías idénticas: mismos pesos
        length(informe1.topologia.pesos) == length(informe2.topologia.pesos) || return false
        for i in eachindex(informe1.topologia.pesos)
            informe1.topologia.pesos[i] == informe2.topologia.pesos[i] || return false
        end

        # Mismos biases
        length(informe1.topologia.biases) == length(informe2.topologia.biases) || return false
        for i in eachindex(informe1.topologia.biases)
            informe1.topologia.biases[i] == informe2.topologia.biases[i] || return false
        end

        # Mismas capas y activaciones
        informe1.topologia.capas == informe2.topologia.capas || return false
        informe1.topologia.activaciones == informe2.topologia.activaciones || return false

        # Misma precisión
        informe1.precision_topologia == informe2.precision_topologia || return false

        # Mismo umbral_alcanzado
        informe1.umbral_alcanzado == informe2.umbral_alcanzado || return false

        # Si ambos alcanzaron el umbral, verificar mapa de soluciones idéntico
        if informe1.umbral_alcanzado && informe2.umbral_alcanzado
            (informe1.mapa_soluciones !== nothing && informe2.mapa_soluciones !== nothing) || return false
            informe1.mapa_soluciones.data == informe2.mapa_soluciones.data || return false
            informe1.precisiones_subredes == informe2.precisiones_subredes || return false
        end

        true
    end
end
