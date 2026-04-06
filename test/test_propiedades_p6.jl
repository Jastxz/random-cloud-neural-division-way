# Feature: neural-cloud-division, Property 6: Invariante de tiempos de ejecución
# **Valida: Requisito 2.5**

using Test
using Random
using Supposition
using Supposition.Data

# ─── Stub de dependencias ─────────────────────────────────────────────────────
module _StubsP6
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

const Config_p6 = _StubsP6.Configuracion_RCND
const Motor_p6 = _StubsP6.Motor_RCND
const ejecutar_pipeline_p6 = _StubsP6.ejecutar_pipeline

# ─── Generadores (reutilizados de P5) ────────────────────────────────────────

const gen_semilla_p6 = Data.Integers(1, 10_000_000)
const gen_n_features_p6 = Data.Integers(2, 10)
const gen_n_samples_p6 = Data.Integers(5, 50)
const gen_n_outputs_p6 = Data.Integers(1, 3)

# ─── Test de propiedad ────────────────────────────────────────────────────────

@testset "P6: Invariante de tiempos de ejecución" begin

    @check max_examples=100 function p6_invariante_tiempos(
        semilla     = gen_semilla_p6,
        n_features  = gen_n_features_p6,
        n_samples   = gen_n_samples_p6,
        n_outputs   = gen_n_outputs_p6,
    )
        # Generar datos deterministas
        data_rng = Random.MersenneTwister(semilla + 777_777)
        datos_x = randn(data_rng, Float64, n_features, n_samples)
        datos_y = randn(data_rng, Float64, n_outputs, n_samples)

        # Config con umbral bajo para que el pipeline complete ambas fases
        config = Config_p6{Float64}(
            semilla=semilla,
            umbral_precision=0.5,
        )

        motor = Motor_p6(config, copy(datos_x), copy(datos_y))
        informe = ejecutar_pipeline_p6(motor)

        # Verificar: tiempo_fase_nube >= 0
        informe.tiempo_fase_nube >= 0.0 || return false

        # Verificar: tiempo_fase_division >= 0 (when not nothing)
        if informe.tiempo_fase_division !== nothing
            informe.tiempo_fase_division >= 0.0 || return false
        end

        # Verificar: tiempo_total >= tiempo_fase_nube + (tiempo_fase_division or 0)
        t_div = informe.tiempo_fase_division !== nothing ? informe.tiempo_fase_division : 0.0
        informe.tiempo_total >= informe.tiempo_fase_nube + t_div || return false

        true
    end
end
