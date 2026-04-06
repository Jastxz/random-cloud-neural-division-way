# Feature: neural-cloud-division, Property 13: Invariantes de ejecución en GPU
# **Valida: Requisitos 10.6, 10.7**

using Test
using Random
using Supposition
using Supposition.Data

# ─── Stub de dependencias ─────────────────────────────────────────────────────
module _StubsP13
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

const Config_p13 = _StubsP13.Configuracion_RCND
const Motor_p13 = _StubsP13.Motor_RCND
const ejecutar_pipeline_p13 = _StubsP13.ejecutar_pipeline
const TopologiaOptima_p13 = _StubsP13.TopologiaOptima

# ─── Generadores ──────────────────────────────────────────────────────────────

const gen_semilla_p13 = Data.Integers(1, 10_000_000)
const gen_n_features_p13 = Data.Integers(2, 10)
const gen_n_samples_p13 = Data.Integers(5, 50)
const gen_n_outputs_p13 = Data.Integers(1, 3)

# ─── Tests de propiedad ──────────────────────────────────────────────────────

@testset "P13: Invariantes de ejecución en GPU" begin

    # CPU path: verify that when backend is :cpu, the final Informe_RCND
    # contains regular Array types (not GPU arrays), and pesos/biases are
    # Matrix{T}/Vector{T}.
    @testset "CPU path: Informe contiene Array (CPU), no CuArray" begin
        @check max_examples=100 function p13_cpu_arrays(
            semilla     = gen_semilla_p13,
            n_features  = gen_n_features_p13,
            n_samples   = gen_n_samples_p13,
            n_outputs   = gen_n_outputs_p13,
        )
            data_rng = Random.MersenneTwister(semilla + 131313)
            datos_x = randn(data_rng, Float64, n_features, n_samples)
            datos_y = randn(data_rng, Float64, n_outputs, n_samples)

            # Config con backend :cpu y umbral bajo para completar ambas fases
            config = Config_p13{Float64}(
                semilla=semilla,
                umbral_precision=0.5,
                backend_computo=:cpu,
            )

            motor = Motor_p13(config, copy(datos_x), copy(datos_y))
            informe = ejecutar_pipeline_p13(motor)

            # Verificar que topología pesos son Array (Matrix{Float64}), no CuArray
            for p in informe.topologia.pesos
                p isa Matrix{Float64} || return false
            end

            # Verificar que topología biases son Array (Vector{Float64}), no CuArray
            for b in informe.topologia.biases
                b isa Vector{Float64} || return false
            end

            # Verificar que capas es Vector{Int} estándar
            informe.topologia.capas isa Vector{Int} || return false

            # Verificar que activaciones es Vector{Symbol} estándar
            informe.topologia.activaciones isa Vector{Symbol} || return false

            # Verificar que precisiones_subredes (si existe) es Vector estándar
            if informe.precisiones_subredes !== nothing
                informe.precisiones_subredes isa Vector{Float64} || return false
            end

            true
        end
    end

    # GPU path: skipped when no GPU is available.
    # When a real GPU is present, this would verify Float32 intermediates
    # and Array (CPU) in the final Informe_RCND.
    @testset "GPU path: Float32 intermedios y Array final (skip sin GPU)" begin
        has_gpu = try
            _StubsP13.gpu_disponible()
        catch
            false
        end

        if !has_gpu
            @test_skip "GPU no disponible — se omite test de invariantes GPU reales"
        else
            @check max_examples=100 function p13_gpu_invariantes(
                semilla     = gen_semilla_p13,
                n_features  = gen_n_features_p13,
                n_samples   = gen_n_samples_p13,
                n_outputs   = gen_n_outputs_p13,
            )
                data_rng = Random.MersenneTwister(semilla + 131313)
                datos_x = randn(data_rng, Float64, n_features, n_samples)
                datos_y = randn(data_rng, Float64, n_outputs, n_samples)

                # Config con backend :gpu forzado
                config = Config_p13{Float32}(
                    semilla=semilla,
                    umbral_precision=Float32(0.5),
                    backend_computo=:gpu,
                )

                motor = Motor_p13(config, Float32.(datos_x), Float32.(datos_y))
                informe = ejecutar_pipeline_p13(motor)

                # Verificar que pesos finales son Array (CPU), no CuArray
                for p in informe.topologia.pesos
                    p isa Matrix{Float32} || return false
                    # Must be regular Array, not a GPU array
                    p isa Array || return false
                end

                # Verificar que biases finales son Array (CPU), no CuArray
                for b in informe.topologia.biases
                    b isa Vector{Float32} || return false
                    b isa Array || return false
                end

                # Verificar tipo Float32 en la topología
                informe.precision_topologia isa Float32 || return false

                true
            end
        end
    end

end
