# Feature: neural-cloud-division, Property 11: Clampeo de mini-batch
# **Valida: Requisito 9.2**

using Test
using Supposition
using Supposition.Data

# ─── Stub de dependencias ─────────────────────────────────────────────────────
module _StubsP11
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

const Motor_RCND_p11 = _StubsP11.Motor_RCND
const Config_p11 = _StubsP11.Configuracion_RCND

# ─── Generadores ──────────────────────────────────────────────────────────────

# n_samples: 1-50
const gen_n_samples_p11 = Data.Integers(1, 50)

# Offset positivo para que batch_size > n_samples
const gen_offset_p11 = Data.Integers(1, 200)

# n_features: 1-10
const gen_n_features_p11 = Data.Integers(1, 10)

# n_outputs: 1-5
const gen_n_outputs_p11 = Data.Integers(1, 5)

# ─── Test de propiedad ────────────────────────────────────────────────────────

@testset "P11: Clampeo de mini-batch" begin

    @check max_examples=100 function p11_clampeo_mini_batch(
        n_samples  = gen_n_samples_p11,
        offset     = gen_offset_p11,
        n_features = gen_n_features_p11,
        n_outputs  = gen_n_outputs_p11,
    )
        batch_size = n_samples + offset  # Garantiza batch_size > n_samples

        config = Config_p11{Float64}(tamano_mini_batch=batch_size)
        datos_x = zeros(Float64, n_features, n_samples)
        datos_y = zeros(Float64, n_outputs, n_samples)

        motor = Motor_RCND_p11(config, datos_x, datos_y)

        # El tamaño efectivo de batch (columnas del buffer) debe ser n_samples
        size(motor.buffer_activacion, 2) == n_samples || return false
        size(motor.buffer_gradiente, 2) == n_samples || return false

        true
    end
end
