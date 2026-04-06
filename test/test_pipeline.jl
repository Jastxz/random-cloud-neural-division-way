# test_pipeline.jl — Tests de integración del pipeline completo
# Valida: Requisitos 2.1 (pipeline XOR), 7.1, 7.2 (fases individuales), 8.3 (sin callback)

using Test
using Random

# ─── Stub de dependencias ─────────────────────────────────────────────────────
module _StubsPipeline
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

    # Stub de MapaDeSoluciones (DivisionNeuronal.jl)
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
                pesos, biases,
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

const Motor_P = _StubsPipeline.Motor_RCND
const Config_P = _StubsPipeline.Configuracion_RCND
const ejecutar_pipeline_P = _StubsPipeline.ejecutar_pipeline
const ejecutar_fase_nube_P = _StubsPipeline.ejecutar_fase_nube
const ejecutar_fase_division_P = _StubsPipeline.ejecutar_fase_division
const RedBase_P = _StubsPipeline.RedBase
const Informe_P = _StubsPipeline.Informe_RCND
const TopologiaOptima_P = _StubsPipeline.TopologiaOptima
const MapaDeSoluciones_P = _StubsPipeline.MapaDeSoluciones

# ─── Helpers: datos XOR ───────────────────────────────────────────────────────

"""Crea un dataset XOR simple: 4 muestras, 2 features, 1 output."""
function crear_datos_xor(::Type{T}=Float64) where {T}
    # XOR: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0
    datos_x = T[0 0 1 1;
                0 1 0 1]   # (2, 4)
    datos_y = T[0 1 1 0]   # (1, 4) — reshape to matrix
    datos_y = reshape(datos_y, 1, 4)
    return datos_x, datos_y
end

"""Crea una RedBase{T} simple para tests de fase_division."""
function crear_red_base_simple(::Type{T}, n_in::Int, n_out::Int) where {T}
    pesos = [randn(T, n_out, n_in)]
    biases = [randn(T, n_out)]
    RedBase_P{T}([n_in, n_out], pesos, biases, [:sigmoid])
end

# ─── Tests ────────────────────────────────────────────────────────────────────

@testset "Pipeline — integración" begin

    # ─── Pipeline completo con datos XOR (Req 2.1) ───────────────────────────

    @testset "Pipeline completo con datos XOR (Req 2.1)" begin
        datos_x, datos_y = crear_datos_xor()
        config = Config_P{Float64}(
            semilla=42,
            umbral_precision=0.5,
            tamano_mini_batch=4,
        )
        motor = Motor_P(config, datos_x, datos_y)

        informe = ejecutar_pipeline_P(motor)

        @test informe isa Informe_P{Float64}
        @test informe.topologia isa TopologiaOptima_P{Float64}
        @test informe.topologia.capas[1] == 2   # 2 features de entrada
        @test informe.topologia.capas[end] == 1  # 1 output
        @test informe.precision_topologia >= 0.0
        @test informe.tiempo_fase_nube >= 0.0
        @test informe.tiempo_total >= 0.0
        @test informe.config_utilizada === config
        @test informe.umbral_alcanzado == true
        @test informe.mapa_soluciones !== nothing
    end

    @testset "Pipeline XOR con Float32" begin
        datos_x, datos_y = crear_datos_xor(Float32)
        config = Config_P{Float32}(semilla=42, umbral_precision=Float32(0.5))
        motor = Motor_P(config, datos_x, datos_y)

        informe = ejecutar_pipeline_P(motor)

        @test informe isa Informe_P{Float32}
        @test informe.topologia.pesos[1] isa Matrix{Float32}
    end

    # ─── Fases individuales (Req 7.1, 7.2) ───────────────────────────────────

    @testset "ejecutar_fase_nube individual (Req 7.1)" begin
        datos_x, datos_y = crear_datos_xor()
        config = Config_P{Float64}(semilla=42, umbral_precision=0.9)
        motor = Motor_P(config, datos_x, datos_y)

        red, metricas = ejecutar_fase_nube_P(motor)

        @test red isa _StubsPipeline.RedNeuronal
        @test length(red.capas) >= 2
        @test red.capas[1] == 2   # n_features
        @test red.capas[end] == 1 # n_outputs
        @test metricas.precision >= 0.0
        @test metricas.tiempo >= 0.0
        @test motor.tiempo_fase_nube >= 0.0
    end

    @testset "ejecutar_fase_division individual (Req 7.2)" begin
        datos_x, datos_y = crear_datos_xor()
        config = Config_P{Float64}(semilla=42)
        motor = Motor_P(config, datos_x, datos_y)

        red = crear_red_base_simple(Float64, 2, 1)
        mapa, metricas = ejecutar_fase_division_P(motor, red)

        @test mapa isa MapaDeSoluciones_P{Float64}
        @test metricas.tiempo >= 0.0
        @test motor.tiempo_fase_division >= 0.0
    end

    @testset "ejecutar_fase_division con red externa multicapa (Req 7.3)" begin
        datos_x, datos_y = crear_datos_xor()
        config = Config_P{Float64}(semilla=42)
        motor = Motor_P(config, datos_x, datos_y)

        # Red externa con arquitectura diferente a los datos
        red_externa = RedBase_P{Float64}(
            [2, 8, 4, 1],
            [randn(8, 2), randn(4, 8), randn(1, 4)],
            [randn(8), randn(4), randn(1)],
            [:relu, :relu, :sigmoid],
        )

        mapa, metricas = ejecutar_fase_division_P(motor, red_externa)
        @test mapa isa MapaDeSoluciones_P{Float64}
        @test metricas.tiempo >= 0.0
    end

    # ─── Pipeline sin callback (Req 8.3) ─────────────────────────────────────

    @testset "Pipeline sin callback ejecuta sin error (Req 8.3)" begin
        datos_x, datos_y = crear_datos_xor()
        config = Config_P{Float64}(
            semilla=42,
            umbral_precision=0.5,
            callback_fase=nothing,
        )
        motor = Motor_P(config, datos_x, datos_y)

        informe = ejecutar_pipeline_P(motor)

        @test informe isa Informe_P{Float64}
        @test informe.umbral_alcanzado == true
        @test informe.error_fase_division === nothing
    end

    # ─── Pipeline con callback ────────────────────────────────────────────────

    @testset "Pipeline con callback registra todas las fases" begin
        registro = []
        callback = (args...) -> push!(registro, args)

        datos_x, datos_y = crear_datos_xor()
        config = Config_P{Float64}(
            semilla=42,
            umbral_precision=0.5,
            callback_fase=callback,
        )
        motor = Motor_P(config, datos_x, datos_y)

        informe = ejecutar_pipeline_P(motor)

        # Debe haber 4 invocaciones: inicio/fin de cada fase
        @test length(registro) == 4
        @test registro[1] == (:fase_nube, :iniciada)
        @test registro[2][1] == :fase_nube
        @test registro[2][2] == :completada
        @test registro[3] == (:fase_division, :iniciada)
        @test registro[4][1] == :fase_division
        @test registro[4][2] == :completada
    end

    # ─── Determinismo con semilla ─────────────────────────────────────────────

    @testset "Pipeline determinista con misma semilla (Req 2.4)" begin
        datos_x, datos_y = crear_datos_xor()
        config = Config_P{Float64}(semilla=99, umbral_precision=0.5)

        motor1 = Motor_P(config, datos_x, datos_y)
        motor2 = Motor_P(config, datos_x, datos_y)

        informe1 = ejecutar_pipeline_P(motor1)
        informe2 = ejecutar_pipeline_P(motor2)

        @test informe1.topologia.pesos == informe2.topologia.pesos
        @test informe1.topologia.biases == informe2.topologia.biases
        @test informe1.precision_topologia == informe2.precision_topologia
    end
end
