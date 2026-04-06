# test_tipos.jl — Tests unitarios de tipos base, constructores y valores por defecto
# Valida: Requisitos 1.4 (valores por defecto), tipos paramétricos

using Test

# ─── Stub de dependencias ficticias ───────────────────────────────────────────
module _StubsTipos
    # Stub de MapaDeSoluciones (requerido por tipos.jl → Informe_RCND)
    struct MapaDeSoluciones{T<:AbstractFloat}
        data::Vector{T}
    end

    include(joinpath(@__DIR__, "..", "src", "tipos.jl"))
end

const Configuracion_RCND_T = _StubsTipos.Configuracion_RCND
const TopologiaOptima_T = _StubsTipos.TopologiaOptima
const Informe_RCND_T = _StubsTipos.Informe_RCND
const MapaDeSoluciones_T = _StubsTipos.MapaDeSoluciones
const ErrorConfiguracion_T = _StubsTipos.ErrorConfiguracion
const ErrorAdaptador_T = _StubsTipos.ErrorAdaptador
const ErrorBackend_T = _StubsTipos.ErrorBackend
const ErrorDeserializacion_T = _StubsTipos.ErrorDeserializacion

# ─── Tests ────────────────────────────────────────────────────────────────────

@testset "Tipos base" begin

    # ─── Configuracion_RCND — valores por defecto (Req 1.4) ──────────────────

    @testset "Configuracion_RCND — valores por defecto Float64 (Req 1.4)" begin
        config = Configuracion_RCND_T()

        @test config isa Configuracion_RCND_T{Float64}
        # Fase Nube defaults
        @test config.n_redes_por_nube == 100
        @test config.activaciones == [:sigmoid, :relu, :identity]
        @test config.umbral_precision == 0.95
        @test config.max_iteraciones_reduccion == 10
        # Fase División defaults
        @test config.umbral_division == 0.5
        @test config.max_epocas == 100
        @test config.tasa_aprendizaje ≈ 0.001
        @test config.tamano_mini_batch == 32
        # Global defaults
        @test config.semilla === nothing
        @test config.verbosidad === :normal
        # GPU defaults (Req 10.9)
        @test config.backend_computo === :auto
        @test config.umbral_gpu == 10_000
        # Callback default
        @test config.callback_fase === nothing
    end

    @testset "Configuracion_RCND{Float32} — tipos paramétricos" begin
        config = Configuracion_RCND_T{Float32}()

        @test config isa Configuracion_RCND_T{Float32}
        @test config.umbral_precision isa Float32
        @test config.umbral_division isa Float32
        @test config.tasa_aprendizaje isa Float32
        @test config.umbral_precision == Float32(0.95)
        @test config.tasa_aprendizaje ≈ Float32(0.001)
    end

    @testset "Configuracion_RCND — constructor con kwargs personalizados" begin
        config = Configuracion_RCND_T{Float64}(
            n_redes_por_nube=50,
            umbral_precision=0.8,
            max_epocas=200,
            semilla=42,
            verbosidad=:detallado,
            backend_computo=:cpu,
        )

        @test config.n_redes_por_nube == 50
        @test config.umbral_precision == 0.8
        @test config.max_epocas == 200
        @test config.semilla == 42
        @test config.verbosidad === :detallado
        @test config.backend_computo === :cpu
    end

    @testset "Configuracion_RCND — alias sin tipo infiere Float64" begin
        config = Configuracion_RCND_T(semilla=123)
        @test config isa Configuracion_RCND_T{Float64}
        @test config.semilla == 123
    end

    # ─── TopologiaOptima ──────────────────────────────────────────────────────

    @testset "TopologiaOptima{Float64} — construcción básica" begin
        capas = [4, 10, 3]
        pesos = [randn(10, 4), randn(3, 10)]
        biases = [randn(10), randn(3)]
        activaciones = [:relu, :sigmoid]

        topo = TopologiaOptima_T{Float64}(capas, pesos, biases, activaciones)

        @test topo isa TopologiaOptima_T{Float64}
        @test topo.capas == [4, 10, 3]
        @test length(topo.pesos) == 2
        @test length(topo.biases) == 2
        @test topo.activaciones == [:relu, :sigmoid]
        @test size(topo.pesos[1]) == (10, 4)
        @test size(topo.pesos[2]) == (3, 10)
    end

    @testset "TopologiaOptima{Float32} — tipos paramétricos" begin
        capas = [2, 5, 1]
        pesos = [randn(Float32, 5, 2), randn(Float32, 1, 5)]
        biases = [randn(Float32, 5), randn(Float32, 1)]
        activaciones = [:identity, :sigmoid]

        topo = TopologiaOptima_T{Float32}(capas, pesos, biases, activaciones)

        @test topo isa TopologiaOptima_T{Float32}
        @test eltype(topo.pesos[1]) == Float32
        @test eltype(topo.biases[1]) == Float32
    end

    # ─── Tipos de error ───────────────────────────────────────────────────────

    @testset "ErrorConfiguracion — construcción" begin
        err = ErrorConfiguracion_T("mensaje de prueba")
        @test err isa Exception
        @test err.mensaje == "mensaje de prueba"
    end

    @testset "ErrorAdaptador — construcción con capa" begin
        err = ErrorAdaptador_T("dimensiones incorrectas", 3)
        @test err isa Exception
        @test err.mensaje == "dimensiones incorrectas"
        @test err.capa == 3
    end

    @testset "ErrorBackend — construcción con backend" begin
        err = ErrorBackend_T("GPU no disponible", :gpu)
        @test err isa Exception
        @test err.mensaje == "GPU no disponible"
        @test err.backend === :gpu
    end

    @testset "ErrorDeserializacion — construcción con campo" begin
        err = ErrorDeserializacion_T("campo faltante", "topologia")
        @test err isa Exception
        @test err.mensaje == "campo faltante"
        @test err.campo == "topologia"
    end

    @testset "ErrorDeserializacion — campo nothing" begin
        err = ErrorDeserializacion_T("JSON inválido", nothing)
        @test err.campo === nothing
    end

    # ─── Informe_RCND ─────────────────────────────────────────────────────────

    @testset "Informe_RCND{Float64} — construcción completa" begin
        topo = TopologiaOptima_T{Float64}(
            [3, 5, 2],
            [randn(5, 3), randn(2, 5)],
            [randn(5), randn(2)],
            [:relu, :sigmoid],
        )
        config = Configuracion_RCND_T{Float64}(semilla=42)
        mapa = MapaDeSoluciones_T{Float64}([0.9, 0.85])

        informe = Informe_RCND_T{Float64}(
            topo, 0.95, 1.5,
            mapa, [0.9, 0.85], 0.8,
            2.3, config, true, nothing,
        )

        @test informe isa Informe_RCND_T{Float64}
        @test informe.topologia === topo
        @test informe.precision_topologia == 0.95
        @test informe.tiempo_fase_nube == 1.5
        @test informe.mapa_soluciones === mapa
        @test informe.precisiones_subredes == [0.9, 0.85]
        @test informe.tiempo_fase_division == 0.8
        @test informe.tiempo_total == 2.3
        @test informe.config_utilizada === config
        @test informe.umbral_alcanzado == true
        @test informe.error_fase_division === nothing
    end

    @testset "Informe_RCND — campos nothing (informe parcial)" begin
        topo = TopologiaOptima_T{Float64}(
            [2, 1], [randn(1, 2)], [randn(1)], [:sigmoid],
        )
        config = Configuracion_RCND_T{Float64}()

        informe = Informe_RCND_T{Float64}(
            topo, 0.4, 2.0,
            nothing, nothing, nothing,
            2.0, config, false, nothing,
        )

        @test informe.mapa_soluciones === nothing
        @test informe.precisiones_subredes === nothing
        @test informe.tiempo_fase_division === nothing
        @test informe.umbral_alcanzado == false
        @test informe.error_fase_division === nothing
    end
end
