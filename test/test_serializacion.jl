# test_serializacion.jl — Tests unitarios de serializar/deserializar JSON
# Valida: Requisitos 5.1, 5.2, 5.3, 5.4

using Test
using JSON3

# ─── Stub de dependencias ficticias ───────────────────────────────────────────
module _StubsSerializacion
    using JSON3

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

    include(joinpath(@__DIR__, "..", "src", "tipos.jl"))
    include(joinpath(@__DIR__, "..", "src", "serializacion.jl"))
end

const Informe_S = _StubsSerializacion.Informe_RCND
const TopologiaOptima_S = _StubsSerializacion.TopologiaOptima
const Configuracion_S = _StubsSerializacion.Configuracion_RCND
const MapaDeSoluciones_S = _StubsSerializacion.MapaDeSoluciones
const ErrorDeserializacion_S = _StubsSerializacion.ErrorDeserializacion
const serializar_S = _StubsSerializacion.serializar
const deserializar_S = _StubsSerializacion.deserializar

# ─── Helpers ──────────────────────────────────────────────────────────────────

"""Crea una TopologiaOptima simple para tests."""
function _crear_topo(capas::Vector{Int}; T=Float64)
    n = length(capas) - 1
    pesos = [randn(T, capas[i+1], capas[i]) for i in 1:n]
    biases = [randn(T, capas[i+1]) for i in 1:n]
    activaciones = [[:sigmoid, :relu, :identity][mod1(i, 3)] for i in 1:n]
    TopologiaOptima_S{T}(capas, pesos, biases, activaciones)
end

"""Crea un Informe_RCND completo (pipeline exitoso con mapa de soluciones)."""
function _crear_informe_completo(; T=Float64)
    topo = _crear_topo([4, 10, 3]; T=T)
    config = Configuracion_S{T}(semilla=42)
    mapa = MapaDeSoluciones_S{T}(T[0.95, 0.92, 0.88])
    Informe_S{T}(
        topo, T(0.95), 1.234,
        mapa, T[0.95, 0.92, 0.88], 0.567,
        1.801, config, true, nothing,
    )
end

"""Crea un Informe_RCND parcial (umbral no alcanzado, sin mapa)."""
function _crear_informe_parcial(; T=Float64)
    topo = _crear_topo([4, 10, 3]; T=T)
    config = Configuracion_S{T}()
    Informe_S{T}(
        topo, T(0.42), 2.5,
        nothing, nothing, nothing,
        2.5, config, false, nothing,
    )
end

"""Crea un Informe_RCND con error en Fase_Division."""
function _crear_informe_con_error(; T=Float64)
    topo = _crear_topo([4, 10, 3]; T=T)
    config = Configuracion_S{T}()
    Informe_S{T}(
        topo, T(0.95), 1.0,
        nothing, nothing, 0.1,
        1.1, config, true,
        "Error simulado en Fase_Division",
    )
end


# ─── Tests de serializar (Req 5.1) ───────────────────────────────────────────

@testset "serializar" begin

    @testset "Retorna String JSON válido" begin
        informe = _crear_informe_completo()
        json = serializar_S(informe)
        @test json isa String
        @test !isempty(json)
        # Debe ser JSON parseable
        parsed = JSON3.read(json)
        @test parsed isa JSON3.Object
    end

    @testset "JSON contiene todos los campos del informe completo" begin
        informe = _crear_informe_completo()
        json = serializar_S(informe)
        parsed = JSON3.read(json)

        @test haskey(parsed, "topologia")
        @test haskey(parsed, "precision_topologia")
        @test haskey(parsed, "tiempo_fase_nube")
        @test haskey(parsed, "mapa_soluciones")
        @test haskey(parsed, "precisiones_subredes")
        @test haskey(parsed, "tiempo_fase_division")
        @test haskey(parsed, "tiempo_total")
        @test haskey(parsed, "config_utilizada")
        @test haskey(parsed, "umbral_alcanzado")
        @test haskey(parsed, "error_fase_division")
    end

    @testset "Matrices serializadas como arrays anidados" begin
        informe = _crear_informe_completo()
        json = serializar_S(informe)
        parsed = JSON3.read(json)

        topo = parsed["topologia"]
        pesos = topo["pesos"]
        # Cada peso es un array de arrays (filas)
        @test pesos isa JSON3.Array
        @test length(pesos) > 0
        @test pesos[1] isa JSON3.Array  # Primera matriz
        @test pesos[1][1] isa JSON3.Array  # Primera fila
    end

    @testset "Campos nothing serializados como null" begin
        informe = _crear_informe_parcial()
        json = serializar_S(informe)
        parsed = JSON3.read(json)

        @test parsed["mapa_soluciones"] === nothing
        @test parsed["precisiones_subredes"] === nothing
        @test parsed["tiempo_fase_division"] === nothing
        @test parsed["error_fase_division"] === nothing
    end

    @testset "Funciona con Float32" begin
        informe = _crear_informe_completo(; T=Float32)
        json = serializar_S(informe)
        @test json isa String
        parsed = JSON3.read(json)
        @test parsed["tipo_numerico"] == "Float32"
    end

    @testset "Config serializada sin callback" begin
        informe = _crear_informe_completo()
        json = serializar_S(informe)
        parsed = JSON3.read(json)

        config = parsed["config_utilizada"]
        @test haskey(config, "n_redes_por_nube")
        @test haskey(config, "umbral_precision")
        @test haskey(config, "tamano_mini_batch")
        @test haskey(config, "semilla")
        @test config["semilla"] == 42
    end
end

# ─── Tests de deserializar (Req 5.2, 5.4) ────────────────────────────────────

@testset "deserializar" begin

    @testset "Ida y vuelta: informe completo (Req 5.3)" begin
        original = _crear_informe_completo()
        json = serializar_S(original)
        reconstruido = deserializar_S(Informe_S{Float64}, json)

        @test reconstruido isa Informe_S{Float64}
        @test reconstruido.precision_topologia ≈ original.precision_topologia
        @test reconstruido.tiempo_fase_nube ≈ original.tiempo_fase_nube
        @test reconstruido.tiempo_total ≈ original.tiempo_total
        @test reconstruido.umbral_alcanzado == original.umbral_alcanzado
        @test reconstruido.error_fase_division === original.error_fase_division
    end

    @testset "Ida y vuelta: topología preservada" begin
        original = _crear_informe_completo()
        json = serializar_S(original)
        reconstruido = deserializar_S(Informe_S{Float64}, json)

        @test reconstruido.topologia.capas == original.topologia.capas
        @test reconstruido.topologia.activaciones == original.topologia.activaciones
        for i in eachindex(original.topologia.pesos)
            @test reconstruido.topologia.pesos[i] ≈ original.topologia.pesos[i]
        end
        for i in eachindex(original.topologia.biases)
            @test reconstruido.topologia.biases[i] ≈ original.topologia.biases[i]
        end
    end

    @testset "Ida y vuelta: informe parcial (nothing fields)" begin
        original = _crear_informe_parcial()
        json = serializar_S(original)
        reconstruido = deserializar_S(Informe_S{Float64}, json)

        @test reconstruido.mapa_soluciones === nothing
        @test reconstruido.precisiones_subredes === nothing
        @test reconstruido.tiempo_fase_division === nothing
        @test reconstruido.umbral_alcanzado == false
    end

    @testset "Ida y vuelta: informe con error" begin
        original = _crear_informe_con_error()
        json = serializar_S(original)
        reconstruido = deserializar_S(Informe_S{Float64}, json)

        @test reconstruido.error_fase_division == original.error_fase_division
        @test reconstruido.mapa_soluciones === nothing
    end

    @testset "Ida y vuelta: mapa de soluciones preservado" begin
        original = _crear_informe_completo()
        json = serializar_S(original)
        reconstruido = deserializar_S(Informe_S{Float64}, json)

        @test reconstruido.mapa_soluciones !== nothing
        @test reconstruido.mapa_soluciones.data ≈ original.mapa_soluciones.data
        @test reconstruido.precisiones_subredes ≈ original.precisiones_subredes
    end

    @testset "Ida y vuelta: config preservada" begin
        original = _crear_informe_completo()
        json = serializar_S(original)
        reconstruido = deserializar_S(Informe_S{Float64}, json)

        rc = reconstruido.config_utilizada
        oc = original.config_utilizada
        @test rc.n_redes_por_nube == oc.n_redes_por_nube
        @test rc.umbral_precision ≈ oc.umbral_precision
        @test rc.max_iteraciones_reduccion == oc.max_iteraciones_reduccion
        @test rc.umbral_division ≈ oc.umbral_division
        @test rc.max_epocas == oc.max_epocas
        @test rc.tasa_aprendizaje ≈ oc.tasa_aprendizaje
        @test rc.tamano_mini_batch == oc.tamano_mini_batch
        @test rc.semilla == oc.semilla
        @test rc.verbosidad == oc.verbosidad
        @test rc.backend_computo == oc.backend_computo
        @test rc.umbral_gpu == oc.umbral_gpu
    end

    @testset "Ida y vuelta con Float32" begin
        original = _crear_informe_completo(; T=Float32)
        json = serializar_S(original)
        reconstruido = deserializar_S(Informe_S{Float32}, json)

        @test reconstruido isa Informe_S{Float32}
        @test reconstruido.precision_topologia isa Float32
        @test reconstruido.topologia.pesos[1] isa Matrix{Float32}
    end

    @testset "ErrorDeserializacion: JSON inválido (Req 5.4)" begin
        @test_throws ErrorDeserializacion_S deserializar_S(Informe_S{Float64}, "esto no es json")
    end

    @testset "ErrorDeserializacion: JSON vacío" begin
        @test_throws ErrorDeserializacion_S deserializar_S(Informe_S{Float64}, "")
    end

    @testset "ErrorDeserializacion: campo topologia faltante (Req 5.4)" begin
        json = JSON3.write(Dict("precision_topologia" => 0.5))
        err = try
            deserializar_S(Informe_S{Float64}, json)
            nothing
        catch e
            e
        end
        @test err isa ErrorDeserializacion_S
        @test err.campo == "topologia"
    end

    @testset "ErrorDeserializacion: campo config_utilizada faltante" begin
        # Minimal valid topologia but missing config
        topo_dict = Dict(
            "capas" => [2, 1],
            "pesos" => [[[0.5, 0.3]]],
            "biases" => [[0.1]],
            "activaciones" => ["sigmoid"],
        )
        json = JSON3.write(Dict(
            "topologia" => topo_dict,
            "precision_topologia" => 0.9,
            "tiempo_fase_nube" => 1.0,
            "tiempo_total" => 1.0,
            "umbral_alcanzado" => true,
        ))
        err = try
            deserializar_S(Informe_S{Float64}, json)
            nothing
        catch e
            e
        end
        @test err isa ErrorDeserializacion_S
        @test err.campo == "config_utilizada"
    end

    @testset "ErrorDeserializacion: campo pesos faltante en topologia" begin
        topo_dict = Dict(
            "capas" => [2, 1],
            "biases" => [[0.1]],
            "activaciones" => ["sigmoid"],
        )
        json = JSON3.write(Dict(
            "topologia" => topo_dict,
            "precision_topologia" => 0.9,
            "tiempo_fase_nube" => 1.0,
            "tiempo_total" => 1.0,
            "umbral_alcanzado" => true,
            "config_utilizada" => Dict(),
        ))
        err = try
            deserializar_S(Informe_S{Float64}, json)
            nothing
        catch e
            e
        end
        @test err isa ErrorDeserializacion_S
        @test err.campo == "pesos"
    end
end
