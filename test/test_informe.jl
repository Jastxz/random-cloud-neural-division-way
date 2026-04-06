# test_informe.jl — Tests unitarios de resumen(informe)
# Valida: Requisito 4.5

using Test

# ─── Stub de dependencias ficticias ───────────────────────────────────────────
module _StubsInforme
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
    include(joinpath(@__DIR__, "..", "src", "informe.jl"))
end

const Informe_RCND_I = _StubsInforme.Informe_RCND
const TopologiaOptima_I = _StubsInforme.TopologiaOptima
const Configuracion_RCND_I = _StubsInforme.Configuracion_RCND
const MapaDeSoluciones_I = _StubsInforme.MapaDeSoluciones
const resumen_I = _StubsInforme.resumen

# ─── Helpers ──────────────────────────────────────────────────────────────────

"""Crea una TopologiaOptima simple para tests."""
function crear_topologia(capas::Vector{Int}; T=Float64)
    n = length(capas) - 1
    pesos = [randn(T, capas[i+1], capas[i]) for i in 1:n]
    biases = [randn(T, capas[i+1]) for i in 1:n]
    activaciones = [[:sigmoid, :relu, :identity][mod1(i, 3)] for i in 1:n]
    TopologiaOptima_I{T}(capas, pesos, biases, activaciones)
end

"""Crea una Configuracion_RCND con valores por defecto."""
crear_config(; T=Float64, kwargs...) = Configuracion_RCND_I{T}(; kwargs...)

"""Crea un Informe_RCND completo (pipeline exitoso con mapa de soluciones)."""
function crear_informe_completo(; T=Float64)
    topo = crear_topologia([4, 10, 3]; T=T)
    config = crear_config(; T=T)
    mapa = MapaDeSoluciones_I{T}(T[0.95, 0.92, 0.88])
    Informe_RCND_I{T}(
        topo,
        T(0.95),
        1.234,
        mapa,
        T[0.95, 0.92, 0.88],
        0.567,
        1.801,
        config,
        true,
        nothing,
    )
end

"""Crea un Informe_RCND parcial (umbral no alcanzado, sin mapa)."""
function crear_informe_parcial(; T=Float64)
    topo = crear_topologia([4, 10, 3]; T=T)
    config = crear_config(; T=T)
    Informe_RCND_I{T}(
        topo,
        T(0.42),
        2.5,
        nothing,
        nothing,
        nothing,
        2.5,
        config,
        false,
        nothing,
    )
end

"""Crea un Informe_RCND con error en Fase_Division."""
function crear_informe_con_error(; T=Float64)
    topo = crear_topologia([4, 10, 3]; T=T)
    config = crear_config(; T=T)
    Informe_RCND_I{T}(
        topo,
        T(0.95),
        1.0,
        nothing,
        nothing,
        0.1,
        1.1,
        config,
        true,
        "Error simulado en Fase_Division: descomposición fallida",
    )
end

# ─── Tests ────────────────────────────────────────────────────────────────────

@testset "resumen(informe)" begin

    @testset "Retorna String no vacío (Req 4.5)" begin
        informe = crear_informe_completo()
        resultado = resumen_I(informe)
        @test resultado isa String
        @test !isempty(resultado)
    end

    @testset "Contiene encabezado" begin
        informe = crear_informe_completo()
        resultado = resumen_I(informe)
        @test occursin("Informe RCND", resultado)
    end

    @testset "Contiene topología con capas y número de capas" begin
        informe = crear_informe_completo()
        resultado = resumen_I(informe)
        @test occursin("[4, 10, 3]", resultado)
        @test occursin("3 capas", resultado)
    end

    @testset "Contiene precisión de topología" begin
        informe = crear_informe_completo()
        resultado = resumen_I(informe)
        @test occursin("0.95", resultado)
    end

    @testset "Contiene indicador de umbral alcanzado (✓)" begin
        informe = crear_informe_completo()
        resultado = resumen_I(informe)
        @test occursin("✓", resultado)
        @test occursin("Umbral alcanzado", resultado)
    end

    @testset "Contiene indicador de umbral NO alcanzado (✗)" begin
        informe = crear_informe_parcial()
        resultado = resumen_I(informe)
        @test occursin("✗", resultado)
    end

    @testset "Contiene tiempos de cada fase y total" begin
        informe = crear_informe_completo()
        resultado = resumen_I(informe)
        @test occursin("Fase Nube:", resultado)
        @test occursin("1.234s", resultado)
        @test occursin("Fase División:", resultado)
        @test occursin("0.567s", resultado)
        @test occursin("Tiempo total:", resultado)
        @test occursin("1.801s", resultado)
    end

    @testset "Informe parcial omite Fase División en tiempos" begin
        informe = crear_informe_parcial()
        resultado = resumen_I(informe)
        @test occursin("Fase Nube:", resultado)
        @test !occursin("Fase División:", resultado)
        @test occursin("Tiempo total:", resultado)
    end

    @testset "Contiene subredes y precisiones cuando hay mapa" begin
        informe = crear_informe_completo()
        resultado = resumen_I(informe)
        @test occursin("Subredes: 3", resultado)
        @test occursin("Precisiones:", resultado)
        @test occursin("0.95", resultado)
        @test occursin("0.92", resultado)
        @test occursin("0.88", resultado)
    end

    @testset "Omite subredes cuando no hay mapa" begin
        informe = crear_informe_parcial()
        resultado = resumen_I(informe)
        @test !occursin("Subredes:", resultado)
        @test !occursin("Precisiones:", resultado)
    end

    @testset "Incluye error de Fase_Division cuando existe" begin
        informe = crear_informe_con_error()
        resultado = resumen_I(informe)
        @test occursin("Error Fase División:", resultado)
        @test occursin("descomposición fallida", resultado)
    end

    @testset "No incluye error cuando no hay error" begin
        informe = crear_informe_completo()
        resultado = resumen_I(informe)
        @test !occursin("Error Fase División:", resultado)
    end

    @testset "Funciona con Float32" begin
        informe = crear_informe_completo(; T=Float32)
        resultado = resumen_I(informe)
        @test resultado isa String
        @test !isempty(resultado)
        @test occursin("[4, 10, 3]", resultado)
        @test occursin("0.95", resultado)
    end
end
