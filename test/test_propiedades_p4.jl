# Feature: neural-cloud-division, Property 4: Ida y vuelta de serialización
# **Valida: Requisitos 5.1, 5.2, 5.3**

using Test
using Random
using Supposition
using Supposition.Data
using JSON3

# ─── Stub de dependencias ficticias ───────────────────────────────────────────
module _StubsP4
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

const Informe_P4 = _StubsP4.Informe_RCND
const TopologiaOptima_P4 = _StubsP4.TopologiaOptima
const Configuracion_P4 = _StubsP4.Configuracion_RCND
const MapaDeSoluciones_P4 = _StubsP4.MapaDeSoluciones
const serializar_P4 = _StubsP4.serializar
const deserializar_P4 = _StubsP4.deserializar

# ─── Generadores ──────────────────────────────────────────────────────────────

# Activaciones válidas
const ACTIVACIONES_P4 = [:sigmoid, :relu, :identity]

# Número de capas totales (2-5, es decir 1-4 conexiones)
const gen_n_capas_p4 = Data.Integers(2, 5)

# Neuronas por capa (1-10)
const gen_neuronas_p4 = Data.Integers(1, 10)

# Semilla para generar pesos/datos aleatorios
const gen_semilla_p4 = Data.Integers(1, 10_000_000)

# Precisión en [0, 1] — finite floats filtered to valid range
const gen_precision_p4 = Data.Satisfying(Data.Floats{Float64}(; nans=false, infs=false), x -> 0.0 <= x <= 1.0)

# Tiempos >= 0 — finite non-negative floats
const gen_tiempo_p4 = Data.Satisfying(Data.Floats{Float64}(; nans=false, infs=false), x -> x >= 0.0 && x <= 1000.0)

# Bool para decidir si incluir mapa de soluciones (50% chance)
const gen_incluir_mapa_p4 = Data.Booleans()

# Parámetros de config válidos
const gen_n_redes_p4 = Data.Integers(1, 50)
const gen_max_iter_p4 = Data.Integers(1, 100)
const gen_max_epocas_p4 = Data.Integers(1, 500)
const gen_lr_p4 = Data.Satisfying(Data.Floats{Float64}(; nans=false, infs=false), x -> 1e-5 <= x <= 1e-1)
const gen_batch_p4 = Data.Integers(1, 128)
const gen_umbral_gpu_p4 = Data.Integers(100, 100_000)

# Verbosidades y backends válidos
const VERBOSIDADES_P4 = [:silencioso, :normal, :detallado]
const BACKENDS_P4 = [:cpu, :gpu, :auto]

# ─── Generador compuesto de Informe_RCND{Float64} ────────────────────────────

const gen_informe_p4 = @composed function gen_informe(
    n_capas       = gen_n_capas_p4,
    semilla       = gen_semilla_p4,
    prec_topo     = gen_precision_p4,
    t_nube        = gen_tiempo_p4,
    t_total_extra = gen_tiempo_p4,
    incluir_mapa  = gen_incluir_mapa_p4,
    umbral_prec   = gen_precision_p4,
    umbral_div    = gen_precision_p4,
    n_redes       = gen_n_redes_p4,
    max_iter      = gen_max_iter_p4,
    max_epocas    = gen_max_epocas_p4,
    lr            = gen_lr_p4,
    batch         = gen_batch_p4,
    umbral_gpu    = gen_umbral_gpu_p4,
)
    rng = Random.MersenneTwister(semilla)

    # ── Generar TopologiaOptima ──
    dims = [produce!(gen_neuronas_p4) for _ in 1:n_capas]
    n_conexiones = n_capas - 1

    pesos = [randn(rng, Float64, dims[i+1], dims[i]) for i in 1:n_conexiones]
    biases = [randn(rng, Float64, dims[i+1]) for i in 1:n_conexiones]
    activaciones = [ACTIVACIONES_P4[rand(rng, 1:3)] for _ in 1:n_conexiones]

    topologia = TopologiaOptima_P4{Float64}(dims, pesos, biases, activaciones)

    # ── Generar Configuracion_RCND ──
    verbosidad = VERBOSIDADES_P4[rand(rng, 1:3)]
    backend = BACKENDS_P4[rand(rng, 1:3)]
    semilla_config = rand(rng, Bool) ? rand(rng, 1:10_000_000) : nothing

    config = Configuracion_P4{Float64}(
        n_redes_por_nube = n_redes,
        umbral_precision = umbral_prec,
        max_iteraciones_reduccion = max_iter,
        umbral_division = umbral_div,
        max_epocas = max_epocas,
        tasa_aprendizaje = lr,
        tamano_mini_batch = batch,
        semilla = semilla_config,
        verbosidad = verbosidad,
        backend_computo = backend,
        umbral_gpu = umbral_gpu,
    )

    # ── Generar MapaDeSoluciones (50% chance) ──
    umbral_alcanzado = prec_topo >= umbral_prec
    mapa_soluciones = nothing
    precisiones_subredes = nothing
    tiempo_fase_division = nothing
    error_fase_division = nothing

    if incluir_mapa
        n_subredes = rand(rng, 1:5)
        mapa_data = randn(rng, Float64, n_subredes)
        mapa_soluciones = MapaDeSoluciones_P4{Float64}(mapa_data)
        precisiones_subredes = [rand(rng, Float64) for _ in 1:n_subredes]
        tiempo_fase_division = rand(rng) * 100.0
    end

    tiempo_total = t_nube + t_total_extra + (tiempo_fase_division === nothing ? 0.0 : tiempo_fase_division)

    Informe_P4{Float64}(
        topologia,
        prec_topo,
        t_nube,
        mapa_soluciones,
        precisiones_subredes,
        tiempo_fase_division,
        tiempo_total,
        config,
        umbral_alcanzado,
        error_fase_division,
    )
end

# ─── Test de propiedad ────────────────────────────────────────────────────────

@testset "P4: Ida y vuelta de serialización" begin

    @check max_examples=100 function p4_ida_y_vuelta_serializacion(informe = gen_informe_p4)
        # Serializar y deserializar
        json_str = serializar_P4(informe)
        reconstruido = deserializar_P4(Informe_P4{Float64}, json_str)

        # ── Verificar campos escalares exactos ──
        reconstruido.umbral_alcanzado == informe.umbral_alcanzado || return false
        reconstruido.error_fase_division == informe.error_fase_division || return false

        # ── Verificar campos flotantes con tolerancia ──
        reconstruido.precision_topologia ≈ informe.precision_topologia || return false
        reconstruido.tiempo_fase_nube ≈ informe.tiempo_fase_nube || return false
        reconstruido.tiempo_total ≈ informe.tiempo_total || return false

        # ── Verificar topología ──
        reconstruido.topologia.capas == informe.topologia.capas || return false
        reconstruido.topologia.activaciones == informe.topologia.activaciones || return false

        for i in eachindex(informe.topologia.pesos)
            reconstruido.topologia.pesos[i] ≈ informe.topologia.pesos[i] || return false
        end
        for i in eachindex(informe.topologia.biases)
            reconstruido.topologia.biases[i] ≈ informe.topologia.biases[i] || return false
        end

        # ── Verificar mapa de soluciones ──
        if informe.mapa_soluciones === nothing
            reconstruido.mapa_soluciones === nothing || return false
            reconstruido.precisiones_subredes === nothing || return false
            reconstruido.tiempo_fase_division === nothing || return false
        else
            reconstruido.mapa_soluciones !== nothing || return false
            reconstruido.mapa_soluciones.data ≈ informe.mapa_soluciones.data || return false
            reconstruido.precisiones_subredes ≈ informe.precisiones_subredes || return false
            reconstruido.tiempo_fase_division ≈ informe.tiempo_fase_division || return false
        end

        # ── Verificar config: escalares exactos ──
        rc = reconstruido.config_utilizada
        oc = informe.config_utilizada
        rc.n_redes_por_nube == oc.n_redes_por_nube || return false
        rc.max_iteraciones_reduccion == oc.max_iteraciones_reduccion || return false
        rc.max_epocas == oc.max_epocas || return false
        rc.tamano_mini_batch == oc.tamano_mini_batch || return false
        rc.semilla == oc.semilla || return false
        rc.verbosidad == oc.verbosidad || return false
        rc.backend_computo == oc.backend_computo || return false
        rc.umbral_gpu == oc.umbral_gpu || return false
        rc.activaciones == oc.activaciones || return false

        # ── Verificar config: flotantes con tolerancia ──
        rc.umbral_precision ≈ oc.umbral_precision || return false
        rc.umbral_division ≈ oc.umbral_division || return false
        rc.tasa_aprendizaje ≈ oc.tasa_aprendizaje || return false

        true
    end
end
