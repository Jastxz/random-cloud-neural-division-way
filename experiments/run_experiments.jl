#!/usr/bin/env julia
# ═══════════════════════════════════════════════════════════════════════════════
# run_experiments.jl — Experimentos aplicados: diagnóstico médico + industria
#
# Uso:
#   julia experiments/run_experiments.jl              # Todos
#   julia experiments/run_experiments.jl medical      # Solo médico
#   julia experiments/run_experiments.jl industrial   # Solo industrial
# ═══════════════════════════════════════════════════════════════════════════════

using Random
using Printf

# Cargar pipeline real
include(joinpath(@__DIR__, "..", "benchmarks", "stubs.jl"))

# Cargar datasets
include(joinpath(@__DIR__, "medical", "datasets.jl"))
include(joinpath(@__DIR__, "industrial", "datasets.jl"))

const ejecutar = RCND.ejecutar_benchmark_rcnd

# ═══════════════════════════════════════════════════════════════════════════════
# Formateo
# ═══════════════════════════════════════════════════════════════════════════════

function separador(titulo)
    println()
    println("  ┌─────────────────────────────────────────────────────────────┐")
    @printf("  │  %-59s│\n", titulo)
    println("  └─────────────────────────────────────────────────────────────┘")
    println()
end

function mostrar_resultado(nombre, r, n_feat, n_samp, n_out)
    estado = r.exitoso_nube ? "✓" : "✗"
    topo = r.topologia_final !== nothing ? join(r.topologia_final, "→") : "no encontrada"

    println("  $estado $nombre")
    println("    Datos: $(n_feat) features × $(n_samp) muestras → $(n_out) clases")

    if r.exitoso_nube
        @printf("    Fase Nube:    precisión=%.1f%%  topología=[%s]  params=%d  (%.0fms)\n",
            r.precision_nube * 100, topo, r.n_parametros_nube, r.tiempo_nube_ms)

        if r.division_ejecutada
            @printf("    Fase División: prec_global=%.1f%%  soluciones_parciales=%d  (%.0fms)\n",
                r.precision_global_div * 100, r.n_soluciones_parciales, r.tiempo_division_ms)
        end

        @printf("    Tiempo total: %.1fms\n", r.tiempo_total_ms)
    else
        @printf("    Fase Nube: no alcanzó umbral (mejor=%.1f%%)  (%.0fms)\n",
            r.precision_nube * 100, r.tiempo_nube_ms)
    end
    println()
end

function bench(nombre, datos_x, datos_y; kwargs...)
    n_feat = size(datos_x, 1)
    n_samp = size(datos_x, 2)
    n_out = size(datos_y, 1)
    r = ejecutar(datos_x, datos_y; kwargs...)
    mostrar_resultado(nombre, r, n_feat, n_samp, n_out)
    r
end

# ═══════════════════════════════════════════════════════════════════════════════
# Experimentos médicos
# ═══════════════════════════════════════════════════════════════════════════════

function experimentos_medicos(rng)
    separador("DIAGNÓSTICO MÉDICO — Clasificación binaria")

    # Cáncer de mama: 10 features, binario, 569 muestras
    bench("Cáncer de mama (Wisconsin)", generar_cancer_mama(; rng=copy(rng))...;
        tamano_nube=50, umbral_acierto=0.5, epocas_refinamiento=1000,
        topologia_inicial=[10, 8, 1], tasa_aprendizaje_nube=0.1)

    # Cardiopatía: 13 features, binario, 303 muestras
    bench("Enfermedad cardíaca (Cleveland)", generar_cardiopatia(; rng=copy(rng))...;
        tamano_nube=50, umbral_acierto=0.5, epocas_refinamiento=1000,
        topologia_inicial=[13, 10, 1], tasa_aprendizaje_nube=0.1)

    # Hepatopatía: 10 features, binario, 583 muestras, desbalanceado
    bench("Enfermedad hepática (Indian Liver)", generar_hepatopatia(; rng=copy(rng))...;
        tamano_nube=50, umbral_acierto=0.5, epocas_refinamiento=1000,
        topologia_inicial=[10, 8, 1], tasa_aprendizaje_nube=0.1)

    # Diabetes: 8 features, binario, 768 muestras
    bench("Diabetes tipo 2 (Pima)", generar_diabetes(; rng=copy(rng))...;
        tamano_nube=50, umbral_acierto=0.5, epocas_refinamiento=1000,
        topologia_inicial=[8, 6, 1], tasa_aprendizaje_nube=0.1)

    # Enfermedad renal: 11 features, binario, 400 muestras
    bench("Enfermedad renal crónica", generar_enfermedad_renal(; rng=copy(rng))...;
        tamano_nube=50, umbral_acierto=0.5, epocas_refinamiento=1000,
        topologia_inicial=[11, 8, 1], tasa_aprendizaje_nube=0.1)

    separador("DIAGNÓSTICO MÉDICO — Clasificación multiclase")

    # Tiroides: 5 features, 3 clases, 215 muestras — rápido y factible
    bench("Tiroides (3 condiciones)", generar_tiroides(; rng=copy(rng))...;
        tamano_nube=50, umbral_acierto=0.4, epocas_refinamiento=1000,
        topologia_inicial=[5, 6, 3], tasa_aprendizaje_nube=0.1,
        umbral_division=0.4, epochs_division=500, paciencia_division=30)

    # Tiroides multicapa
    bench("Tiroides 2-capas", generar_tiroides(; rng=copy(rng))...;
        tamano_nube=50, umbral_acierto=0.4, epocas_refinamiento=1000,
        topologia_inicial=[5, 8, 6, 3], tasa_aprendizaje_nube=0.05,
        umbral_division=0.4, epochs_division=500, paciencia_division=30)

    # Dermatología: 12 features, 6 clases — SOLO Fase Nube (12 inputs = 4095 subconfiguraciones)
    bench("Dermatología (solo Nube)", generar_dermatologia(; n=200, rng=copy(rng))...;
        tamano_nube=30, umbral_acierto=0.25, epocas_refinamiento=500,
        topologia_inicial=[12, 10, 6], tasa_aprendizaje_nube=0.1,
        umbral_division=1.01, epochs_division=1, paciencia_division=1)  # 1.01 = imposible alcanzar

    bench("Dermatología 2-capas (solo Nube)", generar_dermatologia(; n=200, rng=copy(rng))...;
        tamano_nube=30, umbral_acierto=0.25, epocas_refinamiento=500,
        topologia_inicial=[12, 10, 8, 6], tasa_aprendizaje_nube=0.05,
        umbral_division=1.01, epochs_division=1, paciencia_division=1)  # 1.01 = imposible alcanzar
end

# ═══════════════════════════════════════════════════════════════════════════════
# Experimentos industriales
# ═══════════════════════════════════════════════════════════════════════════════

function experimentos_industriales(rng)
    separador("CONTROL DE CALIDAD — Clasificación multiclase")

    # Defectos acero: 13 features, 7 clases — solo Fase Nube (13 inputs = 8191 subconfiguraciones)
    bench("Defectos acero (solo Nube)", generar_defectos_acero(; rng=copy(rng))...;
        tamano_nube=50, umbral_acierto=0.25, epocas_refinamiento=1000,
        topologia_inicial=[13, 10, 7], tasa_aprendizaje_nube=0.1,
        umbral_division=0.99, epochs_division=1, paciencia_division=1)

    bench("Defectos acero 2-capas (solo Nube)", generar_defectos_acero(; rng=copy(rng))...;
        tamano_nube=50, umbral_acierto=0.25, epocas_refinamiento=1000,
        topologia_inicial=[13, 12, 8, 7], tasa_aprendizaje_nube=0.05,
        umbral_division=0.99, epochs_division=1, paciencia_division=1)

    # Calidad soldadura: 10 features, 4 clases — solo Fase Nube (10 inputs = 1023 subconfiguraciones × 15 = ~15K)
    bench("Soldadura (solo Nube)", generar_calidad_soldadura(; rng=copy(rng))...;
        tamano_nube=50, umbral_acierto=0.35, epocas_refinamiento=1000,
        topologia_inicial=[10, 8, 4], tasa_aprendizaje_nube=0.1,
        umbral_division=0.99, epochs_division=1, paciencia_division=1)

    bench("Soldadura 2-capas (solo Nube)", generar_calidad_soldadura(; rng=copy(rng))...;
        tamano_nube=50, umbral_acierto=0.35, epocas_refinamiento=1000,
        topologia_inicial=[10, 10, 6, 4], tasa_aprendizaje_nube=0.05,
        umbral_division=0.99, epochs_division=1, paciencia_division=1)

    # Calidad vino: 11 features, 3 clases — solo Fase Nube
    bench("Calidad de vino (solo Nube)", generar_calidad_vino(; rng=copy(rng))...;
        tamano_nube=50, umbral_acierto=0.4, epocas_refinamiento=1000,
        topologia_inicial=[11, 8, 3], tasa_aprendizaje_nube=0.1,
        umbral_division=0.99, epochs_division=1, paciencia_division=1)

    separador("MANTENIMIENTO PREDICTIVO — Detección de fallos")

    # Rodamientos: 8 features, 4 clases, desbalanceado — División factible (255 × 15 = 3825)
    bench("Fallos de rodamiento (4 tipos)", generar_fallos_rodamiento(; rng=copy(rng))...;
        tamano_nube=50, umbral_acierto=0.35, epocas_refinamiento=1000,
        topologia_inicial=[8, 8, 4], tasa_aprendizaje_nube=0.1,
        umbral_division=0.4, epochs_division=300, paciencia_division=30)

    # Rodamientos multicapa
    bench("Rodamientos 2-capas", generar_fallos_rodamiento(; rng=copy(rng))...;
        tamano_nube=50, umbral_acierto=0.35, epocas_refinamiento=1000,
        topologia_inicial=[8, 10, 6, 4], tasa_aprendizaje_nube=0.05,
        umbral_division=0.4, epochs_division=300, paciencia_division=30)

    # Defectos wafer: 8 features, 5 clases — División factible (255 × 31 = 7905)
    bench("Defectos de wafer (5 tipos)", generar_defectos_wafer(; rng=copy(rng))...;
        tamano_nube=50, umbral_acierto=0.3, epocas_refinamiento=1000,
        topologia_inicial=[8, 8, 5], tasa_aprendizaje_nube=0.1,
        umbral_division=0.4, epochs_division=200, paciencia_division=20)

    # Turbina: 6 features, 3 clases — escenario edge/IoT, División rápida (63 × 7 = 441)
    bench("Turbina (3 estados)", generar_turbina(; rng=copy(rng))...;
        tamano_nube=50, umbral_acierto=0.4, epocas_refinamiento=1000,
        topologia_inicial=[6, 6, 3], tasa_aprendizaje_nube=0.1,
        umbral_division=0.4, epochs_division=500, paciencia_division=30)

    # Turbina multicapa
    bench("Turbina 2-capas", generar_turbina(; rng=copy(rng))...;
        tamano_nube=50, umbral_acierto=0.4, epocas_refinamiento=1000,
        topologia_inicial=[6, 8, 6, 3], tasa_aprendizaje_nube=0.05,
        umbral_division=0.4, epochs_division=500, paciencia_division=30)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

function main()
    rng = MersenneTwister(42)
    filtro = length(ARGS) > 0 ? ARGS[1] : "todos"

    println()
    println("  ╔═══════════════════════════════════════════════════════════════╗")
    println("  ║   RCND — Experimentos Aplicados                              ║")
    println("  ║   RandomCloud.jl + DivisionNeuronal.jl                       ║")
    println("  ╠═══════════════════════════════════════════════════════════════╣")
    println("  ║   Diagnóstico médico + Control de calidad industrial         ║")
    println("  ╚═══════════════════════════════════════════════════════════════╝")
    println()
    println("  Semilla: 42 | Threads: $(Threads.nthreads()) | Filtro: $filtro")

    if filtro in ("todos", "medical")
        experimentos_medicos(rng)
    end

    if filtro in ("todos", "industrial")
        experimentos_industriales(rng)
    end

    println()
    println("  ═══ Fin de experimentos ═══")
    println()
end

main()
