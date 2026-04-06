#!/usr/bin/env julia
# ═══════════════════════════════════════════════════════════════════════════════
# run_benchmarks.jl — Benchmark completo de RCND con paquetes reales
#
# Uso:
#   julia benchmarks/run_benchmarks.jl              # Todos los benchmarks
#   julia benchmarks/run_benchmarks.jl binarios     # Solo clasificación binaria
#   julia benchmarks/run_benchmarks.jl multiclase   # Solo multiclase
#   julia benchmarks/run_benchmarks.jl sinteticos   # Solo sintéticos parametrizables
#   julia benchmarks/run_benchmarks.jl robustez     # Solo robustez/edge cases
# ═══════════════════════════════════════════════════════════════════════════════

using Random
using Printf

include(joinpath(@__DIR__, "stubs.jl"))
include(joinpath(@__DIR__, "datasets.jl"))

const ejecutar = RCND.ejecutar_benchmark_rcnd

# ═══════════════════════════════════════════════════════════════════════════════
# Formateo
# ═══════════════════════════════════════════════════════════════════════════════

function imprimir_encabezado()
    println("┌───────────────────────────────┬────────┬───────┬─────┬────────┬─────────────────┬────────┬───────────┬───────────┬───────────┐")
    println("│ Dataset                       │ Feat.  │ Samp. │ Out │ Prec.  │ Topología       │ Params │ T.Nube    │ T.Div     │ T.Total   │")
    println("├───────────────────────────────┼────────┼───────┼─────┼────────┼─────────────────┼────────┼───────────┼───────────┼───────────┤")
end

function imprimir_resultado(nombre::String, r::RCND.InformeBenchmark)
    umbral = r.exitoso_nube ? "✓" : "✗"
    topo = r.topologia_final !== nothing ? join(r.topologia_final, "→") : "—"
    if length(topo) > 15
        topo = topo[1:12] * "..."
    end
    t_div = r.division_ejecutada ? @sprintf("%8.1fms", r.tiempo_division_ms) : "    —    "
    nombre_fmt = length(nombre) > 29 ? nombre[1:26] * "..." : nombre

    @printf("│ %-29s │ %6d │ %5d │ %3d │ %s%.3f │ %-15s │ %6d │ %8.1fms │ %9s │ %8.1fms │\n",
        nombre_fmt,
        r.topologia_final !== nothing ? r.topologia_final[1] : 0,
        0,  # se llena abajo
        r.topologia_final !== nothing ? r.topologia_final[end] : 0,
        umbral, r.precision_nube,
        topo, r.n_parametros_nube,
        r.tiempo_nube_ms, t_div, r.tiempo_total_ms)
end

function imprimir_resultado_v2(nombre::String, r::RCND.InformeBenchmark, n_feat::Int, n_samp::Int, n_out::Int)
    umbral = r.exitoso_nube ? "✓" : "✗"
    topo = r.topologia_final !== nothing ? join(r.topologia_final, "→") : "—"
    if length(topo) > 15
        topo = topo[1:12] * "..."
    end
    t_div = r.division_ejecutada ? @sprintf("%8.1fms", r.tiempo_division_ms) : "     —   "
    nombre_fmt = length(nombre) > 29 ? nombre[1:26] * "..." : nombre

    @printf("│ %-29s │ %6d │ %5d │ %3d │ %s%.3f │ %-15s │ %6d │ %8.1fms │ %9s │ %8.1fms │\n",
        nombre_fmt, n_feat, n_samp, n_out,
        umbral, r.precision_nube,
        topo, r.n_parametros_nube,
        r.tiempo_nube_ms, t_div, r.tiempo_total_ms)
end

function imprimir_pie()
    println("└───────────────────────────────┴────────┴───────┴─────┴────────┴─────────────────┴────────┴───────────┴───────────┴───────────┘")
end

function imprimir_separador(titulo::String)
    println("├───────────────────────────────┴────────┴───────┴─────┴────────┴─────────────────┴────────┴───────────┴───────────┴───────────┤")
    @printf("│  ▸ %-105s │\n", titulo)
    println("├───────────────────────────────┬────────┬───────┬─────┬────────┬─────────────────┬────────┬───────────┬───────────┬───────────┤")
end

# ═══════════════════════════════════════════════════════════════════════════════
# Helper para ejecutar y mostrar
# ═══════════════════════════════════════════════════════════════════════════════

function bench(nombre::String, datos_x::Matrix{Float64}, datos_y::Matrix{Float64}; kwargs...)
    n_feat = size(datos_x, 1)
    n_samp = size(datos_x, 2)
    n_out = size(datos_y, 1)
    r = ejecutar(datos_x, datos_y; kwargs...)
    imprimir_resultado_v2(nombre, r, n_feat, n_samp, n_out)

    if r.division_ejecutada
        @printf("│   └─ División: prec_global=%.3f, soluciones_parciales=%d\n",
            r.precision_global_div, r.n_soluciones_parciales)
    end

    return r
end

# ═══════════════════════════════════════════════════════════════════════════════
# Benchmarks por categoría
# ═══════════════════════════════════════════════════════════════════════════════

function benchmarks_binarios(rng)
    resultados = RCND.InformeBenchmark[]

    push!(resultados, bench("XOR (n=200)", generar_xor(; n=200, rng=copy(rng))...;
        tamano_nube=10, umbral_acierto=0.9, epocas_refinamiento=500))

    push!(resultados, bench("XOR (n=1000)", generar_xor(; n=1000, ruido=0.05, rng=copy(rng))...;
        tamano_nube=15, umbral_acierto=0.9, epocas_refinamiento=1000))

    push!(resultados, bench("Lunas (n=400)", generar_lunas(; n=400, rng=copy(rng))...;
        tamano_nube=10, umbral_acierto=0.85, epocas_refinamiento=500,
        topologia_inicial=[2, 6, 1]))

    push!(resultados, bench("Lunas (n=2000)", generar_lunas(; n=2000, ruido=0.1, rng=copy(rng))...;
        tamano_nube=15, umbral_acierto=0.9, epocas_refinamiento=1000,
        topologia_inicial=[2, 8, 1], batch_size_nube=64))

    push!(resultados, bench("Espirales (n=400)", generar_espirales(; n=400, rng=copy(rng))...;
        tamano_nube=20, umbral_acierto=0.8, epocas_refinamiento=1000,
        topologia_inicial=[2, 10, 1]))

    push!(resultados, bench("Espirales (n=1000, 2.5v)", generar_espirales(; n=1000, vueltas=2.5, ruido=0.05, rng=copy(rng))...;
        tamano_nube=30, umbral_acierto=0.8, epocas_refinamiento=2000,
        topologia_inicial=[2, 12, 1], batch_size_nube=64))

    resultados
end

function benchmarks_multiclase(rng)
    resultados = RCND.InformeBenchmark[]

    push!(resultados, bench("Iris sintético (n=150)", generar_iris_sintetico(; n=150, rng=copy(rng))...;
        tamano_nube=10, umbral_acierto=0.85, epocas_refinamiento=500,
        topologia_inicial=[4, 8, 3]))

    push!(resultados, bench("Iris sintético (n=600)", generar_iris_sintetico(; n=600, rng=copy(rng))...;
        tamano_nube=15, umbral_acierto=0.9, epocas_refinamiento=1000,
        topologia_inicial=[4, 10, 3], batch_size_nube=32))

    push!(resultados, bench("Wine sintético (n=178)", generar_wine_sintetico(; n=178, rng=copy(rng))...;
        tamano_nube=10, umbral_acierto=0.8, epocas_refinamiento=500,
        topologia_inicial=[13, 10, 3]))

    push!(resultados, bench("Wine sintético (n=500)", generar_wine_sintetico(; n=500, rng=copy(rng))...;
        tamano_nube=15, umbral_acierto=0.85, epocas_refinamiento=1000,
        topologia_inicial=[13, 12, 3], batch_size_nube=32))

    # Nota: dígitos con 10 clases genera 2^10-1 = 1023 subconfiguraciones en división
    # Puede ser lento. Usamos pocas features para mantenerlo manejable.
    push!(resultados, bench("Dígitos (10f, 5c, n=250)", generar_digitos_reducido(; n=250, n_features=10, n_clases=5, rng=copy(rng))...;
        tamano_nube=15, umbral_acierto=0.7, epocas_refinamiento=1000,
        topologia_inicial=[10, 12, 5], batch_size_nube=32,
        epochs_division=200, paciencia_division=30))

    resultados
end

function benchmarks_sinteticos(rng)
    resultados = RCND.InformeBenchmark[]

    push!(resultados, bench("Blobs (2f, 3c, sep=3)", generar_blobs(; n=300, n_clases=3, n_features=2, separacion=3.0, rng=copy(rng))...;
        tamano_nube=10, umbral_acierto=0.85, epocas_refinamiento=500,
        topologia_inicial=[2, 6, 3]))

    push!(resultados, bench("Blobs (2f, 5c, sep=4)", generar_blobs(; n=500, n_clases=5, n_features=2, separacion=4.0, rng=copy(rng))...;
        tamano_nube=15, umbral_acierto=0.8, epocas_refinamiento=1000,
        topologia_inicial=[2, 10, 5], batch_size_nube=32))

    push!(resultados, bench("Blobs (4f, 3c, sep=2)", generar_blobs(; n=300, n_clases=3, n_features=4, separacion=2.0, rng=copy(rng))...;
        tamano_nube=10, umbral_acierto=0.8, epocas_refinamiento=500,
        topologia_inicial=[4, 8, 3]))

    push!(resultados, bench("Checkerboard (4×4, n=600)", generar_checkerboard(; n=600, tamano=4, rng=copy(rng))...;
        tamano_nube=20, umbral_acierto=0.75, epocas_refinamiento=1000,
        topologia_inicial=[2, 10, 1]))

    push!(resultados, bench("Anillos (3, n=600)", generar_anillos(; n=600, n_anillos=3, rng=copy(rng))...;
        tamano_nube=15, umbral_acierto=0.8, epocas_refinamiento=1000,
        topologia_inicial=[2, 8, 3]))

    push!(resultados, bench("Anillos (5, n=1000)", generar_anillos(; n=1000, n_anillos=5, ruido=0.1, rng=copy(rng))...;
        tamano_nube=20, umbral_acierto=0.7, epocas_refinamiento=1000,
        topologia_inicial=[2, 12, 5], batch_size_nube=32,
        epochs_division=200, paciencia_division=30))

    resultados
end

function benchmarks_robustez(rng)
    resultados = RCND.InformeBenchmark[]

    # Ruido con pocas features para que no tarde demasiado
    push!(resultados, bench("Ruido (4+6 feat)", generar_con_ruido(; n=200, n_features_utiles=4, n_features_ruido=6, rng=copy(rng))...;
        tamano_nube=10, umbral_acierto=0.8, epocas_refinamiento=300,
        topologia_inicial=[10, 6, 1]))

    push!(resultados, bench("Desbalanceado (90/10)", generar_desbalanceado(; n=300, ratio_minoritaria=0.1, rng=copy(rng))...;
        tamano_nube=10, umbral_acierto=0.85, epocas_refinamiento=300,
        topologia_inicial=[4, 6, 1]))

    push!(resultados, bench("Desbalanceado (95/5)", generar_desbalanceado(; n=500, ratio_minoritaria=0.05, rng=copy(rng))...;
        tamano_nube=10, umbral_acierto=0.8, epocas_refinamiento=500,
        topologia_inicial=[4, 6, 1], batch_size_nube=32))

    resultados
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

function main()
    rng = MersenneTwister(12345)
    filtro = length(ARGS) > 0 ? ARGS[1] : "todos"

    println()
    println("  ╔═══════════════════════════════════════════════════════════════╗")
    println("  ║   RCND Benchmark Suite — Paquetes Reales                     ║")
    println("  ║   RandomCloud.jl + DivisionNeuronal.jl                       ║")
    println("  ╚═══════════════════════════════════════════════════════════════╝")
    println()
    println("  Semilla global: 12345 | Threads: $(Threads.nthreads())")
    println("  Filtro: $filtro")
    println()

    todos = RCND.InformeBenchmark[]
    imprimir_encabezado()

    if filtro in ("todos", "binarios")
        imprimir_separador("Clasificación binaria")
        append!(todos, benchmarks_binarios(rng))
    end

    if filtro in ("todos", "multiclase")
        imprimir_separador("Clasificación multiclase")
        append!(todos, benchmarks_multiclase(rng))
    end

    if filtro in ("todos", "sinteticos")
        imprimir_separador("Sintéticos parametrizables")
        append!(todos, benchmarks_sinteticos(rng))
    end

    if filtro in ("todos", "robustez")
        imprimir_separador("Robustez y edge cases")
        append!(todos, benchmarks_robustez(rng))
    end

    imprimir_pie()

    # Resumen
    println()
    n_total = length(todos)
    n_exitosos = count(r -> r.exitoso_nube, todos)
    n_division = count(r -> r.division_ejecutada, todos)
    t_total = sum(r -> r.tiempo_total_ms, todos)
    @printf("  Resumen: %d/%d Fase Nube exitosa | %d/%d Fase División ejecutada\n",
        n_exitosos, n_total, n_division, n_total)
    @printf("  Tiempo total: %.1fms (%.2fs)\n", t_total, t_total / 1000)
    println()
end

main()
