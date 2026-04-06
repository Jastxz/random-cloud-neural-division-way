# informe.jl — resumen() y lógica de Informe_RCND
# Requisitos: 4.5

"""
    resumen(informe::Informe_RCND{T})::String where {T}

Devuelve una representación textual legible del informe RCND con las métricas
principales de ambas fases del pipeline.

Incluye: topología (capas y neuronas), precisión, umbral alcanzado, tiempos de
cada fase y total. Si hay mapa de soluciones, incluye número de subredes y
precisiones. Si hay error de Fase_Division, lo incluye en el resumen.

# Valida: Requisito 4.5
"""
function resumen(informe::Informe_RCND{T})::String where {T}
    io = IOBuffer()

    # Encabezado
    println(io, "═══ Informe RCND ═══")

    # Topología
    capas = informe.topologia.capas
    n_capas = length(capas)
    println(io, "Topología: ", capas, " (", n_capas, " capas)")

    # Precisión
    println(io, "Precisión: ", _fmt_precision(informe.precision_topologia))

    # Umbral alcanzado
    indicador = informe.umbral_alcanzado ? "✓" : "✗"
    println(io, "Umbral alcanzado: ", indicador)

    # Tiempos
    println(io)
    println(io, "Fase Nube: ", _fmt_tiempo(informe.tiempo_fase_nube))
    if informe.tiempo_fase_division !== nothing
        println(io, "Fase División: ", _fmt_tiempo(informe.tiempo_fase_division))
    end
    println(io, "Tiempo total: ", _fmt_tiempo(informe.tiempo_total))

    # Mapa de soluciones
    if informe.mapa_soluciones !== nothing && informe.precisiones_subredes !== nothing
        println(io)
        n_subredes = length(informe.precisiones_subredes)
        println(io, "Subredes: ", n_subredes)
        precs_fmt = [_fmt_precision(p) for p in informe.precisiones_subredes]
        println(io, "Precisiones: [", join(precs_fmt, ", "), "]")
    end

    # Error de Fase_Division
    if informe.error_fase_division !== nothing
        println(io)
        println(io, "Error Fase División: ", informe.error_fase_division)
    end

    return String(take!(io))
end

# ─── Helpers internos ─────────────────────────────────────────────────────────

"""Formatea un valor de precisión con 4 decimales."""
_fmt_precision(val) = string(round(Float64(val); digits=4))

"""Formatea un tiempo en segundos con 3 decimales y sufijo 's'."""
_fmt_tiempo(t) = string(round(t; digits=3), "s")
