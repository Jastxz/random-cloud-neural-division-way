# ═══════════════════════════════════════════════════════════════════════════════
# test/test_propiedades.jl — Tests de propiedades (PBT) agrupados
# Feature: neural-cloud-division
#
# Este archivo agrupa las 14 propiedades de corrección definidas en el diseño.
# Cada propiedad se implementa en su propio archivo con stubs aislados para
# evitar conflictos de tipos entre módulos.
#
# ── Índice de propiedades ─────────────────────────────────────────────────────
#
#  P1:  Ida y vuelta de configuración          (Req 1.1, 1.2, 1.3, 9.1)  — OPCIONAL, no implementada
#  P2:  Rechazo de parámetros inválidos        (Req 1.5, 1.6, 9.3)
#  P3:  El adaptador preserva atributos        (Req 3.1, 3.2, 3.3)
#  P4:  Ida y vuelta de serialización          (Req 5.1, 5.2, 5.3)
#  P5:  Determinismo con semilla               (Req 2.4)
#  P6:  Invariante de tiempos de ejecución     (Req 2.5)
#  P7:  Validación de dimensiones de datos     (Req 6.2)
#  P8:  Pipeline completo produce informe      (Req 2.3, 4.1–4.4)        — OPCIONAL, no implementada
#  P9:  Resumen contiene métricas clave        (Req 4.5)                  — OPCIONAL, no implementada
#  P10: Secuencia de callbacks                 (Req 8.1, 8.2)            — OPCIONAL, no implementada
#  P11: Clampeo de mini-batch                  (Req 9.2)
#  P12: Selección automática de backend        (Req 10.3, 10.4)
#  P13: Invariantes de ejecución en GPU        (Req 10.6, 10.7)
#  P14: Fase_Division acepta RedBase externa   (Req 7.3)
#
# ── Generadores compartidos (conceptuales) ────────────────────────────────────
#
# Cada archivo de propiedad define sus propios generadores dentro de un módulo
# stub aislado. Los generadores clave reutilizados conceptualmente son:
#
#   gen_config_valida   — Configuracion_RCND{T} con parámetros válidos aleatorios
#                         (usado en P2, P5, P6, P11, P12, P13, P14)
#
#   gen_dataset         — Par (datos_x, datos_y) con dimensiones consistentes
#                         (usado en P5, P6, P7, P11, P13, P14)
#
#   gen_red_neuronal    — RedNeuronal con capas/pesos/biases/activaciones aleatorios
#                         (usado en P3)
#
#   gen_informe         — Informe_RCND{T} completo con topología y mapa aleatorios
#                         (usado en P4)
#
# ═══════════════════════════════════════════════════════════════════════════════

using Test

@testset "Propiedades de Corrección (PBT)" begin

    # ── P2: Rechazo de parámetros inválidos ──────────────────────────────────
    # Feature: neural-cloud-division, Property 2: Rechazo de parámetros inválidos
    # Valida: Requisitos 1.5, 1.6, 9.3
    include("test_propiedades_p2.jl")

    # ── P3: El adaptador preserva todos los atributos de la red ──────────────
    # Feature: neural-cloud-division, Property 3: El adaptador preserva atributos
    # Valida: Requisitos 3.1, 3.2, 3.3
    include("test_propiedades_p3.jl")

    # ── P4: Ida y vuelta de serialización ────────────────────────────────────
    # Feature: neural-cloud-division, Property 4: Ida y vuelta de serialización
    # Valida: Requisitos 5.1, 5.2, 5.3
    include("test_propiedades_p4.jl")

    # ── P5: Determinismo con semilla ─────────────────────────────────────────
    # Feature: neural-cloud-division, Property 5: Determinismo con semilla
    # Valida: Requisito 2.4
    include("test_propiedades_p5.jl")

    # ── P6: Invariante de tiempos de ejecución ───────────────────────────────
    # Feature: neural-cloud-division, Property 6: Invariante de tiempos de ejecución
    # Valida: Requisito 2.5
    include("test_propiedades_p6.jl")

    # ── P7: Validación de dimensiones de datos ───────────────────────────────
    # Feature: neural-cloud-division, Property 7: Validación de dimensiones de datos
    # Valida: Requisito 6.2
    include("test_propiedades_p7.jl")

    # ── P11: Clampeo de mini-batch ───────────────────────────────────────────
    # Feature: neural-cloud-division, Property 11: Clampeo de mini-batch
    # Valida: Requisito 9.2
    include("test_propiedades_p11.jl")

    # ── P12: Selección automática de backend por umbral ──────────────────────
    # Feature: neural-cloud-division, Property 12: Selección automática de backend
    # Valida: Requisitos 10.3, 10.4
    include("test_propiedades_p12.jl")

    # ── P13: Invariantes de ejecución en GPU ─────────────────────────────────
    # Feature: neural-cloud-division, Property 13: Invariantes de ejecución en GPU
    # Valida: Requisitos 10.6, 10.7
    include("test_propiedades_p13.jl")

    # ── P14: Fase_Division acepta RedBase externa ────────────────────────────
    # Feature: neural-cloud-division, Property 14: Fase_Division acepta RedBase externa
    # Valida: Requisito 7.3
    include("test_propiedades_p14.jl")

    # ── Propiedades opcionales (no implementadas) ────────────────────────────
    # P1:  Ida y vuelta de configuración       — Tarea 1.4 (opcional)
    # P8:  Pipeline completo produce informe   — Tarea 10.2 (opcional)
    # P9:  Resumen contiene métricas clave     — Tarea 8.2 (opcional)
    # P10: Secuencia de callbacks              — Tarea 6.8 (opcional)

end
