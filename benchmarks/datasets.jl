# datasets.jl — Generadores de datasets sintéticos para benchmarks RCND
#
# Todos los datasets devuelven (datos_x, datos_y) en convención column-major:
#   datos_x: (n_features, n_samples)
#   datos_y: (n_outputs, n_samples)
#
# Las etiquetas se codifican en one-hot para clasificación multiclase.

using Random
using LinearAlgebra

# ═══════════════════════════════════════════════════════════════════════════════
# Utilidades
# ═══════════════════════════════════════════════════════════════════════════════

"""One-hot encoding: vector de enteros → matriz (n_clases, n_samples)."""
function onehot(labels::Vector{Int}, n_clases::Int)
    n = length(labels)
    y = zeros(Float64, n_clases, n)
    for i in 1:n
        y[labels[i], i] = 1.0
    end
    y
end

"""Etiquetas binarias como matriz (1, n_samples)."""
binarylabel(labels::Vector{Int}) = reshape(Float64.(labels), 1, :)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Clasificación binaria clásica
# ═══════════════════════════════════════════════════════════════════════════════

"""
    generar_xor(; n=200, ruido=0.1, rng=Random.default_rng())

Dataset XOR: 4 cuadrantes, etiquetas 0/1 según paridad.
"""
function generar_xor(; n::Int=200, ruido::Float64=0.1, rng=Random.default_rng())
    x = randn(rng, 2, n)
    labels = [Int((x[1,i] > 0) ⊻ (x[2,i] > 0)) + 1 for i in 1:n]  # 1 o 2
    x .+= ruido .* randn(rng, 2, n)
    (x, binarylabel(labels .- 1))  # etiquetas 0/1
end

"""
    generar_lunas(; n=400, ruido=0.15, rng=Random.default_rng())

Dos medias lunas entrelazadas (make_moons). Frontera no lineal.
"""
function generar_lunas(; n::Int=400, ruido::Float64=0.15, rng=Random.default_rng())
    n_half = n ÷ 2
    # Luna superior
    θ1 = range(0, π, length=n_half)
    x1 = [cos.(θ1)'; sin.(θ1)']
    # Luna inferior (desplazada)
    θ2 = range(0, π, length=n - n_half)
    x2 = [1.0 .- cos.(θ2)'; 0.5 .- sin.(θ2)']

    datos_x = hcat(x1, x2) .+ ruido .* randn(rng, 2, n)
    labels = vcat(zeros(Int, n_half), ones(Int, n - n_half))
    (datos_x, binarylabel(labels))
end

"""
    generar_espirales(; n=400, vueltas=1.5, ruido=0.1, rng=Random.default_rng())

Dos espirales entrelazadas. Requiere topologías profundas.
"""
function generar_espirales(; n::Int=400, vueltas::Float64=1.5, ruido::Float64=0.1, rng=Random.default_rng())
    n_half = n ÷ 2
    θ = range(0, vueltas * 2π, length=n_half)
    r = range(0.2, 1.0, length=n_half)

    # Espiral 1
    x1 = r' .* [cos.(θ)'; sin.(θ)']
    # Espiral 2 (rotada 180°)
    x2 = r' .* [-cos.(θ)'; -sin.(θ)']

    datos_x = hcat(x1, x2) .+ ruido .* randn(rng, 2, n)
    labels = vcat(zeros(Int, n_half), ones(Int, n - n_half))
    (datos_x, binarylabel(labels))
end

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Clasificación multiclase
# ═══════════════════════════════════════════════════════════════════════════════

"""
    generar_iris_sintetico(; n=150, rng=Random.default_rng())

Iris sintético: 4 features, 3 clases con distribuciones gaussianas separadas.
"""
function generar_iris_sintetico(; n::Int=150, rng=Random.default_rng())
    n_per_class = n ÷ 3
    # Centroides de cada clase (4 features)
    centroides = [
        [5.0, 3.4, 1.5, 0.2],   # Setosa
        [5.9, 2.8, 4.3, 1.3],   # Versicolor
        [6.6, 3.0, 5.6, 2.0],   # Virginica
    ]
    σ = [0.4, 0.3, 0.5, 0.2]

    datos_x = zeros(4, 0)
    labels = Int[]
    for (c, centro) in enumerate(centroides)
        bloque = centro .+ σ .* randn(rng, 4, n_per_class)
        datos_x = hcat(datos_x, bloque)
        append!(labels, fill(c, n_per_class))
    end
    (datos_x, onehot(labels, 3))
end

"""
    generar_wine_sintetico(; n=178, rng=Random.default_rng())

Wine sintético: 13 features, 3 clases. Más features que Iris.
"""
function generar_wine_sintetico(; n::Int=178, rng=Random.default_rng())
    n_per_class = n ÷ 3
    n_features = 13

    datos_x = zeros(n_features, 0)
    labels = Int[]
    for c in 1:3
        centro = randn(rng, n_features) .* 2.0 .+ c * 1.5
        σ = abs.(randn(rng, n_features)) .* 0.5 .+ 0.3
        bloque = centro .+ σ .* randn(rng, n_features, n_per_class)
        datos_x = hcat(datos_x, bloque)
        append!(labels, fill(c, n_per_class))
    end
    (datos_x, onehot(labels, 3))
end

"""
    generar_digitos_reducido(; n=500, n_features=20, n_clases=10, rng=Random.default_rng())

MNIST reducido sintético: simula PCA a n_features dimensiones, n_clases dígitos.
"""
function generar_digitos_reducido(; n::Int=500, n_features::Int=20, n_clases::Int=10, rng=Random.default_rng())
    n_per_class = n ÷ n_clases

    datos_x = zeros(n_features, 0)
    labels = Int[]
    for c in 1:n_clases
        centro = randn(rng, n_features) .* 3.0
        σ = abs.(randn(rng, n_features)) .* 0.8 .+ 0.5
        bloque = centro .+ σ .* randn(rng, n_features, n_per_class)
        datos_x = hcat(datos_x, bloque)
        append!(labels, fill(c, n_per_class))
    end
    (datos_x, onehot(labels, n_clases))
end

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Datasets sintéticos parametrizables
# ═══════════════════════════════════════════════════════════════════════════════

"""
    generar_blobs(; n=300, n_clases=5, n_features=2, separacion=3.0, rng=Random.default_rng())

Blobs gaussianos con N clusters. Control total sobre separabilidad.
"""
function generar_blobs(; n::Int=300, n_clases::Int=5, n_features::Int=2,
                        separacion::Float64=3.0, rng=Random.default_rng())
    n_per_class = n ÷ n_clases

    datos_x = zeros(n_features, 0)
    labels = Int[]
    for c in 1:n_clases
        centro = randn(rng, n_features) .* separacion
        bloque = centro .+ randn(rng, n_features, n_per_class)
        datos_x = hcat(datos_x, bloque)
        append!(labels, fill(c, n_per_class))
    end
    (datos_x, onehot(labels, n_clases))
end

"""
    generar_checkerboard(; n=600, tamano=4, rng=Random.default_rng())

Tablero de ajedrez 2D. Requiere descomposición espacial.
"""
function generar_checkerboard(; n::Int=600, tamano::Int=4, rng=Random.default_rng())
    x = rand(rng, 2, n) .* tamano
    labels = [Int(floor(x[1,i]) + floor(x[2,i])) % 2 for i in 1:n]
    (x, binarylabel(labels))
end

"""
    generar_anillos(; n=400, n_anillos=3, ruido=0.15, rng=Random.default_rng())

Anillos concéntricos. Cada anillo es una clase. Topología mínima crece con N.
"""
function generar_anillos(; n::Int=400, n_anillos::Int=3, ruido::Float64=0.15, rng=Random.default_rng())
    n_per_ring = n ÷ n_anillos

    datos_x = zeros(2, 0)
    labels = Int[]
    for r in 1:n_anillos
        θ = rand(rng, n_per_ring) .* 2π
        radio = Float64(r) .+ ruido .* randn(rng, n_per_ring)
        bloque = [radio' .* cos.(θ)'; radio' .* sin.(θ)']
        datos_x = hcat(datos_x, bloque)
        append!(labels, fill(r, n_per_ring))
    end
    (datos_x, onehot(labels, n_anillos))
end

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Robustez y edge cases
# ═══════════════════════════════════════════════════════════════════════════════

"""
    generar_con_ruido(; n=300, n_features_utiles=4, n_features_ruido=20, rng=Random.default_rng())

Dataset con features ruidosas que no aportan información.
"""
function generar_con_ruido(; n::Int=300, n_features_utiles::Int=4,
                            n_features_ruido::Int=20, rng=Random.default_rng())
    n_half = n ÷ 2
    # Features útiles: dos clusters separados
    x_util = vcat(
        randn(rng, n_features_utiles, n_half) .- 2.0,
        randn(rng, n_features_utiles, n - n_half) .+ 2.0
    )
    # Corregir: hcat en vez de vcat para column-major
    x_util = hcat(
        randn(rng, n_features_utiles, n_half) .- 2.0,
        randn(rng, n_features_utiles, n - n_half) .+ 2.0
    )
    # Features ruido: aleatorias sin correlación con la etiqueta
    x_ruido = randn(rng, n_features_ruido, n)
    datos_x = vcat(x_util, x_ruido)
    labels = vcat(zeros(Int, n_half), ones(Int, n - n_half))
    (datos_x, binarylabel(labels))
end

"""
    generar_desbalanceado(; n=500, ratio_minoritaria=0.1, n_features=4, rng=Random.default_rng())

Dataset desbalanceado: clase mayoritaria vs minoritaria.
"""
function generar_desbalanceado(; n::Int=500, ratio_minoritaria::Float64=0.1,
                                n_features::Int=4, rng=Random.default_rng())
    n_min = max(1, round(Int, n * ratio_minoritaria))
    n_may = n - n_min

    x_may = randn(rng, n_features, n_may)
    x_min = randn(rng, n_features, n_min) .+ 3.0

    datos_x = hcat(x_may, x_min)
    labels = vcat(zeros(Int, n_may), ones(Int, n_min))
    (datos_x, binarylabel(labels))
end

"""
    generar_alta_dimensionalidad(; n=100, n_features=100, n_clases=3, rng=Random.default_rng())

Alta dimensionalidad: muchas features, pocas muestras. Stress test de buffers.
"""
function generar_alta_dimensionalidad(; n::Int=100, n_features::Int=100,
                                       n_clases::Int=3, rng=Random.default_rng())
    n_per_class = n ÷ n_clases

    datos_x = zeros(n_features, 0)
    labels = Int[]
    for c in 1:n_clases
        centro = randn(rng, n_features) .* 2.0
        bloque = centro .+ randn(rng, n_features, n_per_class) .* 0.5
        datos_x = hcat(datos_x, bloque)
        append!(labels, fill(c, n_per_class))
    end
    (datos_x, onehot(labels, n_clases))
end
