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
Basado en las estadísticas reales del dataset UCI Iris.
"""
function generar_iris_sintetico(; n::Int=150, rng=Random.default_rng())
    n_per_class = n ÷ 3
    # Centroides y desviaciones basados en el Iris real
    centroides = [
        [5.006, 3.428, 1.462, 0.246],   # Setosa
        [5.936, 2.770, 4.260, 1.326],   # Versicolor
        [6.588, 2.974, 5.552, 2.026],   # Virginica
    ]
    σ = [
        [0.35, 0.38, 0.17, 0.11],
        [0.52, 0.31, 0.47, 0.20],
        [0.64, 0.32, 0.55, 0.27],
    ]

    datos_x = zeros(4, 0)
    labels = Int[]
    for (c, centro) in enumerate(centroides)
        bloque = centro .+ σ[c] .* randn(rng, 4, n_per_class)
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

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Datasets inspirados en los papers (UCI-like sintéticos)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    generar_glass_sintetico(; n=214, rng=Random.default_rng())

Glass sintético: 9 features, 6 clases. Pocas muestras por clase.
Basado en las estadísticas del UCI Glass Identification dataset.
El paper de DivisionNeuronal reporta +37pp vs red completa aquí.
"""
function generar_glass_sintetico(; n::Int=214, rng=Random.default_rng())
    n_clases = 6
    n_per_class = n ÷ n_clases
    n_features = 9

    # Centroides inspirados en las propiedades del vidrio real
    # RI, Na, Mg, Al, Si, K, Ca, Ba, Fe
    centroides = [
        [1.518, 13.0, 3.5, 1.2, 72.6, 0.6, 8.9, 0.0, 0.0],  # building_windows_float
        [1.517, 13.4, 3.5, 1.4, 72.9, 0.6, 8.4, 0.0, 0.1],  # building_windows_non_float
        [1.518, 13.3, 3.5, 1.4, 72.2, 0.6, 8.6, 0.0, 0.0],  # vehicle_windows
        [1.517, 14.8, 2.0, 3.5, 72.0, 0.0, 7.8, 0.0, 0.0],  # containers
        [1.517, 13.3, 0.0, 2.0, 73.0, 0.8, 11.0, 0.0, 0.0], # tableware
        [1.516, 14.5, 0.0, 2.2, 73.0, 0.0, 8.5, 1.6, 0.0],  # headlamps
    ]
    σ = [0.003, 0.8, 1.0, 0.5, 0.8, 0.3, 1.0, 0.5, 0.1]

    datos_x = zeros(n_features, 0)
    labels = Int[]
    for (c, centro) in enumerate(centroides)
        bloque = centro .+ σ .* randn(rng, n_features, n_per_class)
        datos_x = hcat(datos_x, bloque)
        append!(labels, fill(c, n_per_class))
    end
    (datos_x, onehot(labels, n_clases))
end

"""
    generar_ecoli_sintetico(; n=327, rng=Random.default_rng())

Ecoli sintético: 7 features, 5 clases. Pocas muestras, muchas clases.
Basado en el UCI Ecoli dataset (localización de proteínas).
El paper de DivisionNeuronal reporta +11pp vs red completa aquí.
"""
function generar_ecoli_sintetico(; n::Int=327, rng=Random.default_rng())
    n_clases = 5
    n_per_class = n ÷ n_clases
    n_features = 7

    # mcg, gvh, lip, chg, aac, alm1, alm2
    centroides = [
        [0.50, 0.47, 0.48, 0.50, 0.49, 0.56, 0.40],  # cp (cytoplasm)
        [0.72, 0.48, 0.48, 0.50, 0.54, 0.35, 0.40],  # im (inner membrane)
        [0.62, 0.47, 1.00, 0.50, 0.47, 0.45, 0.42],  # imL
        [0.44, 0.50, 0.48, 0.50, 0.42, 0.67, 0.36],  # om (outer membrane)
        [0.33, 0.47, 0.48, 0.50, 0.55, 0.78, 0.38],  # pp (periplasm)
    ]
    σ = [0.15, 0.10, 0.05, 0.05, 0.10, 0.15, 0.10]

    datos_x = zeros(n_features, 0)
    labels = Int[]
    for (c, centro) in enumerate(centroides)
        bloque = centro .+ σ .* randn(rng, n_features, n_per_class)
        datos_x = hcat(datos_x, bloque)
        append!(labels, fill(c, n_per_class))
    end
    (datos_x, onehot(labels, n_clases))
end

"""
    generar_seeds_sintetico(; n=210, rng=Random.default_rng())

Seeds sintético: 7 features, 3 clases (variedades de trigo).
Basado en el UCI Seeds dataset.
El paper de DivisionNeuronal identifica Area como feature dominante (94%).
"""
function generar_seeds_sintetico(; n::Int=210, rng=Random.default_rng())
    n_per_class = n ÷ 3
    n_features = 7

    # Area, Perimeter, Compactness, KernelLength, KernelWidth, Asymmetry, GrooveLength
    centroides = [
        [14.3, 14.1, 0.88, 5.5, 3.2, 2.7, 5.1],  # Kama
        [18.3, 16.1, 0.88, 6.2, 3.7, 3.6, 6.1],  # Rosa
        [11.9, 13.3, 0.85, 5.2, 2.9, 4.9, 4.9],  # Canadian
    ]
    σ = [1.5, 0.8, 0.02, 0.3, 0.2, 1.0, 0.3]

    datos_x = zeros(n_features, 0)
    labels = Int[]
    for (c, centro) in enumerate(centroides)
        bloque = centro .+ σ .* randn(rng, n_features, n_per_class)
        datos_x = hcat(datos_x, bloque)
        append!(labels, fill(c, n_per_class))
    end
    (datos_x, onehot(labels, 3))
end

"""
    generar_sonar_sintetico(; n=208, rng=Random.default_rng())

Sonar sintético: 10 features (reducido de 60), 2 clases.
Basado en el UCI Sonar dataset (minas vs rocas).
El paper de RandomCloud reporta +4.9pp y 87% reducción de parámetros aquí.
"""
function generar_sonar_sintetico(; n::Int=208, rng=Random.default_rng())
    n_half = n ÷ 2
    n_features = 10

    # Minas: señales más fuertes en frecuencias medias
    centro_minas = [0.3, 0.4, 0.5, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1]
    # Rocas: señales más uniformes
    centro_rocas = [0.2, 0.25, 0.3, 0.35, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]
    σ = fill(0.12, n_features)

    x_minas = centro_minas .+ σ .* randn(rng, n_features, n_half)
    x_rocas = centro_rocas .+ σ .* randn(rng, n_features, n - n_half)

    datos_x = hcat(x_minas, x_rocas)
    labels = vcat(zeros(Int, n_half), ones(Int, n - n_half))
    (datos_x, binarylabel(labels))
end

"""
    generar_ionosphere_sintetico(; n=351, rng=Random.default_rng())

Ionosphere sintético: 10 features (reducido de 34), 2 clases.
Basado en el UCI Ionosphere dataset (buenas vs malas señales de radar).
El paper de RandomCloud reporta 90% accuracy con 81% reducción aquí.
"""
function generar_ionosphere_sintetico(; n::Int=351, rng=Random.default_rng())
    n_good = round(Int, n * 0.64)  # 64% good en el dataset real
    n_bad = n - n_good
    n_features = 10

    centro_good = [0.6, 0.1, 0.5, 0.3, 0.4, 0.2, 0.3, 0.1, 0.2, 0.1]
    centro_bad = [0.1, 0.0, 0.2, 0.1, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0]
    σ = fill(0.2, n_features)

    x_good = centro_good .+ σ .* randn(rng, n_features, n_good)
    x_bad = centro_bad .+ σ .* randn(rng, n_features, n_bad)

    datos_x = hcat(x_good, x_bad)
    labels = vcat(ones(Int, n_good), zeros(Int, n_bad))
    (datos_x, binarylabel(labels))
end
