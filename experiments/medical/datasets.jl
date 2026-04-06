# datasets.jl — Datasets sintéticos de diagnóstico médico
#
# Basados en estadísticas publicadas de datasets clínicos reales.
# Cada dataset simula un escenario donde hay pocas muestras y muchas clases
# de diagnóstico — el sweet spot del pipeline RCND.
#
# Convención: datos_x (features, samples), datos_y (outputs, samples)

using Random
using LinearAlgebra

"""One-hot encoding."""
function onehot(labels::Vector{Int}, n_clases::Int)
    n = length(labels)
    y = zeros(Float64, n_clases, n)
    for i in 1:n
        y[labels[i], i] = 1.0
    end
    y
end

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Cáncer de mama — Wisconsin Breast Cancer (simplificado)
#    10 features citológicas, 2 clases (benigno/maligno)
#    569 muestras en el real. Bien documentado.
# ═══════════════════════════════════════════════════════════════════════════════

"""
    generar_cancer_mama(; n=569, rng)

10 features: radius, texture, perimeter, area, smoothness,
compactness, concavity, concave_points, symmetry, fractal_dimension.
Estadísticas basadas en el UCI Wisconsin Breast Cancer dataset.
"""
function generar_cancer_mama(; n::Int=569, rng=Random.default_rng())
    n_maligno = round(Int, n * 0.37)  # 37% malignos en el real
    n_benigno = n - n_maligno

    # Centroides y σ basados en las medias/std reales del dataset
    #          radius  texture perim   area   smooth compact concav  conc_pt symmetry fractal
    c_benign = [12.15, 17.91, 78.08, 462.8, 0.092, 0.080, 0.046, 0.026, 0.174, 0.063]
    c_malig  = [17.46, 21.60, 115.4, 978.4, 0.103, 0.145, 0.161, 0.088, 0.193, 0.063]
    σ_benign = [ 1.78,  4.00, 11.80, 134.3, 0.014, 0.034, 0.039, 0.016, 0.027, 0.007]
    σ_malig  = [ 3.20,  3.78, 21.80, 368.2, 0.014, 0.049, 0.075, 0.035, 0.023, 0.006]

    x_b = c_benign .+ σ_benign .* randn(rng, 10, n_benigno)
    x_m = c_malig  .+ σ_malig  .* randn(rng, 10, n_maligno)

    datos_x = hcat(x_b, x_m)
    labels = vcat(zeros(Int, n_benigno), ones(Int, n_maligno))

    # Normalizar features a [0, 1] para sigmoid
    for i in 1:10
        mi, ma = extrema(datos_x[i, :])
        if ma > mi
            datos_x[i, :] .= (datos_x[i, :] .- mi) ./ (ma - mi)
        end
    end

    (datos_x, reshape(Float64.(labels), 1, :))
end

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Enfermedad cardíaca — Cleveland Heart Disease
#    13 features clínicas, 2 clases (presencia/ausencia)
#    303 muestras en el real. Clásico en ML médico.
# ═══════════════════════════════════════════════════════════════════════════════

"""
    generar_cardiopatia(; n=303, rng)

13 features: age, sex, chest_pain_type, resting_bp, cholesterol,
fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak,
st_slope, n_major_vessels, thal.
"""
function generar_cardiopatia(; n::Int=303, rng=Random.default_rng())
    n_enfermo = round(Int, n * 0.46)  # 46% con enfermedad en Cleveland
    n_sano = n - n_enfermo

    # Centroides basados en estadísticas publicadas del dataset Cleveland
    #          age    sex   cp    bp    chol   fbs   ecg   maxhr  exang oldpk slope vessels thal
    c_sano   = [52.6, 0.56, 2.8, 129.3, 243.5, 0.14, 0.84, 158.6, 0.14, 0.59, 1.40, 0.27, 3.77]
    c_enferm = [56.6, 0.82, 3.6, 134.6, 251.5, 0.16, 1.17, 139.3, 0.55, 1.59, 1.83, 1.17, 5.80]
    σ_sano   = [ 9.5, 0.50, 0.9,  16.0,  52.0, 0.35, 0.95,  19.0, 0.35, 0.80, 0.60, 0.60, 1.60]
    σ_enferm = [ 8.0, 0.39, 0.7,  18.5,  50.0, 0.37, 0.80,  22.0, 0.50, 1.20, 0.55, 0.90, 1.40]

    x_s = c_sano   .+ σ_sano   .* randn(rng, 13, n_sano)
    x_e = c_enferm .+ σ_enferm .* randn(rng, 13, n_enfermo)

    datos_x = hcat(x_s, x_e)
    labels = vcat(zeros(Int, n_sano), ones(Int, n_enfermo))

    for i in 1:13
        mi, ma = extrema(datos_x[i, :])
        if ma > mi
            datos_x[i, :] .= (datos_x[i, :] .- mi) ./ (ma - mi)
        end
    end

    (datos_x, reshape(Float64.(labels), 1, :))
end

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Enfermedad hepática — Indian Liver Patient Dataset
#    10 features bioquímicas, 2 clases (enfermo/sano)
#    583 muestras, 71% enfermos (desbalanceado).
# ═══════════════════════════════════════════════════════════════════════════════

"""
    generar_hepatopatia(; n=583, rng)

10 features: age, gender, total_bilirubin, direct_bilirubin,
alkaline_phosphatase, alamine_aminotransferase, aspartate_aminotransferase,
total_proteins, albumin, albumin_globulin_ratio.
"""
function generar_hepatopatia(; n::Int=583, rng=Random.default_rng())
    n_enfermo = round(Int, n * 0.71)
    n_sano = n - n_enfermo

    #          age   sex   tbil  dbil  alkphos  alt    ast    tprot  alb   ag_ratio
    c_enferm = [44.7, 0.76, 3.30, 1.49, 290.9, 80.7, 109.9, 6.48, 3.14, 0.95]
    c_sano   = [38.2, 0.58, 0.90, 0.30, 195.0, 25.0,  28.0, 6.80, 3.50, 1.10]
    σ_enferm = [16.0, 0.43, 6.20, 2.80, 243.0, 182., 221.0, 1.09, 0.82, 0.32]
    σ_sano   = [15.0, 0.50, 0.50, 0.20, 100.0, 15.0,  12.0, 0.90, 0.60, 0.25]

    x_e = c_enferm .+ σ_enferm .* randn(rng, 10, n_enfermo)
    x_s = c_sano   .+ σ_sano   .* randn(rng, 10, n_sano)

    datos_x = hcat(x_e, x_s)
    labels = vcat(ones(Int, n_enfermo), zeros(Int, n_sano))

    for i in 1:10
        mi, ma = extrema(datos_x[i, :])
        if ma > mi
            datos_x[i, :] .= (datos_x[i, :] .- mi) ./ (ma - mi)
        end
    end

    (datos_x, reshape(Float64.(labels), 1, :))
end

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Diabetes tipo 2 — Pima Indians (clásico)
#    8 features, 2 clases, 768 muestras, 35% positivos.
# ═══════════════════════════════════════════════════════════════════════════════

"""
    generar_diabetes(; n=768, rng)

8 features: pregnancies, glucose, blood_pressure, skin_thickness,
insulin, bmi, diabetes_pedigree, age.
"""
function generar_diabetes(; n::Int=768, rng=Random.default_rng())
    n_pos = round(Int, n * 0.35)
    n_neg = n - n_pos

    #          preg  glucose  bp     skin   insulin  bmi    pedigree  age
    c_pos    = [4.87, 141.3, 70.8, 22.2, 100.3, 35.4, 0.55, 37.1]
    c_neg    = [3.30, 110.0, 68.2, 19.7,  68.8, 30.3, 0.43, 31.2]
    σ_pos    = [3.74,  31.9, 21.5, 17.7, 138.7,  7.3, 0.37, 10.9]
    σ_neg    = [3.02,  26.1, 18.0, 14.9, 100.0,  7.7, 0.30,  9.1]

    x_p = c_pos .+ σ_pos .* randn(rng, 8, n_pos)
    x_n = c_neg .+ σ_neg .* randn(rng, 8, n_neg)

    datos_x = hcat(x_p, x_n)
    labels = vcat(ones(Int, n_pos), zeros(Int, n_neg))

    for i in 1:8
        mi, ma = extrema(datos_x[i, :])
        if ma > mi
            datos_x[i, :] .= (datos_x[i, :] .- mi) ./ (ma - mi)
        end
    end

    (datos_x, reshape(Float64.(labels), 1, :))
end

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Enfermedad renal crónica — CKD Dataset
#    11 features numéricas, 2 clases, 400 muestras, 62% CKD.
#    Interesante porque tiene features muy discriminativas (hemoglobina, albumina).
# ═══════════════════════════════════════════════════════════════════════════════

"""
    generar_enfermedad_renal(; n=400, rng)

11 features: age, blood_pressure, specific_gravity, albumin, sugar,
blood_glucose, blood_urea, serum_creatinine, sodium, potassium, hemoglobin.
"""
function generar_enfermedad_renal(; n::Int=400, rng=Random.default_rng())
    n_ckd = round(Int, n * 0.62)
    n_sano = n - n_ckd

    #          age    bp     sg     alb   sugar  bg     bu     sc     na     k      hemo
    c_ckd    = [55.6, 77.3, 1.017, 1.53, 0.45, 148.0, 57.4, 4.07, 137.5, 4.63, 10.9]
    c_sano   = [46.0, 74.0, 1.022, 0.10, 0.05, 105.0, 32.0, 1.00, 141.0, 4.40, 15.0]
    σ_ckd    = [17.0, 13.0, 0.006, 1.20, 0.80,  79.0, 25.0, 3.70,   6.0, 1.10,  2.5]
    σ_sano   = [14.0, 10.0, 0.004, 0.20, 0.10,  20.0, 10.0, 0.30,   4.0, 0.40,  1.5]

    x_c = c_ckd  .+ σ_ckd  .* randn(rng, 11, n_ckd)
    x_s = c_sano .+ σ_sano .* randn(rng, 11, n_sano)

    datos_x = hcat(x_c, x_s)
    labels = vcat(ones(Int, n_ckd), zeros(Int, n_sano))

    for i in 1:11
        mi, ma = extrema(datos_x[i, :])
        if ma > mi
            datos_x[i, :] .= (datos_x[i, :] .- mi) ./ (ma - mi)
        end
    end

    (datos_x, reshape(Float64.(labels), 1, :))
end

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Dermatología — Differential Diagnosis of Erythemato-Squamous Diseases
#    12 features clínicas, 6 clases de enfermedad cutánea, 366 muestras.
#    Muchas clases + pocas muestras = escenario ideal para División Neuronal.
# ═══════════════════════════════════════════════════════════════════════════════

"""
    generar_dermatologia(; n=366, rng)

12 features histopatológicas simplificadas, 6 clases:
psoriasis, seborrheic_dermatitis, lichen_planus,
pityriasis_rosea, chronic_dermatitis, pityriasis_rubra.
"""
function generar_dermatologia(; n::Int=366, rng=Random.default_rng())
    n_clases = 6
    n_per = n ÷ n_clases
    n_features = 12

    # Cada enfermedad tiene un perfil histopatológico diferente
    # erythema, scaling, borders, itching, koebner, polygonal_papules,
    # follicular_papules, oral_mucosal, knee_elbow, scalp, family_history, age
    centroides = [
        [2.5, 2.7, 2.3, 1.8, 1.5, 0.2, 0.1, 0.1, 2.0, 2.2, 1.0, 35.0],  # psoriasis
        [2.2, 2.5, 1.5, 1.5, 0.1, 0.1, 0.1, 0.1, 0.3, 2.5, 0.3, 40.0],  # seb_derm
        [2.0, 1.0, 2.0, 2.0, 1.0, 2.5, 0.1, 1.5, 0.5, 0.3, 0.2, 45.0],  # lichen
        [2.0, 1.5, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 25.0],  # pityriasis_r
        [1.5, 1.0, 0.5, 2.5, 0.1, 0.1, 0.1, 0.1, 0.5, 0.3, 0.5, 50.0],  # chronic_derm
        [2.5, 2.0, 1.0, 0.5, 0.1, 0.1, 2.0, 0.1, 1.0, 0.5, 0.2, 55.0],  # pityriasis_rp
    ]
    σ = [0.8, 0.8, 0.7, 0.8, 0.6, 0.5, 0.4, 0.4, 0.7, 0.8, 0.4, 12.0]

    datos_x = zeros(n_features, 0)
    labels = Int[]
    for (c, centro) in enumerate(centroides)
        bloque = centro .+ σ .* randn(rng, n_features, n_per)
        datos_x = hcat(datos_x, bloque)
        append!(labels, fill(c, n_per))
    end

    for i in 1:n_features
        mi, ma = extrema(datos_x[i, :])
        if ma > mi
            datos_x[i, :] .= (datos_x[i, :] .- mi) ./ (ma - mi)
        end
    end

    (datos_x, onehot(labels, n_clases))
end

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Tiroides — Thyroid Disease (New Thyroid)
#    5 features, 3 clases (normal, hiper, hipo), 215 muestras.
#    Pocas features, 3 clases, pocas muestras — ideal para el pipeline.
# ═══════════════════════════════════════════════════════════════════════════════

"""
    generar_tiroides(; n=215, rng)

5 features: T3_resin_uptake, total_serum_thyroxin, total_serum_triiodothyronine,
TSH, max_diff_TSH.
3 clases: normal, hipertiroidismo, hipotiroidismo.
"""
function generar_tiroides(; n::Int=215, rng=Random.default_rng())
    n_per = n ÷ 3

    #          T3_resin  thyroxin  triiodo  TSH    max_TSH
    c_normal = [109.0,    9.8,     2.1,    2.1,    17.0]
    c_hiper  = [127.0,   18.5,     4.5,    0.6,     3.0]
    c_hipo   = [ 96.0,    4.0,     1.0,   12.0,    40.0]
    σ        = [ 12.0,    4.5,     1.2,    3.5,    12.0]

    x_n = c_normal .+ σ .* randn(rng, 5, n_per)
    x_hi = c_hiper .+ σ .* randn(rng, 5, n_per)
    x_ho = c_hipo  .+ σ .* randn(rng, 5, n - 2*n_per)

    datos_x = hcat(x_n, x_hi, x_ho)
    labels = vcat(fill(1, n_per), fill(2, n_per), fill(3, n - 2*n_per))

    for i in 1:5
        mi, ma = extrema(datos_x[i, :])
        if ma > mi
            datos_x[i, :] .= (datos_x[i, :] .- mi) ./ (ma - mi)
        end
    end

    (datos_x, onehot(labels, 3))
end
