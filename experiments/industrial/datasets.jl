# datasets.jl — Datasets sintéticos de control de calidad industrial
#
# Basados en escenarios reales de manufactura y monitoreo de procesos.
# Cada dataset simula sensores en una línea de producción con
# diferentes tipos de defectos (clases) y pocas muestras de fallo.
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
# 1. Defectos en chapa de acero — Steel Plates Faults
#    7 tipos de defecto, 13 features geométricas, 1941 muestras.
#    Basado en el UCI Steel Plates Faults dataset.
# ═══════════════════════════════════════════════════════════════════════════════

"""
    generar_defectos_acero(; n=500, rng)

13 features: x_min, x_max, y_min, y_max, pixels_area, x_perimeter,
y_perimeter, sum_luminosity, min_luminosity, max_luminosity,
length_convex_hull, steel_type, thickness.
7 clases de defecto: pastry, z_scratch, k_scratch, stains,
dirtiness, bumps, other.
"""
function generar_defectos_acero(; n::Int=500, rng=Random.default_rng())
    n_clases = 7
    n_per = n ÷ n_clases
    n_features = 13

    # Cada tipo de defecto tiene un perfil geométrico diferente
    centroides = [
        # x_min x_max y_min y_max area  xper  yper  lum   lmin  lmax  hull  type  thick
        [100., 200., 50.,  150., 5000., 300., 200., 120., 80.,  160., 250., 1.0, 80.],  # pastry
        [ 50., 400., 10.,   20., 2000., 500.,  30., 100., 60.,  140., 450., 1.0, 60.],  # z_scratch
        [ 80., 350., 20.,   40., 3000., 400.,  50., 110., 70.,  150., 380., 2.0, 70.],  # k_scratch
        [150., 250., 100., 200., 8000., 250., 250., 140., 100., 180., 200., 1.0, 90.],  # stains
        [120., 180., 80.,  120., 1500., 150., 100.,  90., 50.,  130., 130., 2.0, 75.],  # dirtiness
        [200., 300., 150., 250., 10000.,350., 300., 160., 120., 200., 320., 1.0, 100.], # bumps
        [ 70., 170., 60.,  100., 4000., 200., 150., 105., 65.,  145., 180., 1.5, 85.],  # other
    ]
    σ = [40., 60., 30., 40., 2000., 80., 60., 25., 20., 25., 60., 0.5, 15.]

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
# 2. Monitoreo de rodamientos — Bearing Fault Detection
#    8 features vibracionales, 4 clases (normal + 3 tipos de fallo).
#    Basado en el CWRU Bearing Dataset (Case Western Reserve University).
#    Escenario típico de mantenimiento predictivo.
# ═══════════════════════════════════════════════════════════════════════════════

"""
    generar_fallos_rodamiento(; n=400, ratio_fallo=0.3, rng)

8 features: rms_vibration, peak_vibration, crest_factor, kurtosis,
skewness, spectral_centroid, spectral_bandwidth, temperature.
4 clases: normal, inner_race_fault, outer_race_fault, ball_fault.
ratio_fallo controla el desbalance (30% fallos por defecto = realista).
"""
function generar_fallos_rodamiento(; n::Int=400, ratio_fallo::Float64=0.3, rng=Random.default_rng())
    n_fallo_total = round(Int, n * ratio_fallo)
    n_normal = n - n_fallo_total
    n_per_fallo = n_fallo_total ÷ 3

    #          rms    peak   crest  kurt   skew   spec_c  spec_bw  temp
    c_normal = [0.15, 0.45, 3.0,  3.0,  0.0,  150.0, 80.0,  45.0]
    c_inner  = [0.85, 2.50, 2.9,  5.5,  0.8,  280.0, 120.0, 52.0]
    c_outer  = [0.65, 1.80, 2.8,  4.8, -0.3,  220.0, 150.0, 48.0]
    c_ball   = [0.45, 1.20, 2.7,  4.0,  0.2,  350.0, 200.0, 47.0]
    σ        = [0.12, 0.35, 0.4,  0.8,  0.3,   50.0,  40.0,  3.0]

    x_n = c_normal .+ σ .* randn(rng, 8, n_normal)
    x_i = c_inner  .+ σ .* randn(rng, 8, n_per_fallo)
    x_o = c_outer  .+ σ .* randn(rng, 8, n_per_fallo)
    x_b = c_ball   .+ σ .* randn(rng, 8, n_fallo_total - 2*n_per_fallo)

    datos_x = hcat(x_n, x_i, x_o, x_b)
    labels = vcat(fill(1, n_normal), fill(2, n_per_fallo),
                  fill(3, n_per_fallo), fill(4, n_fallo_total - 2*n_per_fallo))

    for i in 1:8
        mi, ma = extrema(datos_x[i, :])
        if ma > mi
            datos_x[i, :] .= (datos_x[i, :] .- mi) ./ (ma - mi)
        end
    end

    (datos_x, onehot(labels, 4))
end

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Calidad de soldadura — Weld Quality Classification
#    10 features de proceso, 4 clases (buena + 3 defectos).
#    Basado en parámetros reales de soldadura por arco.
# ═══════════════════════════════════════════════════════════════════════════════

"""
    generar_calidad_soldadura(; n=300, rng)

10 features: voltage, current, wire_feed_speed, travel_speed,
gas_flow_rate, contact_tip_distance, arc_length, heat_input,
interpass_temperature, humidity.
4 clases: good, porosity, lack_of_fusion, crack.
"""
function generar_calidad_soldadura(; n::Int=300, rng=Random.default_rng())
    n_good = round(Int, n * 0.5)
    n_defect = n - n_good
    n_per_defect = n_defect ÷ 3

    #          volt   curr   wfs    ts     gas    ctd    arc    heat   interp humid
    c_good   = [25.0, 220., 8.0,  30.0, 18.0, 15.0, 3.0, 1.5, 150., 45.]
    c_poros  = [23.0, 200., 7.0,  35.0, 12.0, 18.0, 4.0, 1.1, 120., 70.]  # bajo gas, alta humedad
    c_fusion = [22.0, 180., 6.0,  40.0, 18.0, 20.0, 5.0, 0.8, 100., 45.]  # baja corriente, alta velocidad
    c_crack  = [28.0, 260., 9.0,  25.0, 18.0, 12.0, 2.5, 2.2, 250., 40.]  # alto calor, alta interpass
    σ        = [ 2.0,  20., 1.0,   5.0,  2.0,  2.0, 0.5, 0.3,  30., 10.]

    x_g = c_good   .+ σ .* randn(rng, 10, n_good)
    x_p = c_poros  .+ σ .* randn(rng, 10, n_per_defect)
    x_f = c_fusion .+ σ .* randn(rng, 10, n_per_defect)
    x_c = c_crack  .+ σ .* randn(rng, 10, n_defect - 2*n_per_defect)

    datos_x = hcat(x_g, x_p, x_f, x_c)
    labels = vcat(fill(1, n_good), fill(2, n_per_defect),
                  fill(3, n_per_defect), fill(4, n_defect - 2*n_per_defect))

    for i in 1:10
        mi, ma = extrema(datos_x[i, :])
        if ma > mi
            datos_x[i, :] .= (datos_x[i, :] .- mi) ./ (ma - mi)
        end
    end

    (datos_x, onehot(labels, 4))
end

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Calidad de vino — Wine Quality (simplificado)
#    11 features fisicoquímicas, 3 clases (malo/medio/bueno).
#    Basado en el UCI Wine Quality dataset (vino tinto).
# ═══════════════════════════════════════════════════════════════════════════════

"""
    generar_calidad_vino(; n=500, rng)

11 features: fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
chlorides, free_so2, total_so2, density, pH, sulphates, alcohol.
3 clases: malo (quality 3-4), medio (5-6), bueno (7-8).
"""
function generar_calidad_vino(; n::Int=500, rng=Random.default_rng())
    n_malo = round(Int, n * 0.15)
    n_bueno = round(Int, n * 0.15)
    n_medio = n - n_malo - n_bueno

    #          facid  vacid  citric sugar  chlor  fso2   tso2   dens   pH     sulph  alc
    c_malo  = [ 8.5, 0.68, 0.22,  2.8, 0.10, 10.0,  40.0, 0.998, 3.35, 0.55, 9.5]
    c_medio = [ 8.3, 0.53, 0.27,  2.5, 0.09, 15.0,  46.0, 0.997, 3.31, 0.62, 10.4]
    c_bueno = [ 8.1, 0.37, 0.34,  2.7, 0.08, 14.0,  35.0, 0.996, 3.28, 0.74, 11.5]
    σ       = [ 1.7, 0.18, 0.20,  1.4, 0.05,  10.0,  33.0, 0.002, 0.15, 0.17, 1.1]

    x_m = c_malo  .+ σ .* randn(rng, 11, n_malo)
    x_d = c_medio .+ σ .* randn(rng, 11, n_medio)
    x_b = c_bueno .+ σ .* randn(rng, 11, n_bueno)

    datos_x = hcat(x_m, x_d, x_b)
    labels = vcat(fill(1, n_malo), fill(2, n_medio), fill(3, n_bueno))

    for i in 1:11
        mi, ma = extrema(datos_x[i, :])
        if ma > mi
            datos_x[i, :] .= (datos_x[i, :] .- mi) ./ (ma - mi)
        end
    end

    (datos_x, onehot(labels, 3))
end

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Semiconductores — Wafer Map Defect Classification
#    8 features estadísticas de wafer maps, 5 clases de defecto.
#    Basado en el WM-811K Wafer Map dataset.
#    Muy desbalanceado: 80% normal, 20% defectos repartidos.
# ═══════════════════════════════════════════════════════════════════════════════

"""
    generar_defectos_wafer(; n=400, rng)

8 features: density, n_regions, max_region_size, eccentricity,
solidity, extent, perimeter_ratio, radial_distribution.
5 clases: normal, center, edge_loc, edge_ring, scratch.
"""
function generar_defectos_wafer(; n::Int=400, rng=Random.default_rng())
    n_normal = round(Int, n * 0.60)
    n_defect = n - n_normal
    n_per = n_defect ÷ 4

    #          dens   nreg   maxreg  ecc    solid  extent perrat  radial
    c_normal = [0.02, 1.0,  50.,  0.3, 0.95, 0.85, 0.10, 0.50]
    c_center = [0.15, 1.5, 200.,  0.2, 0.90, 0.70, 0.25, 0.15]  # concentrado en centro
    c_edge_l = [0.08, 3.0, 100.,  0.7, 0.60, 0.40, 0.50, 0.85]  # en borde, localizado
    c_edge_r = [0.12, 1.2, 300.,  0.1, 0.80, 0.75, 0.60, 0.90]  # anillo en borde
    c_scratc = [0.05, 2.0, 150.,  0.9, 0.30, 0.20, 0.80, 0.50]  # lineal, alta excentricidad
    σ        = [0.03, 0.8,  60.,  0.15, 0.10, 0.12, 0.12, 0.15]

    x_n = c_normal .+ σ .* randn(rng, 8, n_normal)
    x_c = c_center .+ σ .* randn(rng, 8, n_per)
    x_l = c_edge_l .+ σ .* randn(rng, 8, n_per)
    x_r = c_edge_r .+ σ .* randn(rng, 8, n_per)
    x_s = c_scratc .+ σ .* randn(rng, 8, n_defect - 3*n_per)

    datos_x = hcat(x_n, x_c, x_l, x_r, x_s)
    labels = vcat(fill(1, n_normal), fill(2, n_per), fill(3, n_per),
                  fill(4, n_per), fill(5, n_defect - 3*n_per))

    for i in 1:8
        mi, ma = extrema(datos_x[i, :])
        if ma > mi
            datos_x[i, :] .= (datos_x[i, :] .- mi) ./ (ma - mi)
        end
    end

    (datos_x, onehot(labels, 5))
end

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Mantenimiento predictivo de turbinas — Turbine Condition Monitoring
#    6 features de sensores, 3 clases (normal, degradado, fallo inminente).
#    Escenario edge/IoT: la red mínima debe caber en un microcontrolador.
# ═══════════════════════════════════════════════════════════════════════════════

"""
    generar_turbina(; n=300, rng)

6 features: rotor_speed, exhaust_temp, vibration_level,
oil_pressure, fuel_flow, power_output.
3 clases: normal, degraded, imminent_failure.
"""
function generar_turbina(; n::Int=300, rng=Random.default_rng())
    n_normal = round(Int, n * 0.6)
    n_degraded = round(Int, n * 0.25)
    n_failure = n - n_normal - n_degraded

    #          rpm     exhaust  vibr   oil_p  fuel   power
    c_normal = [3000., 450.,   0.5,  45.0, 120., 95.0]
    c_degrad = [2850., 480.,   1.2,  38.0, 125., 88.0]
    c_fail   = [2600., 520.,   2.5,  30.0, 135., 75.0]
    σ        = [ 100.,  20.,   0.3,   4.0,   8.,  5.0]

    x_n = c_normal .+ σ .* randn(rng, 6, n_normal)
    x_d = c_degrad .+ σ .* randn(rng, 6, n_degraded)
    x_f = c_fail   .+ σ .* randn(rng, 6, n_failure)

    datos_x = hcat(x_n, x_d, x_f)
    labels = vcat(fill(1, n_normal), fill(2, n_degraded), fill(3, n_failure))

    for i in 1:6
        mi, ma = extrema(datos_x[i, :])
        if ma > mi
            datos_x[i, :] .= (datos_x[i, :] .- mi) ./ (ma - mi)
        end
    end

    (datos_x, onehot(labels, 3))
end
