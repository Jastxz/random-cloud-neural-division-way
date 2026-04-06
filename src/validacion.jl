# validacion.jl — Validación de datos de entrada del pipeline RCND

"""
    validar_datos!(datos_x::AbstractMatrix, datos_y::AbstractMatrix)

Valida que los datos de entrada cumplen las invariantes requeridas por el pipeline RCND.

Convención column-major:
- `datos_x`: `(n_features, n_samples)` — cada columna es una muestra.
- `datos_y`: `(n_outputs, n_samples)` — cada columna es una etiqueta.

# Errores
- `ArgumentError` si los datos están vacíos (0 muestras).
- `ArgumentError` si el número de muestras (columnas) de `datos_x` y `datos_y` no coincide.
"""
function validar_datos!(datos_x::AbstractMatrix, datos_y::AbstractMatrix)
    n_samples_x = size(datos_x, 2)
    n_samples_y = size(datos_y, 2)

    if n_samples_x == 0
        throw(ArgumentError(
            "Los datos de entrada están vacíos: datos_x tiene 0 muestras (columnas). " *
            "Se requiere al menos 1 muestra."
        ))
    end

    if n_samples_x != n_samples_y
        throw(ArgumentError(
            "Dimensiones inconsistentes: datos_x tiene $n_samples_x muestras (columnas) " *
            "pero datos_y tiene $n_samples_y muestras (columnas). " *
            "Ambos deben tener el mismo número de columnas (n_samples)."
        ))
    end

    nothing
end
