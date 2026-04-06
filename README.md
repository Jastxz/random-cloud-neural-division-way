# RandomCloudNeuralDivision.jl (RCND)

Pipeline unificado que combina dos métodos de búsqueda de arquitecturas neuronales:

1. **Random Cloud** — Encuentra la topología mínima de una red neuronal sin entrenarla. Genera una nube de redes con pesos aleatorios, evalúa cuáles clasifican mejor *sin backpropagation*, reduce progresivamente la topología eliminando neuronas, y solo entrena la mejor candidata al final.

2. **División Neuronal** — Descompone la red encontrada en subredes especializadas mediante búsqueda exhaustiva sobre todos los subconjuntos de entradas y salidas. Produce un *Mapa de Soluciones* que revela qué features son necesarias para qué clases, si el problema es descomponible, y cuál es la subred más eficiente.

El resultado es una red mínima + un mapa de soluciones que no solo clasifica, sino que explica la estructura interna del problema.

## Instalación

```julia
# Clonar el repositorio
git clone https://github.com/Jastxz/random-cloud-neural-division-way.git
cd random-cloud-neural-division-way

# Instalar dependencias
julia -e 'using Pkg; Pkg.add(["JSON3", "StructTypes", "KernelAbstractions", "JLD2"])'
```

Requiere Julia ≥ 1.9.

## Uso rápido

```julia
include("benchmarks/stubs.jl")

using .RCND
using Random

# Datos: XOR (2 features, 1 output, 200 muestras)
rng = MersenneTwister(42)
datos_x = randn(rng, 2, 200)
datos_y = reshape(Float64[Int((x[1]>0) ⊻ (x[2]>0)) for x in eachcol(datos_x)], 1, :)

# Ejecutar pipeline completo
informe = RCND.ejecutar_benchmark_rcnd(datos_x, datos_y;
    tamano_nube = 50,
    topologia_inicial = [2, 8, 1],
    umbral_acierto = 0.5,
    epocas_refinamiento = 1000,
    umbral_division = 0.4,
)

# Resultados
println("Precisión Fase Nube: $(informe.precision_nube)")
println("Topología encontrada: $(informe.topologia_final)")
println("Parámetros: $(informe.n_parametros_nube)")
println("División ejecutada: $(informe.division_ejecutada)")
```

## Cómo funciona

```
┌─────────────┐     ┌───────────┐     ┌──────────────────┐
│  Datos (X,Y) │────▶│ Fase Nube │────▶│ Fase División    │
└─────────────┘     │           │     │                  │
                    │ N redes   │     │ Búsqueda         │
                    │ aleatorias│     │ exhaustiva de     │
                    │ → evaluar │     │ subredes          │
                    │ → reducir │     │                  │
                    │ → entrenar│     │ → Mapa de        │
                    │ la mejor  │     │   Soluciones     │
                    └───────────┘     └──────────────────┘
                         │                    │
                         ▼                    ▼
                    Red mínima          Subredes
                    [10→2→4]           especializadas
                    28 params          por subproblema
```

**Fase Nube (RandomCloud.jl):**
- Genera N redes con pesos aleatorios U(-1,1)
- Evalúa cada una con forward pass (sin backpropagation)
- Reduce progresivamente eliminando neuronas de la última capa oculta
- Entrena solo la mejor candidata mínima

**Fase División (DivisionNeuronal.jl):**
- Toma la red encontrada y explora todos los subconjuntos de entradas × salidas
- Evalúa cada subconfiguracion con pesos aleatorios (forward pass)
- Retiene la más simple que supere un umbral de precisión
- Entrena las subredes seleccionadas con Adam + early stopping
- Produce un Mapa de Soluciones: {subconjunto de salidas → subred más eficiente}

## Resultados

### Diagnóstico médico

| Dataset | Features | Muestras | Clases | Precisión Nube | Topología | Params |
|---|---|---|---|---|---|---|
| Cáncer de mama | 10 | 569 | 2 | 100% | 10→8→1 | 97 |
| Cardiopatía | 13 | 303 | 2 | 100% | 13→10→1 | 151 |
| Hepatopatía | 10 | 583 | 2 | 100% | 10→8→1 | 97 |
| Diabetes | 8 | 768 | 2 | 100% | 8→6→1 | 61 |
| Enfermedad renal | 11 | 400 | 2 | 100% | 11→8→1 | 105 |
| Tiroides | 5 | 215 | 3 | 94% | 5→5→3 | 48 |
| Tiroides 2-capas | 5 | 215 | 3 | 94.4% | 5→8→6→3 | 123 |
| Dermatología | 12 | 198 | 6 | 96.5% | 12→4→6 | 82 |

### Control de calidad industrial

| Dataset | Features | Muestras | Clases | Precisión Nube | Topología | Params | División |
|---|---|---|---|---|---|---|---|
| Turbina | 6 | 300 | 3 | 97.7% | 6→3→3 | 30 | 91.7% (7 parciales) |
| Rodamientos | 8 | 400 | 4 | 97.5% | 8→2→4 | 28 | 90.0% (15 parciales) |
| Rodamientos 2-capas | 8 | 400 | 4 | 100% | 8→10→5→4 | — | 90.0% (15 parciales) |
| Wafer | 8 | 400 | 5 | 100% | 8→6→5 | — | 90.0% (31 parciales) |

Hallazgos clave:
- **Rodamientos**: redujo de [8,8,4] a [8,2,4] — solo 2 neuronas ocultas (28 parámetros) con 97.5% de precisión. Cabe en cualquier microcontrolador.
- **Turbina**: [6,3,3] con 30 parámetros y 97.7% — ideal para edge/IoT.
- **Dermatología**: 6 clases de enfermedad cutánea con solo [12,4,6] (82 parámetros) al 96.5%.
- La División Neuronal encontró 15 soluciones parciales en rodamientos, revelando qué sensores importan para cada tipo de fallo.

### Datasets de los papers originales

| Dataset | Precisión Nube | Topología | División global | Parciales |
|---|---|---|---|---|
| Iris (4f, 3c) | 85.3% | 4→7→3 | 95.3% | 7 |
| Wine (13f, 3c) | 100% | 13→10→3 | 100% | 7 |
| Ecoli (7f, 5c) | 64% | 7→4→5 | 80.6% | 31 |
| Seeds (7f, 3c) | 63.8% | 7→2→3 | 66.7% | 7 |
| Sonar (10f, 2c) | 100% | 10→8→1 | — | — |

## Estructura del proyecto

```
src/
├── RandomCloud/              # Paquete: NAS por nubes aleatorias
├── NeuralDivision/           # Paquete: búsqueda exhaustiva de subredes
├── RandomCloudNeuralDivision.jl  # Módulo principal del pipeline
├── tipos.jl                  # Tipos: Configuracion_RCND, TopologiaOptima, Informe_RCND
├── validacion.jl             # Validación de datos de entrada
├── adaptador.jl              # Conversión RedNeuronal → RedBase
├── backend.jl                # Selección CPU/GPU
├── buffers.jl                # Pre-alocación de buffers
├── motor.jl                  # Motor_RCND, ejecutar_pipeline, fases individuales
├── informe.jl                # resumen() textual
└── serializacion.jl          # JSON ida y vuelta

test/                         # 447 tests (unitarios + 10 propiedades PBT)
benchmarks/                   # 26 benchmarks sintéticos
experiments/
├── medical/                  # 9 experimentos de diagnóstico médico
└── industrial/               # 10 experimentos de control de calidad
papers/                       # Papers originales de ambos métodos
```

## Tests

```bash
# Tests unitarios + propiedades (Supposition.jl)
julia --project=test test/runtests.jl

# Benchmarks
julia benchmarks/run_benchmarks.jl              # Todos (26 datasets)
julia benchmarks/run_benchmarks.jl binarios     # Solo clasificación binaria
julia benchmarks/run_benchmarks.jl multiclase   # Solo multiclase
julia benchmarks/run_benchmarks.jl papers        # Datasets de los papers

# Experimentos aplicados
julia experiments/run_experiments.jl             # Todos
julia experiments/run_experiments.jl medical     # Solo diagnóstico médico
julia experiments/run_experiments.jl industrial  # Solo control de calidad
```

## Parámetros recomendados

| Escenario | Cloud | Umbral | Épocas | Topología |
|---|---|---|---|---|
| Binario (≤10 features) | 50 | 0.5 | 1000 | [n_in, 2×n_in, 1] |
| Multiclase (3 clases) | 50 | 0.4 | 1000 | [n_in, 2×n_in, n_out] |
| Multiclase (5+ clases) | 50 | 0.25-0.3 | 1000 | [n_in, 2×n_in, n_out] |
| Multicapa | 50 | 0.3-0.4 | 1000 | [n_in, h1, h2, n_out] |
| División Neuronal | — | 0.4 | 500 | ≤15 inputs para exhaustivo |

## Limitaciones

- **Búsqueda exhaustiva**: la División Neuronal es exponencial en el número de inputs (2^n). Factible para ≤15 features; para más, solo se ejecuta la Fase Nube.
- **Redes feedforward**: no soporta arquitecturas recurrentes, convolucionales ni atención.
- **Sigmoid**: la evaluación con pesos aleatorios funciona porque sigmoid acota las salidas en (0,1). Otras activaciones pueden requerir ajustes.
- **Datos tabulares**: el sweet spot son datos con 4-15 features numéricas, 2-6 clases, y 100-1000 muestras.

## Papers

- Gil Blázquez, J. (2026). *Random Cloud: Finding Minimal Neural Architectures Without Training*. [paper](papers/paper-random-cloud.tex)
- Gil Blázquez, J. (2026). *Neural Division: Exhaustive Subnetwork Search for Efficient Classification and Problem Decomposition*. [paper](papers/paper-neural-division.tex)

## Licencia

MIT — Javier Gil Blázquez, 2026.
