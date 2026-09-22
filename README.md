# Predicción de abandono de clientes

Proyecto de Machine Learning para predecir el abandono de clientes de una empresa de telecomunicaciones.

A partir de datos históricos de clientes, se realiza un análisis exploratorio, preparación de los datos y entrenamiento de distintos modelos de clasificación para estimar si un cliente abandonará o no el servicio.

En esta primera etapa se comparan un modelo baseline, Regresión Logística y Random Forest utilizando distintas métricas de evaluación.

## Estructura del proyecto

```text
proyecto_churn/
├── data/
│   └── customer_churn_historical.csv.dvc
├── notebooks/
│   ├── analisis.ipynb
│   └── preparacion.ipynb
├── src/
│   └── entrenamiento.py
├── .dvcignore
├── .gitignore
├── requirements.txt
└── README.md
```

- `data/`: contiene los datos utilizados por el proyecto, versionados con DVC.
- `notebooks/`: contiene el análisis exploratorio y las pruebas de preparación y modelado.
- `src/`: contiene el código Python para ejecutar el entrenamiento y la evaluación de los modelos.
- `requirements.txt`: contiene las dependencias necesarias para ejecutar el proyecto.

## Instalación

Para ejecutar el proyecto es necesario tener instalado Python.

Se recomienda crear un entorno virtual e instalar las dependencias del proyecto:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Datos

El dataset utilizado para el entrenamiento está versionado con DVC y almacenado en DagsHub.

Para descargar los datos del proyecto:

```powershell
dvc pull
```

El archivo utilizado para el entrenamiento se encuentra en:

```text
data/customer_churn_historical.csv
```

## Entrenamiento

Para entrenar y comparar los modelos, ejecutar desde la raíz del proyecto:

```powershell
python src/entrenamiento.py
```

El script realiza la división de los datos en entrenamiento y prueba, aplica el preprocesamiento mediante un Pipeline de scikit-learn y entrena tres modelos:

- Baseline (DummyClassifier)
- Regresión Logística
- Random Forest

Finalmente, muestra una comparación utilizando Accuracy, Precision, Recall, F1 y ROC-AUC.

## Resultados

Los resultados obtenidos sobre el conjunto de prueba fueron:

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.736 | 0.000 | 0.000 | 0.000 | 0.500 |
| Regresión Logística | 0.794 | 0.663 | 0.449 | 0.535 | 0.812 |
| Random Forest | 0.783 | 0.642 | 0.401 | 0.493 | 0.790 |

En esta primera comparación, la Regresión Logística obtuvo los mejores resultados entre los modelos evaluados, con un ROC-AUC de 0.812 y un Recall de 0.449 para la clase de abandono.

El Recall es una métrica importante en este problema, ya que un falso negativo representa un cliente que realmente abandonará el servicio pero que el modelo no logra identificar.