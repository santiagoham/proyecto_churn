# Predicción de abandono de clientes

Proyecto de Machine Learning para predecir si los clientes de una empresa de telecomunicaciones pueden abandonar el servicio.

Para esto, se utilizan datos históricos de clientes, se realiza un análisis de los datos y se entrenan distintos modelos para comparar sus resultados y elegir el que mejor se adapte al problema.

Además, el proyecto cuenta con una API que permite ingresar los datos de un cliente y obtener una predicción sobre la posibilidad de que abandone el servicio.

## Estructura del proyecto

```text
proyecto_churn/
├── app/
│   ├── main.py
│   └── schemas.py
├── data/
│   └── raw/
├── models/
├── notebooks/
│   ├── analisis.ipynb
│   └── preparacion.ipynb
├── reports/
├── src/
│   └── training/
│       └── train.py
├── tests/
│   └── test_api.py
├── dvc.yaml
├── requirements.txt
└── README.md
```

- `app/`: contiene los archivos de la API y la definición de los datos que recibe y devuelve.
- `data/raw/`: contiene los datos originales utilizados para el entrenamiento, guardados con DVC.
- `models/`: contiene el modelo generado durante el entrenamiento.
- `notebooks/`: contiene los notebooks utilizados para analizar y preparar los datos.
- `reports/`: contiene las métricas obtenidas al evaluar los modelos.
- `src/training/`: contiene el código utilizado para entrenar y comparar los modelos.
- `tests/`: contiene las pruebas para comprobar que la API funciona correctamente.
- `dvc.yaml`: define los pasos del entrenamiento para poder volver a ejecutarlo con DVC.
- `requirements.txt`: contiene las librerías necesarias para ejecutar el proyecto.

## Instalación

Para ejecutar el proyecto es necesario tener instalado Python.

Se recomienda crear un entorno virtual e instalar las dependencias:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Datos

El dataset utilizado está guardado en DagsHub y se encuentra versionado con DVC.

Para descargarlo, ejecutar:

```powershell
dvc pull
```

El archivo utilizado para el entrenamiento se encuentra en:

```text
data/raw/customer_churn_historical.csv
```

## Entrenamiento

Para entrenar y comparar los modelos, ejecutar desde la carpeta principal del proyecto:

```powershell
python -m src.training.train
```

El script prepara los datos y entrena distintos modelos de clasificación:

- Baseline (DummyClassifier)
- Regresión Logística
- Random Forest

Después, muestra los resultados de cada modelo utilizando distintas métricas: Accuracy, Precision, Recall, F1 y ROC-AUC.

## Resultados

Los resultados obtenidos sobre el conjunto de prueba fueron:

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.736 | 0.000 | 0.000 | 0.000 | 0.500 |
| Regresión Logística | 0.794 | 0.663 | 0.449 | 0.535 | 0.812 |
| Random Forest | 0.783 | 0.642 | 0.401 | 0.493 | 0.790 |

Entre los modelos comparados, la Regresión Logística obtuvo los mejores resultados y fue elegida para continuar con el proyecto.

En este problema, el Recall es importante porque permite observar cuántos de los clientes que realmente abandonan el servicio logra detectar el modelo. En este caso, todavía hay clientes que el modelo no consigue identificar.

## Registro de experimentos

Se utiliza MLflow para guardar los resultados de los entrenamientos y llevar un registro de los modelos evaluados. También se utiliza su Model Registry para guardar el modelo seleccionado.

## API de predicción

Se desarrolló una API con FastAPI para poder utilizar el modelo y obtener predicciones a partir de los datos de un cliente.

La API cuenta con dos endpoints:

- `GET /health`: permite comprobar que la API está funcionando.
- `POST /predict`: recibe los datos de un cliente y devuelve la predicción, la probabilidad de abandono, el nivel de riesgo, la versión del modelo y un identificador de la solicitud.

Los datos que recibe y devuelve la API tienen validaciones para evitar errores en las solicitudes.

Para iniciar la aplicación, ejecutar desde la carpeta principal:

```powershell
uvicorn app.main:app --reload
```

La documentación de la API se puede consultar en:

```text
http://127.0.0.1:8000/docs
```

## Pruebas

Se agregaron pruebas automáticas para comprobar que la API funciona y que rechaza los datos incorrectos.

Para ejecutarlas, utilizar:

```powershell
python -m pytest -v
```

Las pruebas comprueban el funcionamiento del endpoint de estado, la realización de predicciones y el rechazo de datos con formatos, categorías o valores no permitidos.

## Reproducir el entrenamiento con DVC

También es posible volver a ejecutar el entrenamiento utilizando DVC:

```powershell
dvc repro
```

Este comando ejecuta los pasos definidos en `dvc.yaml` y genera el modelo en `models/churn_pipeline.joblib` y las métricas en `reports/metrics.json`.