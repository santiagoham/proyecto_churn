# Predicción de abandono de clientes

Este proyecto busca predecir qué clientes de una empresa de telecomunicaciones podrían abandonar el servicio.

Para eso, se utilizan datos históricos de clientes, se analizan sus características y se entrenan distintos modelos de Machine Learning para comparar sus resultados y elegir el más adecuado.

Además, se desarrolló una API que permite enviar los datos de un cliente y obtener una predicción sobre la posibilidad de abandono.

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

- `app/`: contiene los archivos de la API.
- `data/raw/`: contiene el dataset original, versionado con DVC.
- `models/`: guarda el modelo entrenado.
- `notebooks/`: contiene los análisis y la preparación de los datos.
- `reports/`: guarda las métricas de los modelos.
- `src/training/`: contiene el código para entrenar y comparar los modelos.
- `tests/`: contiene las pruebas de la API.

## Instalación

Para ejecutar el proyecto, es necesario tener Python instalado.

Primero, crear y activar un entorno virtual:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Después, instalar las dependencias:

```powershell
pip install -r requirements.txt
```

## Datos

El dataset utilizado se encuentra guardado en DagsHub y está versionado con DVC.

Para descargarlo, ejecutar:

```powershell
dvc pull
```

El archivo utilizado para el entrenamiento es:

```text
data/raw/customer_churn_historical.csv
```

## Entrenamiento

Para entrenar y comparar los modelos, ejecutar desde la carpeta principal:

```powershell
python -m src.training.train
```

Se comparan tres tipos de modelos:

- Baseline (`DummyClassifier`).
- Regresión Logística.
- Random Forest.

Para evaluar sus resultados se utilizan Accuracy, Precision, Recall, F1 y ROC-AUC.

## Resultados

Los siguientes resultados corresponden al conjunto de prueba:

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.736 | 0.000 | 0.000 | 0.000 | 0.500 |
| Regresión Logística | 0.794 | 0.663 | 0.449 | 0.535 | 0.812 |
| Random Forest | 0.783 | 0.642 | 0.401 | 0.493 | 0.790 |

La Regresión Logística obtuvo los mejores resultados generales y fue seleccionada para continuar con el proyecto.

El Recall permite observar cuántos de los clientes que realmente abandonan el servicio logra detectar el modelo. En este caso, todavía hay clientes que no consigue identificar, por lo que los resultados tienen margen de mejora.

## Registro de experimentos

Se utiliza MLflow para registrar los entrenamientos y comparar los resultados de los modelos.

También se utiliza MLflow Model Registry para guardar y consultar las versiones del modelo seleccionado.

## API de predicción

Se desarrolló una API con FastAPI para obtener predicciones utilizando el modelo seleccionado.

La API cuenta con dos endpoints:

- `GET /health`: permite comprobar que la API está funcionando.
- `POST /predict`: recibe los datos de un cliente y devuelve una predicción.

La respuesta incluye:

- `prediction`: resultado de la predicción (`CHURN` o `NO_CHURN`).
- `probability`: probabilidad estimada de abandono.
- `risk_level`: nivel de riesgo (`LOW`, `MEDIUM` o `HIGH`).
- `model_version`: versión del modelo utilizado.
- `request_id`: identificador único de la solicitud.

### Niveles de riesgo

La API utiliza los siguientes valores para clasificar el riesgo:

- **LOW:** probabilidad menor a 0.40.
- **MEDIUM:** probabilidad desde 0.40 hasta menos de 0.70.
- **HIGH:** probabilidad igual o superior a 0.70.

Estos valores se utilizan como criterios de segmentación y no como umbrales optimizados estadísticamente.

### Ejemplo de respuesta

```json
{
    "prediction": "CHURN",
    "probability": 0.8319,
    "risk_level": "HIGH",
    "model_version": "3",
    "request_id": "15ea4227-13d5-4768-ba61-7979cfeea0a9"
}
```

### Ejecutar la API

Desde la carpeta principal del proyecto, ejecutar:

```powershell
uvicorn app.main:app --reload
```

La documentación de la API está disponible en:

```text
http://127.0.0.1:8000/docs
```

## Pruebas

Se utilizan pruebas automáticas con pytest para comprobar el funcionamiento de la API.

Para ejecutarlas:

```powershell
python -m pytest -v
```

Las 11 pruebas actuales comprueban el endpoint de estado, las predicciones, la validación de los datos recibidos y la asignación de los niveles de riesgo.

## Reproducir el entrenamiento con DVC

Para volver a ejecutar el entrenamiento definido en `dvc.yaml`, utilizar:

```powershell
dvc repro
```

Este comando ejecuta el proceso y genera el modelo en `models/churn_pipeline.joblib` y las métricas en `reports/metrics.json`.