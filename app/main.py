from pathlib import Path
from uuid import uuid4
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI

from app.schemas import DatosCliente, PredictionResponse


# Para localizar el modelo entrenado
RUTA_MODELO = Path(__file__).resolve().parent.parent / "models" / "churn_pipeline.joblib"

# Para almacenar el modelo cargado
ml = {}


# Para cargar el modelo al iniciar la aplicación
@asynccontextmanager
async def lifespan(app: FastAPI):
    ml["model"] = joblib.load(RUTA_MODELO)
    yield
    ml.clear()


# Para crear la aplicación
app = FastAPI(
    title="API de predicción de abandono de clientes",
    description="Servicio para predecir el abandono de clientes.",
    version="1.0.0",
    lifespan=lifespan
)


# Para comprobar que la API está funcionando
@app.get("/health")
def health():
    return {"status": "ok"}


# Para realizar una predicción
@app.post("/predict", response_model=PredictionResponse)
def predict(datos: DatosCliente):
    modelo = ml["model"]
    
    datos_cliente = pd.DataFrame([datos.model_dump()])
    
    prediccion = int(modelo.predict(datos_cliente)[0])
    
    if prediccion == 1:
        resultado = "CHURN"
    else:
        resultado = "NO_CHURN"
    
    clases = list(modelo.classes_)
    indice_positivo = clases.index(1)
    
    probabilidad = float(
        modelo.predict_proba(datos_cliente)[0][indice_positivo]
    )
    
    if probabilidad >= 0.7:
        nivel_riesgo = "HIGH"
    elif probabilidad >= 0.4:
        nivel_riesgo = "MEDIUM"
    else:
        nivel_riesgo = "LOW"
    
    return {
        "prediction": resultado,
        "probability": round(probabilidad, 4),
        "risk_level": nivel_riesgo,
        "model_version": "3",
        "request_id": str(uuid4())
    }