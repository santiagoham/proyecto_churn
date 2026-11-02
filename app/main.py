from pathlib import Path
from uuid import uuid4
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


# Para localizar y cargar el modelo entrenado
RUTA_MODELO = Path(__file__).resolve().parent.parent / "models" / "churn_pipeline.joblib"
modelo = joblib.load(RUTA_MODELO)


# Para crear la aplicación
app = FastAPI(
    title="API de predicción de abandono de clientes",
    description="Servicio para predecir el abandono de clientes.",
    version="1.0.0"
)


# Para definir y validar los datos que debe recibir la API
class DatosCliente(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=72)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(gt=0)
    TotalCharges: float | None = Field(default=None, ge=0)


# Para definir y validar los datos que devuelve la API
class PredictionResponse(BaseModel):
    prediction: Literal[0, 1]
    probability: float = Field(ge=0, le=1)
    risk_level: Literal["Bajo", "Medio", "Alto"]
    model_version: str
    request_id: str


# Para comprobar que la API está funcionando
@app.get("/health")
def health():
    return {"status": "ok"}


# Para realizar una predicción
@app.post("/predict", response_model=PredictionResponse)
def predict(datos: DatosCliente):
    datos_cliente = pd.DataFrame([datos.model_dump()])
    
    prediccion = int(modelo.predict(datos_cliente)[0])
    
    clases = list(modelo.classes_)
    indice_positivo = clases.index(1)
    
    probabilidad = float(
        modelo.predict_proba(datos_cliente)[0][indice_positivo]
    )
    
    if probabilidad >= 0.7:
        nivel_riesgo = "Alto"
    elif probabilidad >= 0.4:
        nivel_riesgo = "Medio"
    else:
        nivel_riesgo = "Bajo"
    
    return {
        "prediction": prediccion,
        "probability": round(probabilidad, 4),
        "risk_level": nivel_riesgo,
        "model_version": "3",
        "request_id": str(uuid4())
    }