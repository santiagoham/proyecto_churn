import pytest
from fastapi.testclient import TestClient

from app.main import app


# Para iniciar la aplicación durante las pruebas
@pytest.fixture
def client():
    with TestClient(app) as cliente:
        yield cliente


# Para utilizar los mismos datos en las pruebas
@pytest.fixture
def datos_cliente():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 1026.0
    }


# Para comprobar que la API funciona correctamente
def test_health(client):
    respuesta = client.get("/health")
    
    assert respuesta.status_code == 200
    assert respuesta.json() == {"status": "ok"}


# Para comprobar que la API realiza una predicción correctamente
def test_predict(client, datos_cliente):
    respuesta = client.post("/predict", json=datos_cliente)
    
    assert respuesta.status_code == 200
    
    resultado = respuesta.json()
    
    assert resultado["prediction"] in ["CHURN", "NO_CHURN"]
    assert 0 <= resultado["probability"] <= 1
    assert resultado["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert resultado["model_version"] == "3"
    assert "request_id" in resultado
    assert len(resultado["request_id"]) > 0


# Para comprobar la conversión de las predicciones numéricas
@pytest.mark.parametrize(
    "prediccion, resultado_esperado",
    [
        (0, "NO_CHURN"),
        (1, "CHURN")
    ]
)
def test_predict_conversion_prediccion(
    client, monkeypatch, datos_cliente, prediccion, resultado_esperado
):
    from app.main import ml
    
    class ModeloPrueba:
        classes_ = [0, 1]
        
        def predict(self, datos):
            return [prediccion]
        
        def predict_proba(self, datos):
            return [[0.3, 0.7]]
    
    monkeypatch.setitem(ml, "model", ModeloPrueba())
    
    respuesta = client.post("/predict", json=datos_cliente)
    
    assert respuesta.status_code == 200
    assert respuesta.json()["prediction"] == resultado_esperado


# Para comprobar que la API rechaza datos con un formato incorrecto
def test_predict_datos_invalidos(client, datos_cliente):
    datos_cliente["tenure"] = "doce"
    
    respuesta = client.post("/predict", json=datos_cliente)
    
    assert respuesta.status_code == 422


# Para comprobar que la API rechaza categorías inexistentes
def test_predict_categoria_invalida(client, datos_cliente):
    datos_cliente["InternetService"] = "Fibra"
    
    respuesta = client.post("/predict", json=datos_cliente)
    
    assert respuesta.status_code == 422


# Para comprobar que la API rechaza valores fuera de rango
def test_predict_tenure_fuera_de_rango(client, datos_cliente):
    datos_cliente["tenure"] = 80
    
    respuesta = client.post("/predict", json=datos_cliente)
    
    assert respuesta.status_code == 422


# Para comprobar los niveles de riesgo según la probabilidad
@pytest.mark.parametrize(
    "probabilidad, riesgo_esperado",
    [
        (0.3999, "LOW"),
        (0.40, "MEDIUM"),
        (0.6999, "MEDIUM"),
        (0.70, "HIGH")
    ]
)
def test_predict_nivel_riesgo(
    client, monkeypatch, datos_cliente, probabilidad, riesgo_esperado
):
    from app.main import ml
    
    class ModeloPrueba:
        classes_ = [0, 1]
        
        def predict(self, datos):
            return [int(probabilidad >= 0.5)]
        
        def predict_proba(self, datos):
            return [[1 - probabilidad, probabilidad]]
    
    monkeypatch.setitem(ml, "model", ModeloPrueba())
    
    respuesta = client.post("/predict", json=datos_cliente)
    
    assert respuesta.status_code == 200
    assert respuesta.json()["risk_level"] == riesgo_esperado