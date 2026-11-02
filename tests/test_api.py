from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# Para comprobar que la API funciona correctamente
def test_health():
    respuesta = client.get("/health")
    
    assert respuesta.status_code == 200
    assert respuesta.json() == {"status": "ok"}


# Para comprobar que la API realiza una predicción correctamente
def test_predict():
    datos_cliente = {
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
    
    respuesta = client.post("/predict", json=datos_cliente)
    
    assert respuesta.status_code == 200
    
    resultado = respuesta.json()
    
    assert resultado["prediction"] in [0, 1]
    assert 0 <= resultado["probability"] <= 1
    assert resultado["risk_level"] in ["Bajo", "Medio", "Alto"]
    assert resultado["model_version"] == "3"
    
    assert "request_id" in resultado
    assert len(resultado["request_id"]) > 0


# Para comprobar que la API rechaza datos con un formato incorrecto
def test_predict_datos_invalidos():
    datos_cliente = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": "doce",
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
    
    respuesta = client.post("/predict", json=datos_cliente)
    
    assert respuesta.status_code == 422


# Para comprobar que la API rechaza categorías inexistentes
def test_predict_categoria_invalida():
    datos_cliente = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fibra",
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
    
    respuesta = client.post("/predict", json=datos_cliente)
    
    assert respuesta.status_code == 422


# Para comprobar que la API rechaza valores fuera de rango
def test_predict_tenure_fuera_de_rango():
    datos_cliente = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 80,
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
    
    respuesta = client.post("/predict", json=datos_cliente)
    
    assert respuesta.status_code == 422