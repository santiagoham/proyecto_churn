import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import json
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Para cargar los datos que se van a utilizar
def cargar_datos(ruta):
    df = pd.read_csv(ruta)
    return df


# Para preparar y separar los datos en entrenamiento, validación y prueba
def separar_datos(df):
    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"].map({"Yes": 1, "No": 0})
    
    # Primero separamos el conjunto de prueba
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Después separamos entrenamiento y validación
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid, y_train_valid,
        test_size=0.25,
        random_state=42,
        stratify=y_train_valid
    )
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test


# Para crear el preprocesamiento de los datos
def crear_preprocesamiento(X_train):
    columnas_numericas = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns
    
    columnas_categoricas = X_train.select_dtypes(
        include=["object", "str"]
    ).columns
    
    pipeline_numerico = Pipeline([
        ("imputacion", SimpleImputer(strategy="median")),
        ("escalado", StandardScaler())
    ])
    
    pipeline_categorico = Pipeline([
        ("imputacion", SimpleImputer(strategy="most_frequent")),
        ("codificacion", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocesamiento = ColumnTransformer([
        ("numericas", pipeline_numerico, columnas_numericas),
        ("categoricas", pipeline_categorico, columnas_categoricas)
    ])
    
    return preprocesamiento


# Para crear los modelos junto con el preprocesamiento
def crear_modelos(preprocesamiento):
    modelo_baseline = Pipeline([
        ("preprocesamiento", preprocesamiento),
        ("modelo", DummyClassifier(strategy="most_frequent"))
    ])
    
    modelo_logistico = Pipeline([
        ("preprocesamiento", preprocesamiento),
        ("modelo", LogisticRegression(max_iter=1000))
    ])
    
    modelo_logistico_c05 = Pipeline([
        ("preprocesamiento", preprocesamiento),
        ("modelo", LogisticRegression(max_iter=1000, C=0.5))
    ])
    
    modelo_random_forest = Pipeline([
        ("preprocesamiento", preprocesamiento),
        ("modelo", RandomForestClassifier(random_state=42))
    ])
    
    modelo_random_forest_200 = Pipeline([
        ("preprocesamiento", preprocesamiento),
        ("modelo", RandomForestClassifier(
            random_state=42,
            n_estimators=200
        ))
    ])
    
    modelo_random_forest_profundidad = Pipeline([
        ("preprocesamiento", preprocesamiento),
        ("modelo", RandomForestClassifier(
            random_state=42,
            max_depth=8
        ))
    ])
    
    return (
        modelo_baseline,
        modelo_logistico,
        modelo_logistico_c05,
        modelo_random_forest,
        modelo_random_forest_200,
        modelo_random_forest_profundidad
    )


# Para evaluar un modelo
def evaluar_modelo(modelo, X, y):
    y_pred = modelo.predict(X)
    y_prob = modelo.predict_proba(X)[:, 1]
    
    metricas = {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y, y_prob)
    }
    
    matriz = confusion_matrix(y, y_pred)
    
    return metricas, matriz


# Para registrar los resultados de cada modelo en MLflow
def registrar_experimento(nombre, modelo, metricas, matriz):
    with mlflow.start_run(run_name=nombre):
        mlflow.log_params(modelo.named_steps["modelo"].get_params())
        
        mlflow.log_metrics({
            nombre_metrica.lower().replace("-", "_"): valor
            for nombre_metrica, valor in metricas.items()
        })
        
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=["No", "Yes"]
    ).plot(ax=ax)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)
        
        mlflow.sklearn.log_model(
            modelo,
            name="modelo",
            serialization_format="cloudpickle"
        )


# Para ejecutar el proceso completo de entrenamiento
def main():
    mlflow.set_experiment("churn-classification")
    df = cargar_datos("data/raw/customer_churn_historical.csv")
    
    print("Cantidad de filas y columnas:")
    print(df.shape)
    
    X_train, X_valid, X_test, y_train, y_valid, y_test = separar_datos(df)
    
    print(f"Para entrenar: {X_train.shape[0]}")
    print(f"Para validar: {X_valid.shape[0]}")
    print(f"Para probar: {X_test.shape[0]}")
    
    preprocesamiento = crear_preprocesamiento(X_train)
    
    (
    modelo_baseline,
    modelo_logistico,
    modelo_logistico_c05,
    modelo_random_forest,
    modelo_random_forest_200,
    modelo_random_forest_profundidad
    ) = crear_modelos(preprocesamiento)
    
    # Para entrenar los seis modelos
    modelo_baseline.fit(X_train, y_train)
    modelo_logistico.fit(X_train, y_train)
    modelo_logistico_c05.fit(X_train, y_train)
    modelo_random_forest.fit(X_train, y_train)
    modelo_random_forest_200.fit(X_train, y_train)
    modelo_random_forest_profundidad.fit(X_train, y_train)
    
    print("Modelos entrenados correctamente")
    
    # Para evaluar los seis modelos con los datos de validación
    metricas_baseline, matriz_baseline = evaluar_modelo(
    modelo_baseline, X_valid, y_valid
    )
    metricas_logistico, matriz_logistico = evaluar_modelo(
    modelo_logistico, X_valid, y_valid
    )
    metricas_logistico_c05, matriz_logistico_c05 = evaluar_modelo(
    modelo_logistico_c05, X_valid, y_valid
    )
    metricas_random_forest, matriz_random_forest = evaluar_modelo(
    modelo_random_forest, X_valid, y_valid
    )
    metricas_random_forest_200, matriz_random_forest_200 = evaluar_modelo(
    modelo_random_forest_200, X_valid, y_valid
    )
    metricas_random_forest_profundidad, matriz_random_forest_profundidad = evaluar_modelo(
    modelo_random_forest_profundidad, X_valid, y_valid
    )
    
    # Para registrar cada modelo en MLflow
    registrar_experimento(
    "Baseline", modelo_baseline, metricas_baseline, matriz_baseline
    )
    
    registrar_experimento(
    "Regresion Logistica", modelo_logistico, metricas_logistico,
    matriz_logistico
    )
    
    registrar_experimento(
    "Random Forest", modelo_random_forest, metricas_random_forest,
    matriz_random_forest
    )
    
    registrar_experimento(
    "Regresion Logistica C=0.5",
    modelo_logistico_c05,
    metricas_logistico_c05,
    matriz_logistico_c05
    )
    
    registrar_experimento(
    "Random Forest 200 arboles",
    modelo_random_forest_200,
    metricas_random_forest_200,
    matriz_random_forest_200
    )
    
    registrar_experimento(
    "Random Forest profundidad 8",
    modelo_random_forest_profundidad,
    metricas_random_forest_profundidad,
    matriz_random_forest_profundidad
    )
    
    print("Predicciones realizadas correctamente")
    
    # Para comparar los resultados de los tres modelos en validación
    resultados = pd.DataFrame([
    {"Modelo": "Baseline", **metricas_baseline},
    {"Modelo": "Regresión Logística", **metricas_logistico},
    {"Modelo": "Regresión Logística C=0.5", **metricas_logistico_c05},
    {"Modelo": "Random Forest", **metricas_random_forest},
    {"Modelo": "Random Forest 200 árboles", **metricas_random_forest_200},
    {"Modelo": "Random Forest profundidad 8", **metricas_random_forest_profundidad}
    ])
    
    print("\nComparación de modelos:")
    print(resultados.round(3).to_string(index=False))
    
    # Para entrenar el modelo seleccionado con los datos de entrenamiento y validación
    X_train_final = pd.concat([X_train, X_valid])
    y_train_final = pd.concat([y_train, y_valid])
    
    modelo_logistico.fit(X_train_final, y_train_final)
    
    # Para evaluar el modelo seleccionado con los datos de prueba
    metricas_finales, matriz_final = evaluar_modelo(
    modelo_logistico, X_test, y_test
    )
    
    # Para registrar el modelo final en MLflow
    with mlflow.start_run(run_name="Modelo final - Regresion Logistica"):
        mlflow.log_params(modelo_logistico.named_steps["modelo"].get_params())
        
        mlflow.log_metrics({
            nombre_metrica.lower().replace("-", "_"): valor
            for nombre_metrica, valor in metricas_finales.items()
        })
        
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(
            confusion_matrix=matriz_final,
            display_labels=["No", "Yes"]
        ).plot(ax=ax)
        
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)
        
        mlflow.sklearn.log_model(
            modelo_logistico,
            name="modelo",
            serialization_format="cloudpickle",
            registered_model_name="customer-churn-classifier"
        )
    
    print("\nResultados finales en el conjunto de prueba:")
    print(pd.DataFrame([metricas_finales]).round(3).to_string(index=False))
    
    # Para guardar el modelo final con su preprocesamiento
    joblib.dump(modelo_logistico, "models/churn_pipeline.joblib")
    
    print("\nModelo guardado correctamente")
    
    # Para guardar las métricas de validación y prueba
    metricas_guardadas = {
    "validacion": resultados.set_index("Modelo").to_dict(orient="index"),
    "prueba": metricas_finales
    }
    
    with open("reports/metrics.json", "w") as archivo:
        json.dump(
        metricas_guardadas,
        archivo,
        indent=4
        )
    
    print("Métricas guardadas correctamente")


if __name__ == "__main__":
    main()