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
    roc_auc_score
)


# Para cargar los datos que se van a utilizar
def cargar_datos(ruta):
    df = pd.read_csv(ruta)
    return df


# Para preparar y separar los datos en entrenamiento y prueba
def separar_datos(df):
    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"].map({"Yes": 1, "No": 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


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

    modelo_random_forest = Pipeline([
        ("preprocesamiento", preprocesamiento),
        ("modelo", RandomForestClassifier(random_state=42))
    ])

    return modelo_baseline, modelo_logistico, modelo_random_forest


# Para evaluar un modelo
def evaluar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }
    
    
    # Para ejecutar el proceso completo de entrenamiento
def main():
    df = cargar_datos("data/raw/customer_churn_historical.csv")

    print("Cantidad de filas y columnas:")
    print(df.shape)

    X_train, X_test, y_train, y_test = separar_datos(df)

    print(f"Para entrenar: {X_train.shape[0]}")
    print(f"Para probar: {X_test.shape[0]}")

    preprocesamiento = crear_preprocesamiento(X_train)

    modelo_baseline, modelo_logistico, modelo_random_forest = crear_modelos(
        preprocesamiento
    )

    # Para entrenar los tres modelos
    modelo_baseline.fit(X_train, y_train)
    modelo_logistico.fit(X_train, y_train)
    modelo_random_forest.fit(X_train, y_train)

    print("Modelos entrenados correctamente")

    # Para evaluar los tres modelos
    metricas_baseline = evaluar_modelo(modelo_baseline, X_test, y_test)
    metricas_logistico = evaluar_modelo(modelo_logistico, X_test, y_test)
    metricas_random_forest = evaluar_modelo(
        modelo_random_forest, X_test, y_test
    )

    print("Predicciones realizadas correctamente")

    # Para comparar los resultados de los tres modelos
    resultados = pd.DataFrame([
        {"Modelo": "Baseline", **metricas_baseline},
        {"Modelo": "Regresión Logística", **metricas_logistico},
        {"Modelo": "Random Forest", **metricas_random_forest}
    ])

    print("\nComparación de modelos:")
    print(resultados.round(3).to_string(index=False))

    # Para guardar el modelo de Regresión Logística junto con su preprocesamiento
    joblib.dump(modelo_logistico, "models/churn_pipeline.joblib")

    print("\nModelo guardado correctamente")

    # Para guardar las métricas de los modelos
    with open("reports/metrics.json", "w") as archivo:
        json.dump(
            resultados.set_index("Modelo").to_dict(orient="index"),
            archivo,
            indent=4
        )

    print("Métricas guardadas correctamente")


if __name__ == "__main__":
    main()