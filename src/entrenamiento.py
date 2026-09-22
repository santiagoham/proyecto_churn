import pandas as pd
from sklearn.model_selection import train_test_split
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


df = pd.read_csv("data/customer_churn_historical.csv")

print("Cantidad de filas y columnas:")
print(df.shape)


# Para separar los datos de los clientes de la respuesta que queremos predecir

X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"].map({"Yes": 1, "No": 0})


# Para separar en 80% para entrenar y 20% para probar

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Para entrenar: {X_train.shape[0]}")
print(f"Para probar: {X_test.shape[0]}")


# Para separar las columnas numéricas de las categóricas

columnas_numericas = X_train.select_dtypes(include=["int64", "float64"]).columns
columnas_categoricas = X_train.select_dtypes(include=["object", "str"]).columns


# Para preparar las columnas numéricas

pipeline_numerico = Pipeline([
    ("imputacion", SimpleImputer(strategy="median")),
    ("escalado", StandardScaler())
])


# Para preparar las columnas categóricas

pipeline_categorico = Pipeline([
    ("imputacion", SimpleImputer(strategy="most_frequent")),
    ("codificacion", OneHotEncoder(handle_unknown="ignore"))
])


# Para unir la preparación de las columnas numéricas y categóricas

preprocesamiento = ColumnTransformer([
    ("numericas", pipeline_numerico, columnas_numericas),
    ("categoricas", pipeline_categorico, columnas_categoricas)
])


# Para crear los modelos junto con el preprocesamiento

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


# Para entrenar los tres modelos

modelo_baseline.fit(X_train, y_train)
modelo_logistico.fit(X_train, y_train)
modelo_random_forest.fit(X_train, y_train)

print("Modelos entrenados correctamente")


# Para generar las predicciones de cada modelo

y_pred_baseline = modelo_baseline.predict(X_test)
y_pred_logistico = modelo_logistico.predict(X_test)
y_pred_random_forest = modelo_random_forest.predict(X_test)

print("Predicciones realizadas correctamente")


# Para obtener las probabilidades de Churn de cada modelo

y_prob_baseline = modelo_baseline.predict_proba(X_test)[:, 1]
y_prob_logistico = modelo_logistico.predict_proba(X_test)[:, 1]
y_prob_random_forest = modelo_random_forest.predict_proba(X_test)[:, 1]


# Para comparar los resultados de los tres modelos

resultados = pd.DataFrame({
    "Modelo": ["Baseline", "Regresión Logística", "Random Forest"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_baseline),
        accuracy_score(y_test, y_pred_logistico),
        accuracy_score(y_test, y_pred_random_forest)
    ],
    "Precision": [
        precision_score(y_test, y_pred_baseline, zero_division=0),
        precision_score(y_test, y_pred_logistico, zero_division=0),
        precision_score(y_test, y_pred_random_forest, zero_division=0)
    ],
    "Recall": [
        recall_score(y_test, y_pred_baseline, zero_division=0),
        recall_score(y_test, y_pred_logistico, zero_division=0),
        recall_score(y_test, y_pred_random_forest, zero_division=0)
    ],
    "F1": [
        f1_score(y_test, y_pred_baseline, zero_division=0),
        f1_score(y_test, y_pred_logistico, zero_division=0),
        f1_score(y_test, y_pred_random_forest, zero_division=0)
    ],
    "ROC-AUC": [
        roc_auc_score(y_test, y_prob_baseline),
        roc_auc_score(y_test, y_prob_logistico),
        roc_auc_score(y_test, y_prob_random_forest)
    ]
})

print("\nComparación de modelos:")
print(resultados.round(3).to_string(index=False))