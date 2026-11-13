# 📈 SolFX-Predictor
**Predicción del tipo de cambio USD/PEN mediante Machine Learning**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Ensembles-orange.svg)
![Status](https://img.shields.io/badge/Estado-En%20Desarrollo-green.svg)

SolFX-Predictor es un sistema de aprendizaje supervisado diseñado para pronosticar la cotización del dólar estadounidense en soles peruanos (USD/PEN) en múltiples horizontes temporales (1, 5 y 20 días hábiles).

---

## 🎯 Resumen y Propuesta de Valor
Las fluctuaciones actuales del tipo de cambio monetario dificultan la planificación financiera y la gestión del riesgo cambiario en el Perú. 
Para resolver esto, el proyecto propone una arquitectura analítica que combina:
*   **Fundamentos Económicos e Indicadores Técnicos Causales:** Rezagos autorregresivos, distancias a medias móviles, RSI, volatilidad histórica GARCH, MACD y codificación cíclica de calendario, preparados bajo control estricto *anti-leakage* (`shift(1)`).
*   **Machine Learning Avanzado:** Evaluación temporal expansiva (*walk-forward* con purga `gap=h`) comparando cuatro algoritmos de ensamble del estado del arte (**Random Forest, XGBoost, LightGBM y CatBoost**) frente a la caminata aleatoria.
*   **Interpretabilidad y Trazabilidad:** Análisis de importancia de variables por horizonte y artefactos reproducibles listos para auditoría académica.

## 📂 Arquitectura del Repositorio

```text
SolFX-Predictor/
├── data/                        # Datasets: Crudos (Bloomberg, BCRPData) y Procesados (Train/Holdout)
├── docs/                        # Documentación de la investigación desglosada por capítulos (01 a 04)
├── models/                      # Artefactos de modelos entrenados (XGBoost, LightGBM, RF, CatBoost)
├── notebooks/                   # Jupyter Notebooks (EDA, Preprocesamiento y Comparación de Ensambles)
├── resultados_xgboost/          # Métricas, predicciones, curvas e importancias de XGBoost
├── resultados_lightgbm/         # Métricas, predicciones, curvas e importancias de LightGBM
├── resultados_random_forest/    # Métricas, predicciones, curvas e importancias de Random Forest
├── resultados_catboost/         # Métricas, predicciones, curvas e importancias de CatBoost
├── resultados_comparativos/     # Tabla y gráficos consolidados de los 4 modelos frente al baseline
├── src/                         # Módulos fuente (bcrp_loader.py, preprocessing.py)
├── Entrenando_el_XGBoost.py     # Experimento reproducible de XGBoost (h = 1, 5, 20)
├── Entrenando_el_LightGBM.py    # Experimento reproducible de LightGBM (h = 1, 5, 20)
├── Entrenando_el_RandomForest.py# Experimento reproducible de Random Forest (h = 1, 5, 20)
├── Entrenando_el_CatBoost.py    # Experimento reproducible de CatBoost (h = 1, 5, 20)
├── Comparar_Modelos_Ensamble.py # Orquestador comparativo y sincronizador de modelos
├── requirements.txt             # Dependencias del proyecto
└── README.md                    # Panel principal de navegación
```

## 🚀 Ejecución Rápida de los Modelos

Desde la raíz del repositorio:

```powershell
python -m pip install -r requirements.txt
python Entrenando_el_XGBoost.py
python Entrenando_el_LightGBM.py
python Entrenando_el_RandomForest.py
python Entrenando_el_CatBoost.py
python Comparar_Modelos_Ensamble.py
```

## 📊 Resumen de Resultados en Holdout (60 sesiones: 10/07/2026 – 01/10/2026)

| Horizonte ($h$) | Modelo | MAE Nivel (PEN/USD) | MAE Caminata Aleatoria | Theil's U (RMSE) | Acierto Direccional |
|:---:|---|---:|---:|---:|---:|
| **1 día** | **Random Forest** | **0.00692** | 0.00688 | 0.9929 | **50.0 %** |
| **1 día** | **CatBoost** | 0.00696 | 0.00688 | 0.9976 | 46.3 % |
| **1 día** | **XGBoost** | 0.00697 | 0.00688 | 0.9936 | 42.6 % |
| **1 día** | **LightGBM** | 0.00697 | 0.00688 | **0.9861** | 46.3 % |
| **5 días** | **Random Forest** | **0.01758** | 0.01746 | **0.9956** | 48.1 % |
| **5 días** | **XGBoost** | 0.01760 | 0.01746 | 0.9970 | **50.0 %** |
| **5 días** | **LightGBM** | 0.01764 | 0.01746 | 0.9997 | 40.7 % |
| **5 días** | **CatBoost** | 0.01776 | 0.01746 | 0.9987 | **50.0 %** |
| **20 días** | **Random Forest** | **0.03896** | 0.03591 | **1.0678** | **12.5 %** |
| **20 días** | **LightGBM** | 0.03904 | 0.03591 | 1.0694 | 2.5 % |
| **20 días** | **XGBoost** | 0.03912 | 0.03591 | 1.0751 | 5.0 % |
| **20 días** | **CatBoost** | 0.04145 | 0.03591 | 1.1331 | 10.0 % |

El análisis detallado se encuentra en [`docs/04_Resultados_Ensambles.md`](docs/04_Resultados_Ensambles.md).

## 🛠️ Enfoque Metodológico
1.  **Ingeniería de Características:** Creación de variables técnicas (rezagos, distancias a medias móviles, volatilidad histórica GARCH, RSI, MACD e indicadores cíclicos de calendario).
2.  **Validación Temporal con Purga:** `TimeSeriesSplit(n_splits=3, test_size=252, gap=h)` sobre el conjunto de entrenamiento (8 462 días hábiles) para evitar fuga de información entre etiquetas superpuestas.
3.  **Evaluación de Origen Móvil:** Reconstrucción del tipo de cambio proyectado $\hat{P}_{t+h} = P_t \exp(\hat{y}_{t,h}/10\,000)$ sobre el bloque reservado de 60 sesiones.

## 👥 Equipo de Investigación (Grupo 3)
*   **Fredy Alberto Polar Urviola** (20221300)
*   **Renzo Raúl Hipólito Medina** (20190213)
*   **Ignacio Alonso Colán Lamas** (20221193)
*   **Dany Denilson Medina Quinto** (20192730)

---
*Desarrollado como proyecto aplicativo en ciencia de datos e investigación macroeconómica.*
