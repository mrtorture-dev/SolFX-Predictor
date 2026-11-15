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
*   **Interpretabilidad y Trazabilidad:** Análisis de importancia de variables por horizonte, diccionario descriptivo de variables y métricas, y artefactos reproducibles listos para auditoría académica.

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
├── resultados_comparativos/     # Tablas descriptivas, valores exactos de barras y gráficos multi-modelo
├── src/                         # Módulos fuente (bcrp_loader.py, preprocessing.py)
├── Entrenando_el_XGBoost.py     # Experimento reproducible de XGBoost (h = 1, 5, 20)
├── Entrenando_el_LightGBM.py    # Experimento reproducible de LightGBM (h = 1, 5, 20)
├── Entrenando_el_RandomForest.py# Experimento reproducible de Random Forest (h = 1, 5, 20)
├── Entrenando_el_CatBoost.py    # Experimento reproducible de CatBoost (h = 1, 5, 20)
├── Comparar_Modelos_Ensamble.py # Orquestador comparativo y generador de tablas descriptivas
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

---

## 📊 Resultados Comparativos en el Periodo de Prueba (*Holdout*: 60 sesiones, 10/07/2026 – 01/10/2026)

### 1. Tabla Comparativa General con Encabezados Descriptivos

| Horizonte ($h$) | Algoritmo de Ensamble | Error en Validación Cruzada (MAE en pb) | Error Medio del Modelo en Nivel (MAE, Soles/USD) | Error Medio Caminata Aleatoria (MAE Base, Soles/USD) | Error Cuadrático en Nivel (RMSE, Soles/USD) | Ratio U de Theil (< 1 mejora RW) | Acierto en Dirección de Subida/Bajada (% y Días) | Ajuste sobre Retornos ($R^2$) |
|:---:|---|---:|---:|---:|---:|---:|---:|---:|
| **1 día hábil** | **Random Forest** | 27.23 pb | **0.006925 S/** | 0.006881 S/ | 0.009343 S/ | 0.9929 | **50.00 % (27/54)** | 0.0030 |
| **1 día hábil** | **CatBoost** | **27.18 pb** | 0.006955 S/ | 0.006881 S/ | 0.009388 S/ | 0.9976 | 46.30 % (25/54) | -0.0065 |
| **1 día hábil** | **XGBoost** | 27.26 pb | 0.006971 S/ | 0.006881 S/ | 0.009349 S/ | 0.9936 | 42.59 % (23/54) | 0.0017 |
| **1 día hábil** | **LightGBM** | 27.35 pb | 0.006974 S/ | 0.006881 S/ | **0.009277 S/** | **0.9861** | 46.30 % (25/54) | **0.0166** |
| **5 días hábiles** | **Random Forest** | **65.59 pb** | **0.017576 S/** | 0.017458 S/ | **0.024326 S/** | **0.9956** | 48.15 % (26/54) | **-0.0150** |
| **5 días hábiles** | **XGBoost** | 66.47 pb | 0.017602 S/ | 0.017458 S/ | 0.024360 S/ | 0.9970 | **50.00 % (27/54)** | -0.0180 |
| **5 días hábiles** | **LightGBM** | 66.62 pb | 0.017640 S/ | 0.017458 S/ | 0.024425 S/ | 0.9997 | 40.74 % (22/54) | -0.0234 |
| **5 días hábiles** | **CatBoost** | 66.03 pb | 0.017758 S/ | 0.017458 S/ | 0.024400 S/ | 0.9987 | **50.00 % (27/54)** | -0.0212 |
| **20 días hábiles** | **Random Forest** | **132.40 pb** | **0.038959 S/** | 0.035905 S/ | **0.045494 S/** | **1.0678** | **12.50 % (5/40)** | **-0.1636** |
| **20 días hábiles** | **LightGBM** | 133.82 pb | 0.039042 S/ | 0.035905 S/ | 0.045570 S/ | 1.0694 | 2.50 % (1/40) | -0.1670 |
| **20 días hábiles** | **XGBoost** | 134.00 pb | 0.039115 S/ | 0.035905 S/ | 0.045808 S/ | 1.0751 | 5.00 % (2/40) | -0.1794 |
| **20 días hábiles** | **CatBoost** | 133.93 pb | 0.041446 S/ | 0.035905 S/ | 0.048261 S/ | 1.1331 | 10.00 % (4/40) | -0.3100 |

---

### 2. Tablas de Valores Exactos del Gráfico de Barras (`resumen_mae_y_acierto_direccional.png`)

Para complementar visualmente el gráfico [`resumen_mae_y_acierto_direccional.png`](resultados_comparativos/resumen_mae_y_acierto_direccional.png) (donde las barras de MAE dentro de un mismo horizonte tienen alturas muy cercanas), se muestran los valores exactos de cada barra (disponibles también en [`tabla_valores_exactos_grafico_barras.csv`](resultados_comparativos/tabla_valores_exactos_grafico_barras.csv)):

#### Panel A del Gráfico de Barras — Error Absoluto Medio Exacto (MAE en Soles/USD y en Puntos Básicos)

| Horizonte ($h$) | Pronósticos ($N$) | Caminata Aleatoria (Referencia) | Random Forest | CatBoost | XGBoost | LightGBM |
|:---:|---:|---:|---:|---:|---:|---:|
| **1 día hábil** *(Soles/USD)* | 59 | **0.006881 S/** | **0.006925 S/** | 0.006955 S/ | 0.006971 S/ | 0.006974 S/ |
| **1 día hábil** *(Retorno pb)* | 59 | **20.34 pb** | **20.47 pb** | 20.56 pb | 20.61 pb | 20.62 pb |
| **5 días hábiles** *(Soles/USD)* | 55 | **0.017458 S/** | **0.017576 S/** | 0.017758 S/ | 0.017602 S/ | 0.017640 S/ |
| **5 días hábiles** *(Retorno pb)* | 55 | **51.63 pb** | **51.98 pb** | 52.51 pb | 52.05 pb | 52.17 pb |
| **20 días hábiles** *(Soles/USD)* | 40 | **0.035905 S/** | **0.038959 S/** | 0.041446 S/ | 0.039115 S/ | 0.039042 S/ |
| **20 días hábiles** *(Retorno pb)* | 40 | **106.20 pb** | **115.26 pb** | 122.64 pb | 115.71 pb | 115.48 pb |

#### Panel B del Gráfico de Barras — Acierto Direccional Exacto (% y Sesiones Acertadas / Movimientos No Nulos)

| Horizonte ($h$) | Movimientos No Nulos | Referencia Azar | Random Forest | CatBoost | XGBoost | LightGBM |
|:---:|---:|---:|---:|---:|---:|---:|
| **1 día hábil** | 54 sesiones | 50.00 % | **50.00 %** *(27/54)* | 46.30 % *(25/54)* | 42.59 % *(23/54)* | 46.30 % *(25/54)* |
| **5 días hábiles** | 54 sesiones | 50.00 % | 48.15 % *(26/54)* | **50.00 %** *(27/54)* | **50.00 %** *(27/54)* | 40.74 % *(22/54)* |
| **20 días hábiles** | 40 sesiones | 50.00 % | **12.50 %** *(5/40)* | 10.00 % *(4/40)* | 5.00 % *(2/40)* | 2.50 % *(1/40)* |

---

### 3. Guía Rápida de Variables y Métricas

* **Diccionario de las 18 Variables Predictoras (`COLUMNAS_X`):**
  * `PEN_log_ret_lag_{1,2,3,5,10,21}`: Retornos logarítmicos diarios rezagados $1, 2, 3, 5, 10$ y $21$ sesiones hábiles ($\ln(P_{t-k}/P_{t-k-1})$).
  * `price_vs_SMA_{5,10,20}`: Distancia relativa porcentual del último cierre $P_{t-1}$ respecto a su media móvil simple de $5, 10$ y $20$ sesiones.
  * `RSI`: Índice de Fuerza Relativa de 14 sesiones escalado al intervalo $[-1, 1]$ (sobrecompra $>0$, sobreventa $<0$).
  * `GARCH_vol`: Volatilidad histórica móvil de 21 días hábiles calculada sobre retornos rezagados $r_{t-1}$.
  * `MACD_line`, `MACD_signal`, `MACD_histogram`: Indicador de convergencia/divergencia de medias móviles exponenciales (12 y 26 días, señal de 9 días) normalizado por el precio $P_{t-1}$.
  * `month_sin`, `month_cos`, `day_of_week_sin`, `day_of_week_cos`: Transformaciones trigonométricas cíclicas del mes del año y del día de la semana.
* **Diccionario de Métricas de Evaluación:**
  * **MAE en Nivel (Soles/USD):** Error absoluto medio en soles por dólar ($\frac{1}{N}\sum |P_{t+h} - \hat{P}_{t+h}|$).
  * **MAE en Retornos (pb):** Error absoluto medio en puntos básicos de retorno ($1\text{ pb} = 0.01\%$).
  * **RMSE en Nivel (Soles/USD):** Raíz del error cuadrático medio en soles por dólar; castiga más los errores extremos.
  * **Ratio U de Theil:** Cociente $\text{RMSE}_{\text{Modelo}} / \text{RMSE}_{\text{Caminata Aleatoria}}$. Un valor $< 1.0$ indica menor error cuadrático que no predecir cambios.
  * **Acierto Direccional (%):** Porcentaje de sesiones con variación real distinta de cero en que el modelo predijo correctamente si el dólar subiría o bajaría.

El detalle completo y diccionario exhaustivo se encuentran en [`docs/04_Resultados_Ensambles.md`](docs/04_Resultados_Ensambles.md) y en [`resultados_comparativos/diccionario_variables_y_metricas.csv`](resultados_comparativos/diccionario_variables_y_metricas.csv).

## 👥 Equipo de Investigación (Grupo 3)
*   **Fredy Alberto Polar Urviola** (20221300)
*   **Renzo Raúl Hipólito Medina** (20190213)
*   **Ignacio Alonso Colán Lamas** (20221193)
*   **Dany Denilson Medina Quinto** (20192730)

---
*Desarrollado como proyecto aplicativo en ciencia de datos e investigación macroeconómica.*
