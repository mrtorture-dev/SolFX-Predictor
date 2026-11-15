# 📈 SolFX-Predictor
**Predicción del tipo de cambio USD/PEN mediante Machine Learning**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Ensembles-orange.svg)
![Status](https://img.shields.io/badge/Estado-Completado-brightgreen.svg)

SolFX-Predictor es un sistema de aprendizaje supervisado diseñado para pronosticar la cotización del dólar estadounidense en soles peruanos (USD/PEN) en múltiples horizontes temporales (1, 5 y 20 días hábiles), comparando cuatro arquitecturas de ensamble (**Random Forest, XGBoost, LightGBM y CatBoost**) en dos fases experimentales:
* **Fase 1 (Línea Base Endógena — 18 Variables):** Indicadores técnicos y cíclicos de calendario derivados únicamente del USD/PEN.
* **Fase 2 (Ensambles Multivariables con Exógenas Globales y Commodities — 50 Variables):** Incorpora **Cobre, Índice de Volatilidad VIX, Índice Dólar (DXY), Oro, Petróleo WTI, Bono del Tesoro EE. UU. a 10 años (`US10Y`), S&P 500, Términos de Intercambio (Cobre/WTI) e Inflación BCRP** bajo control causal estricto `shift(1)`.

---

## 📂 Arquitectura del Repositorio

```text
SolFX-Predictor/
├── data/                                          # Datasets Crudos y Procesados (Train: 8462d, Holdout: 60d)
│   ├── raw/exogenas_globales_1994_2026.csv        # Series diarias de Cobre, VIX, DXY, Oro, WTI, US10Y y S&P 500
│   └── processed/usdpen_{train,holdout}_exog.csv  # Datasets enriquecidos con las 50 variables causales
├── docs/                                          # Documentación académica por capítulos (01 a 04)
├── models/                                        # 24 Modelos entrenados (12 Fase 1 + 12 Fase 2 con Exógenas)
├── notebooks/                                     # Jupyter Notebooks (EDA, Preprocesamiento y Comparación)
├── resultados_xgboost/                            # Resultados Fase 1 de XGBoost
├── resultados_lightgbm/                           # Resultados Fase 1 de LightGBM
├── resultados_random_forest/                      # Resultados Fase 1 de Random Forest
├── resultados_catboost/                           # Resultados Fase 1 de CatBoost
├── resultados_ensambles_exogenas/                 # Resultados Fase 2 (50 vars con Cobre, VIX, DXY, Oro, WTI, US10Y)
├── resultados_comparativos/                       # Tablas antes/después, valores exactos de barras y gráficos
├── src/                                           # Módulos fuente (bcrp_loader.py, preprocessing.py, exogenous_loader.py)
├── Entrenando_el_XGBoost.py                       # Script Fase 1: XGBoost (18 vars técnicas)
├── Entrenando_el_LightGBM.py                      # Script Fase 1: LightGBM (18 vars técnicas)
├── Entrenando_el_RandomForest.py                  # Script Fase 1: Random Forest (18 vars técnicas)
├── Entrenando_el_CatBoost.py                      # Script Fase 1: CatBoost (18 vars técnicas)
├── Comparar_Modelos_Ensamble.py                   # Consolidador de tablas descriptivas y valores exactos (Fase 1)
├── Entrenando_Ensambles_Multivariables_Exogenas.py# Script Fase 2: Los 4 Ensambles con Cobre, VIX, DXY, WTI, Oro y US10Y
├── requirements.txt                               # Dependencias del proyecto
└── README.md                                      # Panel principal de navegación
```

## 🚀 Ejecución Rápida

Desde la raíz del repositorio:

```powershell
python -m pip install -r requirements.txt
# Fase 1: Modelos Endógenos (18 variables técnicas y calendario)
python Entrenando_el_XGBoost.py
python Entrenando_el_LightGBM.py
python Entrenando_el_RandomForest.py
python Entrenando_el_CatBoost.py
python Comparar_Modelos_Ensamble.py

# Fase 2: Nuevos Modelos Multivariables con Exógenas (50 variables: +Cobre, VIX, DXY, Oro, WTI, US10Y, S&P 500)
python src/exogenous_loader.py
python Entrenando_Ensambles_Multivariables_Exogenas.py
```

---

## 📊 Comparación Central en Holdout (60 sesiones: 10/07/2026 – 01/10/2026): Fase 1 (Sin Exógenas) vs. Fase 2 (Con Cobre, VIX, DXY, Oro, WTI y US10Y)

*(Tabla exportada en [`resultados_comparativos/tabla_comparativa_sin_vs_con_exogenas.csv`](resultados_comparativos/tabla_comparativa_sin_vs_con_exogenas.csv) y graficada en [`resultados_comparativos/impacto_exogenas_h20.png`](resultados_comparativos/impacto_exogenas_h20.png))*

| Horizonte ($h$) | Algoritmo de Ensamble | MAE Caminata Aleatoria (Referencia) | MAE Fase 1: Sin Exógenas (18 Vars) | MAE Fase 2: Con Exógenas Cobre/VIX/DXY (50 Vars) | Reducción de MAE al Incluir Exógenas | ¿Supera a Caminata Aleatoria en Fase 2? | Acierto Direccional Fase 1 (Sin Exógenas) | Acierto Direccional Fase 2 (Con Exógenas) | Mejora en Acierto Direccional |
|:---:|---|---:|---:|---:|---:|:---:|---:|---:|---:|
| **1 día** | **LightGBM** | 0.006881 S/ | 0.006974 S/ | **0.006850 S/** | **-0.000124 S/** | **SÍ** (`-0.000031 S/`) | 46.30 % *(25/54)* | 46.30 % *(25/54)* | 0.00 pp |
| **1 día** | **CatBoost** | 0.006881 S/ | 0.006955 S/ | **0.006896 S/** | **-0.000059 S/** | Casi Empate | 46.30 % *(25/54)* | **55.56 % *(30/54)*** | **+9.26 pp** |
| **1 día** | **XGBoost** | 0.006881 S/ | 0.006971 S/ | **0.006900 S/** | **-0.000070 S/** | Casi Empate | 42.59 % *(23/54)* | **51.85 % *(28/54)*** | **+9.26 pp** |
| **1 día** | **Random Forest** | 0.006881 S/ | 0.006925 S/ | **0.006905 S/** | **-0.000020 S/** | Casi Empate | 50.00 % *(27/54)* | 48.15 % *(26/54)* | -1.85 pp |
| **5 días** | **CatBoost** | 0.017458 S/ | 0.017758 S/ | **0.017262 S/** | **-0.000496 S/** | **SÍ** (`-0.000197 S/`) | 50.00 % *(27/54)* | **64.81 % *(35/54)*** | **+14.81 pp** |
| **5 días** | **XGBoost** | 0.017458 S/ | 0.017602 S/ | **0.017331 S/** | **-0.000271 S/** | **SÍ** (`-0.000127 S/`) | 50.00 % *(27/54)* | **61.11 % *(33/54)*** | **+11.11 pp** |
| **5 días** | **LightGBM** | 0.017458 S/ | 0.017640 S/ | **0.017339 S/** | **-0.000302 S/** | **SÍ** (`-0.000119 S/`) | 40.74 % *(22/54)* | **57.41 % *(31/54)*** | **+16.67 pp** |
| **5 días** | **Random Forest** | 0.017458 S/ | 0.017576 S/ | **0.017415 S/** | **-0.000161 S/** | **SÍ** (`-0.000043 S/`) | 48.15 % *(26/54)* | **55.56 % *(30/54)*** | **+7.41 pp** |
| **20 días** | **Random Forest** | 0.035905 S/ | 0.038959 S/ | **0.035302 S/** | **-0.003658 S/** | **SÍ** (`-0.000603 S/`) | 12.50 % *(5/40)* | **60.00 % *(24/40)*** | **+47.50 pp** |
| **20 días** | **XGBoost** | 0.035905 S/ | 0.039115 S/ | **0.035647 S/** | **-0.003468 S/** | **SÍ** (`-0.000258 S/`) | 5.00 % *(2/40)* | **52.50 % *(21/40)*** | **+47.50 pp** |
| **20 días** | **CatBoost** | 0.035905 S/ | 0.041446 S/ | **0.035868 S/** | **-0.005578 S/** | **SÍ** (`-0.000037 S/`) | 10.00 % *(4/40)* | **60.00 % *(24/40)*** | **+50.00 pp** |
| **20 días** | **LightGBM** | 0.035905 S/ | 0.039042 S/ | **0.036428 S/** | **-0.002613 S/** | Casi Empate | 2.50 % *(1/40)* | **50.00 % *(20/40)*** | **+47.50 pp** |

---

## 📋 Tablas de Valores Exactos de los Gráficos de Barras

### 1. Valores Exactos del Gráfico de Barras Multivariable con Exógenas ([`resumen_mae_y_acierto_direccional_exogenas.png`](resultados_comparativos/resumen_mae_y_acierto_direccional_exogenas.png))

*(Archivo CSV: [`resultados_comparativos/tabla_valores_exactos_grafico_barras_exogenas.csv`](resultados_comparativos/tabla_valores_exactos_grafico_barras_exogenas.csv))*

#### Panel A (Con Exógenas) — Error Absoluto Medio Exacto (MAE en Soles/USD y en Puntos Básicos)

| Horizonte ($h$) | Pronósticos ($N$) | Caminata Aleatoria (Referencia) | Random Forest + Exógenas | LightGBM + Exógenas | XGBoost + Exógenas | CatBoost + Exógenas |
|:---:|---:|---:|---:|---:|---:|---:|
| **1 día hábil** *(Soles/USD)* | 59 | 0.006881 S/ | 0.006905 S/ | **0.006850 S/** | 0.006900 S/ | 0.006896 S/ |
| **1 día hábil** *(Retorno pb)* | 59 | 20.34 pb | 20.41 pb | **20.25 pb** | 20.40 pb | 20.39 pb |
| **5 días hábiles** *(Soles/USD)* | 55 | 0.017458 S/ | **0.017415 S/** | **0.017339 S/** | **0.017331 S/** | **0.017262 S/** |
| **5 días hábiles** *(Retorno pb)* | 55 | 51.63 pb | **51.51 pb** | **51.28 pb** | **51.26 pb** | **51.05 pb** |
| **20 días hábiles** *(Soles/USD)* | 40 | 0.035905 S/ | **0.035302 S/** | 0.036428 S/ | **0.035647 S/** | **0.035868 S/** |
| **20 días hábiles** *(Retorno pb)* | 40 | 106.20 pb | **104.52 pb** | 107.89 pb | **105.53 pb** | **106.20 pb** |

#### Panel B (Con Exógenas) — Acierto Direccional Exacto (% y Conteo Exacto de Sesiones Acertadas)

| Horizonte ($h$) | Movimientos No Nulos | Referencia Azar | Random Forest + Exógenas | LightGBM + Exógenas | XGBoost + Exógenas | CatBoost + Exógenas |
|:---:|---:|---:|---:|---:|---:|---:|
| **1 día hábil** | 54 sesiones | 50.00 % | 48.15 % *(26/54)* | 46.30 % *(25/54)* | **51.85 %** *(28/54)* | **55.56 %** *(30/54)* |
| **5 días hábiles** | 54 sesiones | 50.00 % | **55.56 %** *(30/54)* | **57.41 %** *(31/54)* | **61.11 %** *(33/54)* | **64.81 %** *(35/54)* |
| **20 días hábiles** | 40 sesiones | 50.00 % | **60.00 %** *(24/40)* | **50.00 %** *(20/40)* | **52.50 %** *(21/40)* | **60.00 %** *(24/40)* |

---

### 2. Valores Exactos del Gráfico de Barras Fase 1 Sin Exógenas ([`resumen_mae_y_acierto_direccional.png`](resultados_comparativos/resumen_mae_y_acierto_direccional.png))

*(Archivo CSV: [`resultados_comparativos/tabla_valores_exactos_grafico_barras.csv`](resultados_comparativos/tabla_valores_exactos_grafico_barras.csv))*

| Horizonte ($h$) | Pronósticos ($N$) | Caminata Aleatoria | Random Forest (18 Vars) | CatBoost (18 Vars) | XGBoost (18 Vars) | LightGBM (18 Vars) | Acierto RF | Acierto CatBoost | Acierto XGBoost | Acierto LightGBM |
|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1 día** | 59 | 0.006881 S/ | 0.006925 S/ | 0.006955 S/ | 0.006971 S/ | 0.006974 S/ | 50.00 % *(27/54)* | 46.30 % *(25/54)* | 42.59 % *(23/54)* | 46.30 % *(25/54)* |
| **5 días** | 55 | 0.017458 S/ | 0.017576 S/ | 0.017758 S/ | 0.017602 S/ | 0.017640 S/ | 48.15 % *(26/54)* | 50.00 % *(27/54)* | 50.00 % *(27/54)* | 40.74 % *(22/54)* |
| **20 días** | 40 | 0.035905 S/ | 0.038959 S/ | 0.041446 S/ | 0.039115 S/ | 0.039042 S/ | 12.50 % *(5/40)* | 10.00 % *(4/40)* | 5.00 % *(2/40)* | 2.50 % *(1/40)* |

---

## 📖 Diccionario Resumen de Variables y Métricas

* **Variables Endógenas (18):** Rezagos del log-retorno (`PEN_log_ret_lag_{1,2,3,5,10,21}`), distancias a medias móviles (`price_vs_SMA_{5,10,20}`), oscilador `RSI`, volatilidad `GARCH_vol`, tendencia `MACD_{line,signal,histogram}`, estacionalidad (`month_{sin,cos}`, `day_of_week_{sin,cos}`) y momentum acumulado (`PEN_ret_acum_{5d,20d}`).
* **Variables Exógenas Causales (32):**
  * **Cobre (`Cobre_ret_{1d,5d,20d}`, `Cobre_vs_SMA_20`):** Retornos y desviación respecto a tendencia del principal metal de exportación peruano.
  * **VIX (`VIX_nivel_lag1`, `VIX_cambio_{1d,5d,20d}`, `VIX_zscore_20d`):** Aversión al riesgo e incertidumbre en mercados financieros globales.
  * **Índice Dólar (`DXY_ret_{1d,5d,20d}`, `DXY_vs_SMA_20`):** Fortaleza multilateral del dólar estadounidense.
  * **Bono EE. UU. 10 Años (`US10Y_nivel_lag1`, `US10Y_cambio_{5d,20d}`, `US10Y_vs_SMA_20`):** Rendimiento soberano de referencia en EE. UU.
  * **Oro y Petróleo WTI (`Oro_ret_{5d,20d}`, `Oro_vs_SMA_20`, `Petroleo_WTI_ret_{5d,20d}`, `Petroleo_WTI_vs_SMA_20`):** Cotizaciones de exportación aurífera e importación de crudo.
  * **S&P 500, Términos de Intercambio e Inflación BCRP (`SP500_ret_{5d,20d}`, `TermIntercambio_Cu_WTI_ret_20d`, `Cobre_vs_DXY_ret_20d`, `Inflacion_12M_pub_segura`):** Apetito global por riesgo, razón de precios Cobre/Petróleo e inflación interanual con rezago seguro de publicación (42 días hábiles).

El informe detallado completo se encuentra en [`docs/04_Resultados_Ensambles.md`](docs/04_Resultados_Ensambles.md).

## 👥 Equipo de Investigación (Grupo 3)
*   **Fredy Alberto Polar Urviola** (20221300)
*   **Renzo Raúl Hipólito Medina** (20190213)
*   **Ignacio Alonso Colán Lamas** (20221193)
*   **Dany Denilson Medina Quinto** (20192730)

---
*Desarrollado como proyecto aplicativo en ciencia de datos e investigación macroeconómica.*
