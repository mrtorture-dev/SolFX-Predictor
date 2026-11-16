# 📈 SolFX-Predictor
**Predicción del tipo de cambio entre el dólar y el sol mediante aprendizaje automático**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![XGBoost + PSO](https://img.shields.io/badge/Modelo-XGBoost%20%2B%20PSO-orange.svg)
![Status](https://img.shields.io/badge/Estado-Alineado%20al%20Informe%20Final-brightgreen.svg)

Sistema de aprendizaje supervisado para pronosticar la cotización del dólar en soles (**USD/PEN**) a **1, 5 y 20 días hábiles** —con prioridad en el **horizonte semanal ($h = 5$)**—, diseñado para apoyar a empresas peruanas con pagos en dólares a decidir cuándo comprar esa moneda o si conviene cubrir el pago con un contrato *forward*.

---

## 🎯 Enfoque Metodológico (Alineado al Informe Final)
1. **Datos y Fuentes (Sección 3.1 — Tabla 1):**
   * **Bloomberg (`PEN REGN Curncy`):** Tipo de cambio spot diario USD/PEN al cierre (`1994–2026`).
   * **BCRPData:** IPC de Lima Metropolitana e inflación mensual y a 12 meses, asignadas por su fecha de publicación.
   * **Yahoo Finance y FRED:** VIX, índice dólar (`DXY`), bono del Tesoro de EE. UU. a 10 años (`US10Y`), S&P 500 y precios de **cobre, oro y petróleo WTI**, con rezago causal $t - 1$.
2. **Preprocesamiento e Ingeniería de >80 Características (Sección 3.2):**
   * **97 características predictoras** divididas en **Grupo 1 (Propias del USD/PEN: 28 vars)**, **Grupo 2 (Globales y Commodities con indicador de tensión $s_t = \mathbf{1}[\text{VIX}_{t-1} > P_{80}]$ e interacciones: 55 vars)** y **Grupo 3 (Macroeconómicas BCRP: 14 vars)**.
   * Winsorización en percentiles 1 y 99 y estandarización calculadas únicamente sobre la ventana de entrenamiento.
3. **Reducción de Dimensionalidad con Lasso (Sección 3.3):**
   * Selección automática mediante **Lasso** dentro de la ventana de entrenamiento (conservando 35 a 38 variables clave y reduciendo más del 60 % de la dimensionalidad), comparando el conjunto completo de 97 variables frente al subconjunto seleccionado por Lasso.
4. **Modelos y Optimización con PSO vs. Búsqueda Aleatoria (Sección 3.4):**
   * Referencias: **Caminata Aleatoria (CA)**, **Regresión Ridge** y **Regresión Lasso**.
   * Único modelo candidato de árboles: **XGBoost** (profundidad entre 3 y 6 y parada temprana), ajustado mediante **Enjambre de Partículas (PSO)** ($\omega = 0,7$, $c_1 = c_2 = 1,5$) y comparado con una **Búsqueda Aleatoria** del mismo costo.
5. **Validación Progresiva, Métricas, SHAP y Exclusión por Grupos (Sección 3.5):**
   * Validación progresiva con ventana expansiva (avance de 20 en 20 sesiones con purga $t + h \le \tau$).
   * Métricas: $EAM_h$ (Soles/USD), **Error en soles por cada USD 10,000 pagados**, cociente relativo $U_h = EAM_h(\text{modelo}) / EAM_h(CA)$, Acierto Direccional $AD_h$ (%), pruebas de **Diebold-Mariano** y **Pesaran-Timmermann**, atribución **SHAP** y **prueba de exclusión por grupos (ablación)**.

---

## 📂 Arquitectura del Repositorio

```text
SolFX-Predictor/
├── data/
│   ├── raw/                                  # Datos crudos: Bloomberg, BCRPData y Yahoo Finance / FRED
│   └── processed/                            # Dataset oficial con 97 predictores (usdpen_train.csv, usdpen_holdout.csv)
├── docs/
│   ├── 01_Introduccion.md                    # Problema, pregunta, hipótesis (h=5 principal) y objetivos
│   ├── 02_Trabajos_Relacionados.md           # Pfahler (2021), Filippou et al. (2026), Palma & Salazar (2025), Pebrianti et al. (2023)
│   ├── 03_Metodologia.md                     # Formulación, Tabla 1, >80 vars, selección Lasso, PSO (Ec. 2) y métricas (Ec. 3)
│   └── 04_Experimentacion_y_Resultados.md    # Tablas descriptivas, valores exactos de barras, SHAP y ablación por grupos
├── models/                                   # Modelos entrenados de XGBoost + PSO (h = 1, 5, 20)
├── notebooks/                                # Notebooks de EDA, Preprocesamiento y Experimentación XGBoost + PSO
├── resultados_xgboost/                       # Tablas CSV descriptivas, gráficos de pronóstico, SHAP y barras con valores
├── src/                                      # Módulos fuente (bcrp_loader.py, preprocessing.py, exogenous_loader.py)
├── Entrenando_el_XGBoost.py                  # Script principal de experimentación (CA, Ridge, Lasso, XGBoost RS y PSO)
├── Informe_Resultados_XGBoost.md             # Redacción académica lista para la Sección 4 y 5 del informe
├── XGBoost.md                                # Guía de ejecución local
├── requirements.txt                          # Dependencias del proyecto
└── README.md                                 # Panel principal de navegación
```

---

## 🚀 Ejecución Reproducible

```powershell
python -m pip install -r requirements.txt
python src/exogenous_loader.py
python Entrenando_el_XGBoost.py
```

---

## 📊 Tabla Principal de Resultados y Valores Exactos del Gráfico de Barras (`resumen_metricas_uh_y_ad.png`)

*(Archivo CSV: [`resultados_xgboost/tabla_resultados_descriptiva.csv`](resultados_xgboost/tabla_resultados_descriptiva.csv))*

| Horizonte ($h$) | Modelo Evaluado | Predictores Activos | $EAM_h$ (Soles/USD) | Cociente $U_h = \frac{EAM(\text{Mod})}{EAM(CA)}$ | Error por Factura de USD 10,000 (S/) | Ahorro vs. CA por USD 10,000 (S/) | Acierto Direccional $AD_h$ (% y Sesiones) |
|:---:|---|---:|---:|---:|---:|---:|---:|
| **1 día** | Caminata Aleatoria (CA) | 0 | 0,006881 S/ | 1,0000 | S/ 68,81 | S/ 0,00 | — *(0/54)* |
| **1 día** | Regresión Ridge | 35 | 0,006944 S/ | 1,0092 | S/ 69,44 | -S/ 0,63 | 51,85 % *(28/54)* |
| **1 día** | Regresión Lasso | 35 | 0,006951 S/ | 1,0101 | S/ 69,51 | -S/ 0,70 | 51,85 % *(28/54)* |
| **1 día** | XGBoost + Búsqueda Aleatoria (Lasso) | 35 | 0,007008 S/ | 1,0185 | S/ 70,08 | -S/ 1,27 | 46,30 % *(25/54)* |
| **1 día** | **XGBoost + PSO (97 Vars Completas)** | 97 | **0,006875 S/** | **0,9991** | **S/ 68,75** | **+S/ 0,06** | **55,56 % *(30/54)*** |
| **1 día** | XGBoost + PSO + Selección Lasso | 35 | 0,007128 S/ | 1,0359 | S/ 71,28 | -S/ 2,47 | 51,85 % *(28/54)* |
| **5 días (Principal)** | Caminata Aleatoria (CA) | 0 | 0,017458 S/ | 1,0000 | S/ 174,58 | S/ 0,00 | — *(0/54)* |
| **5 días (Principal)** | Regresión Ridge | 38 | 0,017852 S/ | 1,0226 | S/ 178,52 | -S/ 3,94 | 44,44 % *(24/54)* |
| **5 días (Principal)** | Regresión Lasso | 38 | 0,018288 S/ | 1,0475 | S/ 182,88 | -S/ 8,30 | 46,30 % *(25/54)* |
| **5 días (Principal)** | **XGBoost + Búsqueda Aleatoria (Lasso)** | 38 | **0,017312 S/** | **0,9916** | **S/ 173,12** | **+S/ 1,46** | **57,41 % *(31/54)*** |
| **5 días (Principal)** | XGBoost + PSO (97 Vars Completas) | 97 | 0,017468 S/ | 1,0005 | S/ 174,68 | -S/ 0,09 | **57,41 % *(31/54)*** |
| **5 días (Principal)** | **XGBoost + PSO + Selección Lasso** | 38 | **0,017439 S/** | **0,9989** | **S/ 174,39** | **+S/ 0,19** | **57,41 % *(31/54)*** |
| **20 días** | Caminata Aleatoria (CA) | 0 | 0,035905 S/ | 1,0000 | S/ 359,05 | S/ 0,00 | — *(0/40)* |
| **20 días** | **Regresión Ridge** | 38 | **0,034420 S/** | **0,9586** | **S/ 344,20** | **+S/ 14,85** | **62,50 % *(25/40)*** |
| **20 días** | **Regresión Lasso** | 38 | **0,034876 S/** | **0,9713** | **S/ 348,76** | **+S/ 10,29** | **55,00 % *(22/40)*** |
| **20 días** | XGBoost + Búsqueda Aleatoria (Lasso) | 38 | 0,037029 S/ | 1,0313 | S/ 370,29 | -S/ 11,24 | 52,50 % *(21/40)* |
| **20 días** | XGBoost + PSO (97 Vars Completas) | 97 | 0,037398 S/ | 1,0416 | S/ 373,98 | -S/ 14,93 | **60,00 % *(24/40)*** |
| **20 días** | **XGBoost + PSO + Selección Lasso** | 38 | **0,035925 S/** | **1,0006** | **S/ 359,25** | **-S/ 0,20** | **60,00 % *(24/40)*** |

El detalle completo de la selección de variables con Lasso, las pruebas de Diebold-Mariano y Pesaran-Timmermann, el análisis SHAP y la prueba de exclusión por grupos se encuentran en [`docs/04_Experimentacion_y_Resultados.md`](docs/04_Experimentacion_y_Resultados.md) e [`Informe_Resultados_XGBoost.md`](Informe_Resultados_XGBoost.md).

## 👥 Equipo de Investigación (Grupo 3)
*   **Fredy Alberto Polar Urviola** (20221300)
*   **Renzo Raúl Hipólito Medina** (20190213)
*   **Ignacio Alonso Colán Lamas** (20221193)
*   **Dany Denilson Medina Quinto** (20192730)
