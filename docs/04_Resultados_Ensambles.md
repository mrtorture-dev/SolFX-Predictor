# 4. Implementación y Resultados Experimentales de Modelos de Ensamble (USD/PEN)

## 4.1 Diseño Experimental Común (Random Forest, XGBoost, LightGBM y CatBoost)

Se implementaron cuatro arquitecturas de aprendizaje supervisado basadas en ensambles de árboles de decisión —**Random Forest**, **XGBoost**, **LightGBM** y **CatBoost**— para estimar de manera **directa** el retorno logarítmico acumulado del tipo de cambio interbancario USD/PEN en horizontes de **1, 5 y 20 días hábiles**. Se emplearon los conjuntos de datos preprocesados del repositorio `SolFX-Predictor`, conformados por **8 462 observaciones de entrenamiento** (`1994-02-02` a `2026-07-09`) y **60 sesiones reservadas para evaluación fuera de muestra (holdout)** (`2026-07-10` a `2026-10-01`).

### Variables Predictoras y Control Anti-Leakage
Para garantizar comparabilidad estricta entre los cuatro algoritmos y evitar fuga de información temporal (*data leakage*), se utilizaron exactamente las mismas **18 variables técnicas rezagadas y de calendario**:
- **Rezagos autorregresivos (6):** `PEN_log_ret_lag_1`, `PEN_log_ret_lag_2`, `PEN_log_ret_lag_3`, `PEN_log_ret_lag_5`, `PEN_log_ret_lag_10`, `PEN_log_ret_lag_21`.
- **Distancias porcentuales a medias móviles simples (3):** `price_vs_SMA_5`, `price_vs_SMA_10`, `price_vs_SMA_20`.
- **Osciladores, tendencia y volatilidad causal (5):** `RSI` (normalizado en $[-1, 1]$), `GARCH_vol` (volatilidad móvil de 21 sesiones sobre $r_{t-1}$), `MACD_line`, `MACD_signal`, `MACD_histogram`.
- **Codificación cíclica de calendario (4):** `month_sin`, `month_cos`, `day_of_week_sin`, `day_of_week_cos`.

> **Nota de auditoría temporal:** Las series mensuales de IPC del archivo preprocesado se alinean por mes de referencia sin consignar la fecha exacta de publicación oficial del INEI/BCRP. Por rigor metodológico, se excluyeron de esta línea base hasta auditar su calendario real de divulgación.

### Formulación de la Variable Objetivo y Reconstrucción del Nivel
Para cada horizonte $h \in \{1, 5, 20\}$, la etiqueta supervisada se define en puntos básicos (pb):
$$y_{t, h} = 10\,000 \times \ln\left(\frac{P_{t+h}}{P_t}\right)$$
donde $P_t$ es la cotización de cierre del dólar en soles en la fecha de origen $t$. A partir del retorno predicho $\hat{y}_{t,h}$, el tipo de cambio proyectado se reconstruye mediante:
$$\hat{P}_{t+h} = P_t \times \exp\left(\frac{\hat{y}_{t,h}}{10\,000}\right)$$

### Selección de Hiperparámetros y Validación Temporal
En cada modelo y horizonte se evaluaron tres configuraciones conservadoras de hiperparámetros mediante validación temporal expansiva (`TimeSeriesSplit` de 3 cortes con ventanas de validación de 252 sesiones bursátiles y una separación de purga `gap = h` para impedir solapamiento entre etiquetas de entrenamiento y validación). La evaluación en el bloque *holdout* se realizó bajo esquema de **origen móvil (*rolling origin*)**, comparando cada modelo frente al benchmark de **caminata aleatoria sin deriva** ($\hat{P}_{t+h}^{RW} = P_t$).

---

## 4.2 Resultados Comparativos Fuera de Muestra (Holdout: 10/07/2026 – 01/10/2026)

### Horizonte $h = 1$ día hábil ($N = 59$ pronósticos evaluados)

| Modelo | CV MAE (pb) | MAE Nivel (soles/USD) | MAE Caminata Aleatoria | RMSE Nivel (soles/USD) | Theil's U (RMSE) | Acierto Direccional | $R^2$ Retornos |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Random Forest** | 27.23 | **0.00692** | 0.00688 | 0.00891 | 0.9929 | **50.0 %** | 0.0030 |
| **CatBoost** | **27.18** | 0.00696 | 0.00688 | 0.00895 | 0.9976 | 46.3 % | -0.0065 |
| **LightGBM** | 27.35 | 0.00697 | 0.00688 | **0.00885** | **0.9861** | 46.3 % | **0.0166** |
| **XGBoost** | 27.26 | 0.00697 | 0.00688 | 0.00892 | 0.9936 | 42.6 % | 0.0017 |

### Horizonte $h = 5$ días hábiles ($N = 55$ pronósticos evaluados)

| Modelo | CV MAE (pb) | MAE Nivel (soles/USD) | MAE Caminata Aleatoria | RMSE Nivel (soles/USD) | Theil's U (RMSE) | Acierto Direccional | $R^2$ Retornos |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Random Forest** | **65.59** | **0.01758** | 0.01746 | **0.02266** | **0.9956** | 48.1 % | **-0.0150** |
| **XGBoost** | 66.47 | 0.01760 | 0.01746 | 0.02269 | 0.9970 | **50.0 %** | -0.0180 |
| **LightGBM** | 66.62 | 0.01764 | 0.01746 | 0.02275 | 0.9997 | 40.7 % | -0.0234 |
| **CatBoost** | 66.03 | 0.01776 | 0.01746 | 0.02272 | 0.9987 | **50.0 %** | -0.0212 |

### Horizonte $h = 20$ días hábiles ($N = 40$ pronósticos evaluados)

| Modelo | CV MAE (pb) | MAE Nivel (soles/USD) | MAE Caminata Aleatoria | RMSE Nivel (soles/USD) | Theil's U (RMSE) | Acierto Direccional | $R^2$ Retornos |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Random Forest** | **132.40** | **0.03896** | 0.03591 | **0.04422** | **1.0678** | **12.5 %** | **-0.1636** |
| **LightGBM** | 133.82 | 0.03904 | 0.03591 | 0.04428 | 1.0694 | 2.5 % | -0.1670 |
| **XGBoost** | 134.00 | 0.03912 | 0.03591 | 0.04451 | 1.0751 | 5.0 % | -0.1794 |
| **CatBoost** | 133.93 | 0.04145 | 0.03591 | 0.04689 | 1.1331 | 10.0 % | -0.3100 |

---

## 4.3 Análisis de Importancia de Variables y Comportamiento por Horizonte

1. **Corto plazo ($h = 1$ día hábil):**
   - Todos los modelos lograron un coeficiente **Theil's U menor a 1.0 en RMSE** (`0.9861` en LightGBM y `0.9929` en Random Forest), lo que indica una ligera reducción del error cuadrático medio frente a la caminata aleatoria al amortiguar picos extremos.
   - En términos de **MAE en soles/USD**, **Random Forest** obtuvo el menor error entre los ensambles (`0.00692` PEN/USD) y un acierto direccional de `50.0 %`, quedando prácticamente empatado con la caminata aleatoria (`0.00688` PEN/USD).
   - Las variables dominantes en los cuatro algoritmos fueron las distancias relativas a medias móviles (`price_vs_SMA_20`, `price_vs_SMA_10`, `price_vs_SMA_5`) y el primer rezago autorregresivo (`PEN_log_ret_lag_1`), capturando reversión a la media de muy corto plazo.

2. **Mediano plazo semanal ($h = 5$ días hábiles):**
   - **Random Forest** volvió a liderar en validación cruzada (`65.59` pb) y en prueba fuera de muestra (`0.01758` PEN/USD vs. `0.01746` PEN/USD de la caminata aleatoria), manteniendo además el menor Theil's U (`0.9956`).
   - **XGBoost** y **CatBoost** alcanzaron un `50.0 %` de acierto direccional. Las variables más relevantes pasaron a ser `GARCH_vol`, `price_vs_SMA_20`, `RSI`, `MACD_signal` y la estacionalidad mensual (`month_cos`).

3. **Horizonte mensual ($h = 20$ días hábiles):**
   - En el horizonte de 20 sesiones, ningún modelo superó a la caminata aleatoria (`0.03591` PEN/USD); **Random Forest** (`0.03896`) y **LightGBM** (`0.03904`) mostraron menor degradación que **XGBoost** (`0.03912`) y **CatBoost** (`0.04145`).
   - Las variables técnicas autorregresivas pierden capacidad predictiva a un mes vista, concentrándose la importancia en `GARCH_vol`, `month_sin` y las líneas `MACD`. Esto confirma empíricamente la hipótesis de Meese y Rogoff y los hallazgos de Palma y Salazar (2025) cuando no se incluyen fundamentos macroeconómicos y financieros globales (DXY, VIX, Cobre, diferencial de tasas).

---

## 4.4 Conclusiones y Trabajo Futuro

1. **Viabilidad y consistencia del pipeline:** Se consolidó una arquitectura reproducible y libre de *data leakage* que entrena, valida y compara en condiciones idénticas **Random Forest**, **XGBoost**, **LightGBM** y **CatBoost**.
2. **Desempeño comparativo entre ensambles:** El ensamblado por *bagging* (**Random Forest**) mostró la mayor estabilidad frente al ruido de alta frecuencia en el MAE de niveles para los tres horizontes, mientras que **LightGBM** destacó en RMSE y $R^2$ a 1 día hábil (`Theil U = 0.9861`, $R^2 = 0.0166$).
3. **Siguientes pasos metodológicos:**
   - Integrar las series diarias exógenas globales (DXY, VIX, Cobre, Oro, WTI y tasas de rendimientos de bonos soberanos) respetando rezagos causales `shift(1)`.
   - Auditar el calendario de publicación oficial de las series mensuales del BCRP antes de activarlas en producción.
   - Incorporar la optimización estocástica mediante **Enjambre de Partículas (PSO)** y el análisis de atribución local con **valores SHAP**.
