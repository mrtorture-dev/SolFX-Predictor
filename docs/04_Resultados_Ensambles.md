# 4. Implementación y Resultados Experimentales de Modelos de Ensamble (USD/PEN)

## 4.1 Diseño Experimental Común (Random Forest, XGBoost, LightGBM y CatBoost)

Se implementaron cuatro arquitecturas de aprendizaje supervisado basadas en ensambles de árboles de decisión —**Random Forest**, **XGBoost**, **LightGBM** y **CatBoost**— para estimar de manera **directa** el retorno logarítmico acumulado del tipo de cambio interbancario USD/PEN en horizontes de **1, 5 y 20 días hábiles**. Se emplearon los conjuntos de datos preprocesados del repositorio `SolFX-Predictor`, conformados por **8 462 observaciones de entrenamiento** (`1994-02-02` a `2026-07-09`) y **60 sesiones reservadas para evaluación fuera de muestra (*holdout*)** (`2026-07-10` a `2026-10-01`).

### Formulación de la Variable Objetivo y Reconstrucción del Nivel
Para cada horizonte $h \in \{1, 5, 20\}$, la etiqueta supervisada se define en **puntos básicos (pb)** ($1 \text{ pb} = 0.01\% = 10^{-4}$):
$$y_{t, h} = 10\,000 \times \ln\left(\frac{P_{t+h}}{P_t}\right)$$
donde $P_t$ es la cotización de cierre del dólar en soles en la fecha de origen $t$. A partir del retorno predicho $\hat{y}_{t,h}$, el tipo de cambio proyectado se reconstruye mediante:
$$\hat{P}_{t+h} = P_t \times \exp\left(\frac{\hat{y}_{t,h}}{10\,000}\right)$$

---

## 4.2 Diccionario Descriptivo de Variables Predictoras y Métricas de las Tablas

Para facilitar la lectura de las tablas de resultados, los archivos CSV y los gráficos de importancia, a continuación se detallan tanto las **18 variables predictoras (`COLUMNAS_X`)** como las **métricas de evaluación**.

### Tabla 4.1 — Explicación de las 18 Variables Predictoras (`COLUMNAS_X`)

> Todas las variables técnicas se calculan aplicando previamente un desplazamiento causal `shift(1)` sobre el precio o el retorno, garantizando que al cierre del día $t$ no exista fuga de información (*lookahead bias*).

| Familia | Código de Variable en Tablas / Gráficos | Nombre Descriptivo | Fórmula / Construcción | Interpretación Financiera en el Mercado USD/PEN |
|---|---|---|---|---|
| **Rezagos Autorregresivos (AR)** | `PEN_log_ret_lag_1` | Retorno logarítmico rezagado 1 día hábil | $\ln(P_{t-1} / P_{t-2})$ | Captura la autocorrelación de primer orden y el efecto de reversión o impulso inmediato del día previo. |
| **Rezagos Autorregresivos (AR)** | `PEN_log_ret_lag_2` | Retorno logarítmico rezagado 2 días hábiles | $\ln(P_{t-2} / P_{t-3})$ | Mide la variación registrada hace 2 sesiones bursátiles. |
| **Rezagos Autorregresivos (AR)** | `PEN_log_ret_lag_3` | Retorno logarítmico rezagado 3 días hábiles | $\ln(P_{t-3} / P_{t-4})$ | Refleja la dinámica intrasemanal de corto plazo (hace 3 sesiones). |
| **Rezagos Autorregresivos (AR)** | `PEN_log_ret_lag_5` | Retorno logarítmico rezagado 5 días hábiles (1 semana) | $\ln(P_{t-5} / P_{t-6})$ | Captura efectos de memoria semanal (5 días hábiles equivalen a una semana bursátil). |
| **Rezagos Autorregresivos (AR)** | `PEN_log_ret_lag_10` | Retorno logarítmico rezagado 10 días hábiles (2 semanas) | $\ln(P_{t-10} / P_{t-11})$ | Mide persistencia quincenal en las variaciones del tipo de cambio. |
| **Rezagos Autorregresivos (AR)** | `PEN_log_ret_lag_21` | Retorno logarítmico rezagado 21 días hábiles (1 mes) | $\ln(P_{t-21} / P_{t-22})$ | Captura ciclos mensuales de liquidez, pago de obligaciones y cierre contable. |
| **Distancias a Medias Móviles** | `price_vs_SMA_5` | Distancia relativa a la Media Móvil Simple de 5 días | $\frac{P_{t-1} - \text{SMA}_5(P_{t-1})}{\text{SMA}_5(P_{t-1})}$ | Desviación porcentual del precio frente a su promedio de la última semana; mide extensión de corto plazo. |
| **Distancias a Medias Móviles** | `price_vs_SMA_10` | Distancia relativa a la Media Móvil Simple de 10 días | $\frac{P_{t-1} - \text{SMA}_{10}(P_{t-1})}{\text{SMA}_{10}(P_{t-1})}$ | Desviación porcentual frente al nivel promedio quincenal (10 sesiones). |
| **Distancias a Medias Móviles** | `price_vs_SMA_20` | Distancia relativa a la Media Móvil Simple de 20 días | $\frac{P_{t-1} - \text{SMA}_{20}(P_{t-1})}{\text{SMA}_{20}(P_{t-1})}$ | Desviación porcentual frente al equilibrio promedio mensual (20 sesiones); actúa como ancla de reversión a la media. |
| **Osciladores y Volatilidad** | `RSI` | Índice de Fuerza Relativa (14 sesiones) normalizado | $\frac{\text{RSI}_{14}(P_{t-1}) - 50}{50} \in [-1, 1]$ | Mide la velocidad de subidas frente a bajadas. Valores cercanos a $+1$ indican sobrecompra del dólar; cercanos a $-1$, sobreventa. |
| **Osciladores y Volatilidad** | `GARCH_vol` | Volatilidad histórica móvil de 21 sesiones (Proxy GARCH) | $\sigma_{21}(r_{t-1})$ | Desviación estándar móvil de un mes sobre retornos diarios; cuantifica el régimen de incertidumbre y tensión cambiaria. |
| **Tendencia (MACD)** | `MACD_line` | Línea MACD normalizada por nivel de precio | $\frac{\text{EMA}_{12}(P_{t-1}) - \text{EMA}_{26}(P_{t-1})}{P_{t-1}}$ | Diferencial relativo entre la media exponencial rápida (12 días) y lenta (26 días); identifica la dirección de la tendencia. |
| **Tendencia (MACD)** | `MACD_signal` | Línea de Señal del MACD (EMA de 9 sesiones) | $\text{EMA}_9(\text{MACD\_line})$ | Media móvil exponencial de la línea MACD usada para detectar cruces y agotamiento de tendencia. |
| **Tendencia (MACD)** | `MACD_histogram` | Histograma de divergencia MACD | $\text{MACD\_line} - \text{MACD\_signal}$ | Mide la aceleración o desaceleración del momento alcista/bajista del tipo de cambio. |
| **Calendario Cíclico** | `month_sin` | Componente Seno del mes calendario | $\sin(2\pi \cdot \text{mes} / 12)$ | Codifica la posición estacional dentro del año (ej. periodos tributarios en marzo/abril o gratificaciones en julio/diciembre). |
| **Calendario Cíclico** | `month_cos` | Componente Coseno del mes calendario | $\cos(2\pi \cdot \text{mes} / 12)$ | Complemento ortogonal de `month_sin` que garantiza continuidad matemática entre diciembre (mes 12) y enero (mes 1). |
| **Calendario Cíclico** | `day_of_week_sin` | Componente Seno del día de la semana | $\sin(2\pi \cdot \text{día} / 7)$ | Captura patrones recurrentes de liquidez intrasemanal (apertura de lunes a cierre de viernes). |
| **Calendario Cíclico** | `day_of_week_cos` | Componente Coseno del día de la semana | $\cos(2\pi \cdot \text{día} / 7)$ | Complemento ortogonal de `day_of_week_sin` para representar cíclicamente el día de la semana. |

---

### Tabla 4.2 — Explicación de las Columnas y Métricas de las Tablas de Resultados

| Columna en CSV (`tabla_comparativa_ensambles.csv`) | Nombre Descriptivo en Tablas | Unidad de Medida | Definición Matemática e Interpretación Práctica |
|---|---|---|---|
| `horizonte_dias_habiles` | **Horizonte de Pronóstico ($h$)** | Días hábiles | Plazo hacia adelante de la predicción: **1 día** (diario), **5 días** (1 semana bursátil) y **20 días** (~1 mes bursátil). |
| `n_pronosticos` | **Pronósticos Evaluados ($N$)** | Observaciones | Cantidad de pares $(t, t+h)$ evaluables dentro de los 60 días del *holdout*: $N = 60 - h$ (**59**, **55** y **40** pronósticos). |
| `n_cambios_no_nulos` | **Sesiones con Variación Real ($\Delta P \neq 0$)** | Observaciones | Número de pronósticos donde el tipo de cambio real tuvo un movimiento distinto de cero (**54** en $h=1$, **54** en $h=5$ y **40** en $h=20$). |
| `mae_cv_bp` | **Error en Validación Cruzada Temporal (CV MAE)** | Puntos básicos (pb) | Error Absoluto Medio promedio en los 3 cortes de validación (`TimeSeriesSplit` de 252 días con `gap=h`), usado para elegir hiperparámetros. |
| `mae_precio_modelo_pen` | **Error Medio del Modelo en Nivel (MAE Modelo)** | **Soles por USD (S/)** | $\frac{1}{N}\sum |P_{t+h} - \hat{P}_{t+h}|$. Diferencia promedio en soles entre el dólar real y el predicho (ej. `0.00692` S/ $\approx 0.69$ céntimos). **Menor es mejor.** |
| `mae_precio_baseline_pen` | **Error Medio Caminata Aleatoria (MAE Base)** | **Soles por USD (S/)** | $\frac{1}{N}\sum |P_{t+h} - P_t|$. Error promedio del benchmark ingenuo que predice que el tipo de cambio no cambiará ($\hat{P}_{t+h} = P_t$). |
| `diferencia_mae_vs_baseline_pen` | **Diferencia de MAE vs. Caminata Aleatoria ($\Delta\text{MAE}$)** | **Soles por USD (S/)** | $\text{MAE}_{\text{Modelo}} - \text{MAE}_{\text{Base}}$. Valores negativos indican que el modelo redujo el error absoluto medio respecto a la caminata aleatoria. |
| `rmse_precio_modelo_pen` | **Error Cuadrático Medio en Nivel (RMSE Modelo)** | **Soles por USD (S/)** | $\sqrt{\frac{1}{N}\sum (P_{t+h} - \hat{P}_{t+h})^2}$. Penaliza con mayor severidad las desviaciones grandes o picos abruptos del tipo de cambio. |
| `rmse_precio_baseline_pen` | **Error Cuadrático Caminata Aleatoria (RMSE Base)** | **Soles por USD (S/)** | $\sqrt{\frac{1}{N}\sum (P_{t+h} - P_t)^2}$. RMSE del pronóstico sin cambio en soles por dólar. |
| `mae_ret_modelo_bp` | **Error Medio en Retornos (MAE Retorno)** | Puntos básicos (pb) | Error absoluto medio medido directamente sobre la tasa de variación logarítmica $y_{t,h}$ en puntos básicos ($1\text{ pb} = 0.01\%$). |
| `theil_u_rmse` | **Coeficiente U de Theil (Ratio RMSE)** | Adimensional (Ratio) | $\text{RMSE}_{\text{Modelo}} / \text{RMSE}_{\text{Base}}$ sobre retornos. **Si $U < 1.0$, el modelo supera a la caminata aleatoria** en error cuadrático medio; si $U > 1.0$, tiene mayor error. |
| `acierto_direccional_porcentaje` | **Acierto Direccional de Subida/Bajada (Hit Rate)** | Porcentaje (`%`) | Porcentaje de sesiones con variación no nula en las que el modelo acertó el signo del movimiento real ($\text{sign}(\hat{y}_{t,h}) = \text{sign}(y_{t,h})$). |
| `r2_ret` | **Coeficiente de Determinación ($R^2$ Retornos)** | Adimensional | Proporción de la varianza del retorno explicada fuera de muestra. En series cambiarias ruidosas suele ser cercano a $0$ o negativo. |

---

## 4.3 Tablas de Valores Exactos del Gráfico de Barras (`resumen_mae_y_acierto_direccional.png`)

Dado que en el gráfico de barras comparativo ([`resumen_mae_y_acierto_direccional.png`](../resultados_comparativos/resumen_mae_y_acierto_direccional.png)) las barras dentro de cada horizonte para el **Error Absoluto Medio (Panel A)** se encuentran muy próximas entre sí en escala visual, a continuación se presentan las **tablas con los valores exactos** que alimentan cada una de las barras de ambos paneles (también exportadas en [`tabla_valores_exactos_grafico_barras.csv`](../resultados_comparativos/tabla_valores_exactos_grafico_barras.csv)).

### Tabla 4.3A — Valores Exactos del Panel Izquierdo del Gráfico de Barras: Error Absoluto Medio (MAE en Soles por USD y en Puntos Básicos)

| Horizonte de Pronóstico | Pronósticos Evaluados ($N$) | Caminata Aleatoria (Referencia Sin Cambio) | Random Forest | CatBoost | XGBoost | LightGBM | Mejor Ensamble en MAE (y Diferencia vs. Referencia) |
|---|---:|---:|---:|---:|---:|---:|---|
| **$h = 1$ día hábil** *(en Soles/USD)* | 59 | **0.006881 S/** | **0.006925 S/** | 0.006955 S/ | 0.006971 S/ | 0.006974 S/ | **Random Forest** (`+0.000043 S/` vs. RW) |
| **$h = 1$ día hábil** *(en Retorno pb)* | 59 | **20.34 pb** | **20.47 pb** | 20.56 pb | 20.61 pb | 20.62 pb | **Random Forest** (`+0.13 pb` vs. RW) |
| **$h = 5$ días hábiles** *(en Soles/USD)* | 55 | **0.017458 S/** | **0.017576 S/** | 0.017758 S/ | 0.017602 S/ | 0.017640 S/ | **Random Forest** (`+0.000118 S/` vs. RW) |
| **$h = 5$ días hábiles** *(en Retorno pb)* | 55 | **51.63 pb** | **51.98 pb** | 52.51 pb | 52.05 pb | 52.17 pb | **Random Forest** (`+0.35 pb` vs. RW) |
| **$h = 20$ días hábiles** *(en Soles/USD)* | 40 | **0.035905 S/** | **0.038959 S/** | 0.041446 S/ | 0.039115 S/ | 0.039042 S/ | **Random Forest** (`+0.003054 S/` vs. RW) |
| **$h = 20$ días hábiles** *(en Retorno pb)* | 40 | **106.20 pb** | **115.26 pb** | 122.64 pb | 115.71 pb | 115.48 pb | **Random Forest** (`+9.06 pb` vs. RW) |

---

### Tabla 4.3B — Valores Exactos del Panel Derecho del Gráfico de Barras: Acierto Direccional (%) y Conteo Exacto de Sesiones Acertadas

| Horizonte de Pronóstico | Sesiones con Variación Real ($\Delta P \neq 0$) | Umbral de Referencia (Azar Puro) | Random Forest (% y Conteo Exacto) | CatBoost (% y Conteo Exacto) | XGBoost (% y Conteo Exacto) | LightGBM (% y Conteo Exacto) | Mayor Acierto Direccional en el Horizonte |
|---|---:|---:|---:|---:|---:|---:|---|
| **$h = 1$ día hábil** | 54 de 59 sesiones | 50.00 % | **50.00 %** *(27 / 54 días)* | 46.30 % *(25 / 54 días)* | 42.59 % *(23 / 54 días)* | 46.30 % *(25 / 54 días)* | **Random Forest** (`27 de 54` sesiones) |
| **$h = 5$ días hábiles** | 54 de 55 sesiones | 50.00 % | 48.15 % *(26 / 54 días)* | **50.00 %** *(27 / 54 días)* | **50.00 %** *(27 / 54 días)* | 40.74 % *(22 / 54 días)* | **XGBoost y CatBoost** (`27 de 54` sesiones) |
| **$h = 20$ días hábiles** | 40 de 40 sesiones | 50.00 % | **12.50 %** *(5 / 40 días)* | 10.00 % *(4 / 40 días)* | 5.00 % *(2 / 40 días)* | 2.50 % *(1 / 40 días)* | **Random Forest** (`5 de 40` sesiones) |

---

## 4.4 Tablas Detalladas por Horizonte Fuera de Muestra (Holdout: 10/07/2026 – 01/10/2026)

### Horizonte $h = 1$ día hábil ($N = 59$ pronósticos evaluados; $54$ con movimiento no nulo)

| Algoritmo de Ensamble | Error Validación Cruzada (CV MAE, pb) | MAE del Modelo en Nivel (Soles/USD) | MAE Caminata Aleatoria (Soles/USD) | RMSE del Modelo en Nivel (Soles/USD) | RMSE Caminata Aleatoria (Soles/USD) | Ratio U de Theil (< 1 mejora RW) | Acierto Direccional (% y Días) | Ajuste sobre Retornos ($R^2$) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Random Forest** | 27.23 pb | **0.006925 S/** | 0.006881 S/ | 0.009343 S/ | 0.009411 S/ | 0.9929 | **50.00 % (27/54)** | 0.0030 |
| **CatBoost** | **27.18 pb** | 0.006955 S/ | 0.006881 S/ | 0.009388 S/ | 0.009411 S/ | 0.9976 | 46.30 % (25/54) | -0.0065 |
| **XGBoost** | 27.26 pb | 0.006971 S/ | 0.006881 S/ | 0.009349 S/ | 0.009411 S/ | 0.9936 | 42.59 % (23/54) | 0.0017 |
| **LightGBM** | 27.35 pb | 0.006974 S/ | 0.006881 S/ | **0.009277 S/** | 0.009411 S/ | **0.9861** | 46.30 % (25/54) | **0.0166** |

### Horizonte $h = 5$ días hábiles ($N = 55$ pronósticos evaluados; $54$ con movimiento no nulo)

| Algoritmo de Ensamble | Error Validación Cruzada (CV MAE, pb) | MAE del Modelo en Nivel (Soles/USD) | MAE Caminata Aleatoria (Soles/USD) | RMSE del Modelo en Nivel (Soles/USD) | RMSE Caminata Aleatoria (Soles/USD) | Ratio U de Theil (< 1 mejora RW) | Acierto Direccional (% y Días) | Ajuste sobre Retornos ($R^2$) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Random Forest** | **65.59 pb** | **0.017576 S/** | 0.017458 S/ | **0.024326 S/** | 0.024435 S/ | **0.9956** | 48.15 % (26/54) | **-0.0150** |
| **XGBoost** | 66.47 pb | 0.017602 S/ | 0.017458 S/ | 0.024360 S/ | 0.024435 S/ | 0.9970 | **50.00 % (27/54)** | -0.0180 |
| **LightGBM** | 66.62 pb | 0.017640 S/ | 0.017458 S/ | 0.024425 S/ | 0.024435 S/ | 0.9997 | 40.74 % (22/54) | -0.0234 |
| **CatBoost** | 66.03 pb | 0.017758 S/ | 0.017458 S/ | 0.024400 S/ | 0.024435 S/ | 0.9987 | **50.00 % (27/54)** | -0.0212 |

### Horizonte $h = 20$ días hábiles ($N = 40$ pronósticos evaluados; $40$ con movimiento no nulo)

| Algoritmo de Ensamble | Error Validación Cruzada (CV MAE, pb) | MAE del Modelo en Nivel (Soles/USD) | MAE Caminata Aleatoria (Soles/USD) | RMSE del Modelo en Nivel (Soles/USD) | RMSE Caminata Aleatoria (Soles/USD) | Ratio U de Theil (< 1 mejora RW) | Acierto Direccional (% y Días) | Ajuste sobre Retornos ($R^2$) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Random Forest** | **132.40 pb** | **0.038959 S/** | 0.035905 S/ | **0.045494 S/** | 0.042619 S/ | **1.0678** | **12.50 % (5/40)** | **-0.1636** |
| **LightGBM** | 133.82 pb | 0.039042 S/ | 0.035905 S/ | 0.045570 S/ | 0.042619 S/ | 1.0694 | 2.50 % (1/40) | -0.1670 |
| **XGBoost** | 134.00 pb | 0.039115 S/ | 0.035905 S/ | 0.045808 S/ | 0.042619 S/ | 1.0751 | 5.00 % (2/40) | -0.1794 |
| **CatBoost** | 133.93 pb | 0.041446 S/ | 0.035905 S/ | 0.048261 S/ | 0.042619 S/ | 1.1331 | 10.00 % (4/40) | -0.3100 |

---

## 4.5 Análisis de Importancia de Variables y Comportamiento por Horizonte

1. **Corto plazo ($h = 1$ día hábil):**
   - Los cuatro modelos lograron un coeficiente **Theil's U menor a 1.0 en RMSE** (`0.9861` en LightGBM, `0.9929` en Random Forest, `0.9936` en XGBoost y `0.9976` en CatBoost), reduciendo ligeramente el error cuadrático medio (`0.009277 S/` en LightGBM frente a `0.009411 S/` de la caminata aleatoria).
   - En **MAE en Soles/USD**, **Random Forest** obtuvo el menor error entre los ensambles (`0.006925 S/`), quedando a solo `0.000043 S/` (4.3 cienmilésimas de sol) de la caminata aleatoria (`0.006881 S/`), con un acierto direccional de `50.00 %` (`27 de 54` días).
   - Las variables dominantes fueron las distancias a medias móviles (`price_vs_SMA_20`, `price_vs_SMA_10`, `price_vs_SMA_5`) y el primer rezago autorregresivo (`PEN_log_ret_lag_1`).

2. **Mediano plazo semanal ($h = 5$ días hábiles):**
   - **Random Forest** lideró tanto en validación cruzada (`65.59 pb`) como en MAE fuera de muestra (`0.017576 S/` vs. `0.017458 S/` de la caminata aleatoria) y en RMSE (`0.024326 S/` vs. `0.024435 S/`, con `Theil U = 0.9956`).
   - **XGBoost** y **CatBoost** alcanzaron el mayor acierto direccional (`50.00 %`, acertando `27 de 54` sesiones). Las variables más influyentes fueron `GARCH_vol`, `price_vs_SMA_20`, `RSI`, `MACD_signal` y `month_cos`.

3. **Horizonte mensual ($h = 20$ días hábiles):**
   - A 20 sesiones hábiles, ningún modelo basado únicamente en indicadores técnicos endógenos superó a la caminata aleatoria (`0.035905 S/`). **Random Forest** (`0.038959 S/`) y **LightGBM** (`0.039042 S/`) registraron el menor error entre los ensambles.
   - Esto respalda empíricamente la necesidad de incorporar las covariables macroeconómicas y financieras globales (DXY, VIX, Cobre, Oro, WTI y diferencial de tasas) para capturar tendencias de mediano plazo.
