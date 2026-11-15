# 4. Implementación y Resultados Experimentales de Modelos de Ensamble (USD/PEN)

## 4.1 Por Qué se Desarrollaron Dos Fases Experimentales (Fase 1 Endógena vs. Fase 2 Multivariable con Cobre, VIX, DXY, Oro, WTI y US10Y)

1. **Diagnóstico de la exportación original de Bloomberg (Fase 1 — Línea Base Endógena de 18 variables):**
   En los archivos crudos exportados originalmente de la terminal Bloomberg (`data/raw/grid_*.xlsx` y `Datos_Bloomberg_PEN_FX.xlsx`), las hojas correspondientes a **Cobre LME (`LMCADY LME Comdty`)**, **Índice Dólar (`DXY Curncy`)** y **Bono del Tesoro de EE. UU. a 10 años (`USGG10YR Index`)** contenían únicamente fórmulas `=BDH(...)` sin resolver (`#NAME?`, con `0` observaciones válidas). Por esa razón, los scripts iniciales de referencia solo emplearon las **18 variables técnicas y de calendario endógenas** derivadas del propio tipo de cambio USD/PEN. Como se demuestra en la Fase 1, los indicadores puramente técnicos funcionan a 1 día hábil, pero pierden capacidad predictiva a **20 días hábiles** (`MAE = 0.038959` a `0.041446 S/` frente a `0.035905 S/` de la caminata aleatoria, y acierto direccional de solo `2.5 %` a `12.5 %`).

2. **Solución e Integración Multivariable (Fase 2 — Ensambles con 50 Variables incluyendo Cobre, VIX, DXY, Oro, Petróleo WTI, Bono EE. UU. 10Y, S&P 500 e Inflación BCRP):**
   Mediante el módulo [`src/exogenous_loader.py`](../src/exogenous_loader.py), se descargaron e integraron las series históricas diarias reales (`1994-01-03` a `2026-10-30`) de **Cobre, VIX, DXY, Oro, Petróleo WTI, Bono del Tesoro de EE. UU. a 10 años (`US10Y`), S&P 500 e Inflación BCRP**, construyendo **32 nuevas variables exógenas multiescala (a 1, 5 y 20 sesiones)** bajo control causal estricto `shift(1)` (cero *data leakage*). Adicionalmente, se incorporó ponderación exponencial por régimen macroeconómico reciente (*half-life* temporal) para evitar que la devaluación estructural de los años 90 sesgue positivamente los pronósticos a 20 días.

---

## 4.2 Diccionario Descriptivo de Variables Predictoras (50 Variables) y Métricas de Evaluación

### Tabla 4.1A — Las 18 Variables Endógenas Técnicas y de Calendario (Fase 1 y Fase 2)

| Familia | Código de Variable | Nombre Descriptivo | Fórmula Causal (`shift(1)`) | Interpretación Financiera en el Mercado USD/PEN |
|---|---|---|---|---|
| **Rezagos Autorregresivos (AR)** | `PEN_log_ret_lag_1` a `lag_21` | Retornos logarítmicos del USD/PEN rezagados $1, 2, 3, 5, 10$ y $21$ sesiones | $\ln(P_{t-k} / P_{t-k-1})$ | Capturan memoria de corto plazo (1 a 3 días), semanal (5 días), quincenal (10 días) y mensual (21 días). |
| **Distancias a Medias Móviles** | `price_vs_SMA_5`, `10`, `20` | Distancia relativa del USD/PEN a su Media Móvil Simple de 5, 10 y 20 días | $\frac{P_{t-1} - \text{SMA}_w(P_{t-1})}{\text{SMA}_w(P_{t-1})}$ | Miden sobrecompra o sobreventa porcentual respecto al equilibrio semanal, quincenal y mensual. |
| **Oscilador de Momento** | `RSI` | Índice de Fuerza Relativa de 14 sesiones normalizado | $\frac{\text{RSI}_{14}(P_{t-1}) - 50}{50} \in [-1, 1]$ | Velocidad de subidas vs. bajadas: valores cercanos a $+1$ indican sobrecompra; cercanos a $-1$, sobreventa. |
| **Volatilidad Histórica** | `GARCH_vol` | Volatilidad móvil de 21 sesiones (Proxy GARCH) | $\sigma_{21}(r_{t-1})$ | Desviación estándar móvil mensual de los retornos diarios; cuantifica el régimen de incertidumbre local. |
| **Tendencia (MACD)** | `MACD_line`, `MACD_signal`, `MACD_histogram` | Línea MACD (12 vs 26 días), Señal (9 días) e Histograma | $\frac{\text{EMA}_{12} - \text{EMA}_{26}}{P_{t-1}}$ | Dirección, cruces de tendencia y aceleración del impulso alcista o bajista del dólar. |
| **Calendario Cíclico** | `month_sin`, `month_cos`, `day_of_week_sin`, `day_of_week_cos` | Codificación trigonométrica de mes y día de la semana | $\sin,\cos(2\pi \cdot m/12)$ y $\sin,\cos(2\pi \cdot d/7)$ | Capturan estacionalidad anual (impuestos en marzo/abril, gratificaciones en julio/diciembre) y liquidez intrasemanal. |

---

### Tabla 4.1B — Las 32 Nuevas Variables Exógenas Globales, Commodities y Macroeconómicas (Fase 2)

> Todas las variables exógenas parten de la cotización de cierre del día anterior $X_{t-1} = X.\text{shift}(1)$ para garantizar que la información esté 100% disponible antes de emitir el pronóstico en $t$.

| Bloque Económico | Códigos de Variables en Tablas / Gráficos | Nombre Descriptivo | Fórmula Causal (`shift(1)`) | Mecanismo de Transmisión hacia el Tipo de Cambio USD/PEN |
|---|---|---|---|---|
| **Cobre Internacional** | `Cobre_ret_1d`, `Cobre_ret_5d`, `Cobre_ret_20d`, `Cobre_vs_SMA_20` | Retornos acumulados del Cobre a 1, 5 y 20 sesiones y distancia a su SMA de 20 días | $\ln\left(\frac{\text{Cu}_{t-1}}{\text{Cu}_{t-1-k}}\right)$ y $\frac{\text{Cu}_{t-1}-\text{SMA}_{20}}{\text{SMA}_{20}}$ | Principal producto de exportación tradicional del Perú: un alza sostenida del cobre incrementa el ingreso de divisas y aprecia el sol (reduce el USD/PEN). |
| **Índice de Volatilidad VIX** | `VIX_nivel_lag1`, `VIX_cambio_1d`, `VIX_cambio_5d`, `VIX_cambio_20d`, `VIX_zscore_20d` | Nivel del VIX en $t-1$, variaciones a 1, 5 y 20 días y *Z-Score* móvil de 20 días | $\text{VIX}_{t-1} - \text{VIX}_{t-1-k}$ y $\frac{\text{VIX}_{t-1}-\mu_{20}}{\sigma_{20}}$ | Termómetro de aversión global al riesgo (*fear gauge*): saltos en el VIX disparan refugio hacia el dólar (*flight to safety*), presionando al alza el USD/PEN. |
| **Índice Dólar Global (DXY)** | `DXY_ret_1d`, `DXY_ret_5d`, `DXY_ret_20d`, `DXY_vs_SMA_20` | Retornos del índice multilateral del dólar DXY a 1, 5 y 20 sesiones y brecha vs. SMA 20 | $\ln\left(\frac{\text{DXY}_{t-1}}{\text{DXY}_{t-1-k}}\right)$ | Mide la fortaleza global del dólar estadounidense frente a una canasta de monedas desarrolladas; se transmite directamente a las monedas emergentes. |
| **Bono del Tesoro EE. UU. 10 Años (`US10Y`)** | `US10Y_nivel_lag1`, `US10Y_cambio_5d`, `US10Y_cambio_20d`, `US10Y_vs_SMA_20` | Rendimiento del Treasury a 10 años en $t-1$ (%), cambios a 5 y 20 días y brecha vs. SMA 20 | $i^{\text{US10Y}}_{t-1} - i^{\text{US10Y}}_{t-1-k}$ | Refleja las condiciones monetarias de la Reserva Federal y el costo de oportunidad internacional del capital; afecta los flujos de portafolio hacia el Perú. |
| **Oro Internacional** | `Oro_ret_5d`, `Oro_ret_20d`, `Oro_vs_SMA_20` | Retornos del Oro a 5 y 20 sesiones y distancia a su SMA de 20 días | $\ln\left(\frac{\text{Au}_{t-1}}{\text{Au}_{t-1-k}}\right)$ | Segundo metal de exportación peruana y activo refugio mundial ante riesgos inflacionarios y geopolíticos. |
| **Petróleo WTI** | `Petroleo_WTI_ret_5d`, `Petroleo_WTI_ret_20d`, `Petroleo_WTI_vs_SMA_20` | Retornos del crudo WTI a 5 y 20 sesiones y distancia a su SMA de 20 días | $\ln\left(\frac{\text{WTI}_{t-1}}{\text{WTI}_{t-1-k}}\right)$ | El Perú es importador neto de hidrocarburos: subidas del WTI encarecen la factura de importación y demandan más dólares. |
| **Renta Variable Global (S&P 500)** | `SP500_ret_5d`, `SP500_ret_20d`, `SP500_vs_SMA_20` | Retornos del índice S&P 500 a 5 y 20 sesiones y brecha respecto a su SMA de 20 días | $\ln\left(\frac{\text{SP500}_{t-1}}{\text{SP500}_{t-1-k}}\right)$ | Captura el apetito global por activos de riesgo (*risk-on* vs. *risk-off*) en horizontes mensuales. |
| **Términos de Intercambio y Ratios Relativos** | `TermIntercambio_Cu_WTI_ret_20d`, `Cobre_vs_DXY_ret_20d` | Variación a 20 días del ratio Cobre/Petróleo WTI y del ratio Cobre/DXY | $\ln\left(\frac{\text{Cu}_{t-1}/\text{WTI}_{t-1}}{\text{Cu}_{t-21}/\text{WTI}_{t-21}}\right)$ | Proxy diario de los términos de intercambio del Perú (precio de exportación minera frente a costo de importación energética). |
| **Inflación BCRP con Rezago de Publicación Segura** | `Inflacion_12M_pub_segura`, `Inflacion_12M_cambio_20d` | Inflación interanual de Lima rezagada 42 días hábiles (~2 meses) y su cambio a 20 sesiones | $\pi^{12M}_{t-42}$ | Garantiza que el dato mensual del INEI/BCRP ya era público en la fecha de origen $t$, capturando el régimen inflacionario doméstico sin *data leakage*. |
| **Momentum Multiescala USD/PEN** | `PEN_ret_acum_5d`, `PEN_ret_acum_20d` | Retorno acumulado del propio USD/PEN en las últimas 5 y 20 sesiones hasta $t-1$ | $\ln(P_{t-1} / P_{t-1-k})$ | Sintetiza la tendencia acumulada semanal y mensual del tipo de cambio antes de la fecha de origen. |

---

### Tabla 4.2 — Explicación de las Columnas y Métricas de las Tablas de Resultados

| Nombre Descriptivo en Tablas | Unidad | Qué mide y cómo interpretarlo |
|---|---|---|
| **Horizonte de Pronóstico ($h$)** | Días hábiles | Plazo hacia adelante de la predicción: **1 día** (diario), **5 días** (1 semana bursátil) y **20 días** (~1 mes bursátil). |
| **Pronósticos Evaluados ($N$)** | Observaciones | Pares $(t, t+h)$ evaluables dentro de los 60 días reservados del *holdout*: $N = 60 - h$ (**59**, **55** y **40** pronósticos). |
| **MAE Caminata Aleatoria (Soles/USD)** | **Soles por USD (S/)** | $\frac{1}{N}\sum |P_{t+h} - P_t|$. Error promedio del benchmark que asume que el tipo de cambio no cambiará entre $t$ y $t+h$. |
| **MAE Sin Exógenas (Soles/USD)** | **Soles por USD (S/)** | Error Absoluto Medio del modelo usando únicamente las 18 variables técnicas y de calendario (Fase 1). |
| **MAE Con Exógenas Cobre/VIX/DXY (Soles/USD)** | **Soles por USD (S/)** | Error Absoluto Medio del nuevo modelo multivariable usando las 50 variables que incluyen Cobre, VIX, DXY, Oro, WTI y US10Y (Fase 2). |
| **Reducción de Error MAE al Incluir Exógenas** | **Soles por USD (S/)** | $\text{MAE}_{\text{Con Exógenas}} - \text{MAE}_{\text{Sin Exógenas}}$. Valores negativos confirman que incluir Cobre, VIX y DXY redujo el error de pronóstico. |
| **Diferencia vs. Caminata Aleatoria** | **Soles por USD (S/)** | $\text{MAE}_{\text{Con Exógenas}} - \text{MAE}_{\text{Caminata Aleatoria}}$. Valores negativos indican que **el modelo supera a la Caminata Aleatoria**. |
| **Ratio U de Theil** | Adimensional | $\text{RMSE}_{\text{Modelo}} / \text{RMSE}_{\text{Caminata Aleatoria}}$. Un valor $< 1.0$ indica menor error cuadrático medio que la caminata aleatoria. |
| **Acierto Direccional (% y Sesiones)** | Porcentaje (`%`) | Proporción y conteo exacto de sesiones con movimiento no nulo en que el modelo acertó si el dólar subiría o bajaría. |

---

## 4.3 Comparación Central: Fase 1 (Sin Exógenas, 18 Vars) vs. Fase 2 (Con Exógenas: Cobre, VIX, DXY, Oro, WTI, US10Y — 50 Vars)

*(Exportado en [`resultados_comparativos/tabla_comparativa_sin_vs_con_exogenas.csv`](../resultados_comparativos/tabla_comparativa_sin_vs_con_exogenas.csv) y graficado en [`resultados_comparativos/impacto_exogenas_h20.png`](../resultados_comparativos/impacto_exogenas_h20.png))*

### Tabla 4.3 — Impacto de Incorporar Cobre, VIX, DXY, Oro, Petróleo WTI y Bono EE. UU. 10Y en los Cuatro Ensambles

| Horizonte ($h$) | Algoritmo de Ensamble | MAE Caminata Aleatoria (Referencia) | MAE Fase 1: Sin Exógenas (18 Vars) | MAE Fase 2: Con Exógenas Cobre/VIX/DXY (50 Vars) | Reducción de MAE al Incluir Exógenas | ¿Supera a la Caminata Aleatoria en Fase 2? | Acierto Direccional Fase 1 (Sin Exógenas) | Acierto Direccional Fase 2 (Con Exógenas) | Mejora en Acierto Direccional | Peso de Variables Exógenas en el Modelo |
|:---:|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|
| **1 día** | **LightGBM** | 0.006881 S/ | 0.006974 S/ | **0.006850 S/** | **-0.000124 S/** | **SÍ** (`-0.000031 S/`) | 46.30 % *(25/54)* | 46.30 % *(25/54)* | 0.00 pp | 66.8 % |
| **1 día** | **CatBoost** | 0.006881 S/ | 0.006955 S/ | **0.006896 S/** | **-0.000059 S/** | Casi Empate (`+0.000015 S/`) | 46.30 % *(25/54)* | **55.56 % *(30/54)*** | **+9.26 pp** | 44.6 % |
| **1 día** | **XGBoost** | 0.006881 S/ | 0.006971 S/ | **0.006900 S/** | **-0.000070 S/** | Casi Empate (`+0.000019 S/`) | 42.59 % *(23/54)* | **51.85 % *(28/54)*** | **+9.26 pp** | 66.0 % |
| **1 día** | **Random Forest** | 0.006881 S/ | 0.006925 S/ | **0.006905 S/** | **-0.000020 S/** | Casi Empate (`+0.000023 S/`) | 50.00 % *(27/54)* | 48.15 % *(26/54)* | -1.85 pp | 40.8 % |
| **5 días** | **CatBoost** | 0.017458 S/ | 0.017758 S/ | **0.017262 S/** | **-0.000496 S/** | **SÍ** (`-0.000197 S/`) | 50.00 % *(27/54)* | **64.81 % *(35/54)*** | **+14.81 pp** | 61.7 % |
| **5 días** | **XGBoost** | 0.017458 S/ | 0.017602 S/ | **0.017331 S/** | **-0.000271 S/** | **SÍ** (`-0.000127 S/`) | 50.00 % *(27/54)* | **61.11 % *(33/54)*** | **+11.11 pp** | 64.5 % |
| **5 días** | **LightGBM** | 0.017458 S/ | 0.017640 S/ | **0.017339 S/** | **-0.000302 S/** | **SÍ** (`-0.000119 S/`) | 40.74 % *(22/54)* | **57.41 % *(31/54)*** | **+16.67 pp** | 52.9 % |
| **5 días** | **Random Forest** | 0.017458 S/ | 0.017576 S/ | **0.017415 S/** | **-0.000161 S/** | **SÍ** (`-0.000043 S/`) | 48.15 % *(26/54)* | **55.56 % *(30/54)*** | **+7.41 pp** | 44.9 % |
| **20 días** | **Random Forest** | 0.035905 S/ | 0.038959 S/ | **0.035302 S/** | **-0.003658 S/** | **SÍ** (`-0.000603 S/`) | 12.50 % *(5/40)* | **60.00 % *(24/40)*** | **+47.50 pp** | 55.0 % |
| **20 días** | **XGBoost** | 0.035905 S/ | 0.039115 S/ | **0.035647 S/** | **-0.003468 S/** | **SÍ** (`-0.000258 S/`) | 5.00 % *(2/40)* | **52.50 % *(21/40)*** | **+47.50 pp** | 64.8 % |
| **20 días** | **CatBoost** | 0.035905 S/ | 0.041446 S/ | **0.035868 S/** | **-0.005578 S/** | **SÍ** (`-0.000037 S/`) | 10.00 % *(4/40)* | **60.00 % *(24/40)*** | **+50.00 pp** | 60.1 % |
| **20 días** | **LightGBM** | 0.035905 S/ | 0.039042 S/ | **0.036428 S/** | **-0.002613 S/** | Casi Empate (`+0.000523 S/`) | 2.50 % *(1/40)* | **50.00 % *(20/40)*** | **+47.50 pp** | 68.2 % |

---

## 4.4 Tablas con los Valores Exactos de los Gráficos de Barras

### 4.4.1 Valores Exactos del Nuevo Gráfico de Barras Multivariable con Exógenas ([`resumen_mae_y_acierto_direccional_exogenas.png`](../resultados_comparativos/resumen_mae_y_acierto_direccional_exogenas.png))

*(Exportado en [`resultados_comparativos/tabla_valores_exactos_grafico_barras_exogenas.csv`](../resultados_comparativos/tabla_valores_exactos_grafico_barras_exogenas.csv))*

#### Panel A (Con Exógenas) — Error Absoluto Medio Exacto (MAE en Soles por USD y en Puntos Básicos)

| Horizonte ($h$) | Pronósticos Evaluados ($N$) | Caminata Aleatoria (Referencia) | Random Forest + Exógenas | LightGBM + Exógenas | XGBoost + Exógenas | CatBoost + Exógenas | Mejor Modelo en el Horizonte |
|:---:|---:|---:|---:|---:|---:|---:|---|
| **$h = 1$ día hábil** *(Soles/USD)* | 59 | 0.006881 S/ | 0.006905 S/ | **0.006850 S/** | 0.006900 S/ | 0.006896 S/ | **LightGBM + Exógenas** *(supera a RW)* |
| **$h = 1$ día hábil** *(Retorno pb)* | 59 | 20.34 pb | 20.41 pb | **20.25 pb** | 20.40 pb | 20.39 pb | **LightGBM + Exógenas** (`-0.09 pb` vs. RW) |
| **$h = 5$ días hábiles** *(Soles/USD)* | 55 | 0.017458 S/ | **0.017415 S/** | **0.017339 S/** | **0.017331 S/** | **0.017262 S/** | **CatBoost + Exógenas** *(los 4 superan a RW)* |
| **$h = 5$ días hábiles** *(Retorno pb)* | 55 | 51.63 pb | **51.51 pb** | **51.28 pb** | **51.26 pb** | **51.05 pb** | **CatBoost + Exógenas** (`-0.58 pb` vs. RW) |
| **$h = 20$ días hábiles** *(Soles/USD)* | 40 | 0.035905 S/ | **0.035302 S/** | 0.036428 S/ | **0.035647 S/** | **0.035868 S/** | **Random Forest + Exógenas** *(3 superan a RW)* |
| **$h = 20$ días hábiles** *(Retorno pb)* | 40 | 106.20 pb | **104.52 pb** | 107.89 pb | **105.53 pb** | **106.20 pb** | **Random Forest + Exógenas** (`-1.68 pb` vs. RW) |

#### Panel B (Con Exógenas) — Acierto Direccional Exacto (% y Conteo de Sesiones Acertadas)

| Horizonte ($h$) | Sesiones con Variación Real ($\Delta P \neq 0$) | Umbral Azar | Random Forest + Exógenas | LightGBM + Exógenas | XGBoost + Exógenas | CatBoost + Exógenas | Mayor Acierto Direccional |
|:---:|---:|---:|---:|---:|---:|---:|---|
| **$h = 1$ día hábil** | 54 de 59 sesiones | 50.00 % | 48.15 % *(26 / 54 días)* | 46.30 % *(25 / 54 días)* | **51.85 %** *(28 / 54 días)* | **55.56 %** *(30 / 54 días)* | **CatBoost + Exógenas** (`30 de 54` días) |
| **$h = 5$ días hábiles** | 54 de 55 sesiones | 50.00 % | **55.56 %** *(30 / 54 días)* | **57.41 %** *(31 / 54 días)* | **61.11 %** *(33 / 54 días)* | **64.81 %** *(35 / 54 días)* | **CatBoost + Exógenas** (`35 de 54` días) |
| **$h = 20$ días hábiles** | 40 de 40 sesiones | 50.00 % | **60.00 %** *(24 / 40 días)* | **50.00 %** *(20 / 40 días)* | **52.50 %** *(21 / 40 días)* | **60.00 %** *(24 / 40 días)* | **Random Forest y CatBoost** (`24 de 40` días) |

---

### 4.4.2 Valores Exactos del Gráfico de Barras de Línea Base Fase 1 Sin Exógenas ([`resumen_mae_y_acierto_direccional.png`](../resultados_comparativos/resumen_mae_y_acierto_direccional.png))

*(Exportado en [`resultados_comparativos/tabla_valores_exactos_grafico_barras.csv`](../resultados_comparativos/tabla_valores_exactos_grafico_barras.csv))*

| Horizonte ($h$) | Pronósticos ($N$) | MAE Caminata Aleatoria | MAE Random Forest (18 Vars) | MAE CatBoost (18 Vars) | MAE XGBoost (18 Vars) | MAE LightGBM (18 Vars) | Acierto RF (18 Vars) | Acierto CatBoost (18 Vars) | Acierto XGBoost (18 Vars) | Acierto LightGBM (18 Vars) |
|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1 día** | 59 | 0.006881 S/ | 0.006925 S/ | 0.006955 S/ | 0.006971 S/ | 0.006974 S/ | 50.00 % *(27/54)* | 46.30 % *(25/54)* | 42.59 % *(23/54)* | 46.30 % *(25/54)* |
| **5 días** | 55 | 0.017458 S/ | 0.017576 S/ | 0.017758 S/ | 0.017602 S/ | 0.017640 S/ | 48.15 % *(26/54)* | 50.00 % *(27/54)* | 50.00 % *(27/54)* | 40.74 % *(22/54)* |
| **20 días** | 40 | 0.035905 S/ | 0.038959 S/ | 0.041446 S/ | 0.039115 S/ | 0.039042 S/ | 12.50 % *(5/40)* | 10.00 % *(4/40)* | 5.00 % *(2/40)* | 2.50 % *(1/40)* |

---

## 4.5 Análisis del Peso de las Variables Exógenas por Horizonte

*(Exportado en [`resultados_comparativos/resumen_importancia_exogenas.csv`](../resultados_comparativos/resumen_importancia_exogenas.csv))*

| Horizonte ($h$) | Algoritmo de Ensamble | Peso Total de Variables Exógenas (%) | Peso Total de Variables Endógenas (%) | Top 5 Variables Predictoras Más Importantes en el Modelo |
|:---:|---|---:|---:|---|
| **1 día** | **Random Forest** | 40.77 % | 59.23 % | `price_vs_SMA_20`, `MACD_histogram`, `Oro_ret_20d`, `price_vs_SMA_10`, `MACD_signal` |
| **1 día** | **XGBoost** | 66.05 % | 33.95 % | `Petroleo_WTI_vs_SMA_20`, `Petroleo_WTI_ret_20d`, `Inflacion_12M_cambio_20d`, `DXY_vs_SMA_20`, `Oro_ret_5d` |
| **1 día** | **LightGBM** | 66.81 % | 33.19 % | `Petroleo_WTI_ret_20d`, `VIX_zscore_20d`, `US10Y_cambio_20d`, `PEN_log_ret_lag_10`, `Oro_ret_5d` |
| **1 día** | **CatBoost** | 44.59 % | 55.41 % | `price_vs_SMA_20`, `day_of_week_cos`, `day_of_week_sin`, `VIX_cambio_1d`, `PEN_log_ret_lag_1` |
| **5 días** | **Random Forest** | 44.95 % | 55.05 % | `MACD_histogram`, `MACD_signal`, `GARCH_vol`, `Inflacion_12M_cambio_20d`, `price_vs_SMA_10` |
| **5 días** | **XGBoost** | 64.46 % | 35.54 % | `Petroleo_WTI_ret_5d`, `Inflacion_12M_pub_segura`, `GARCH_vol`, `SP500_vs_SMA_20`, `US10Y_nivel_lag1` |
| **5 días** | **LightGBM** | 52.90 % | 47.10 % | `MACD_signal`, `GARCH_vol`, `US10Y_nivel_lag1`, `MACD_histogram`, `Inflacion_12M_cambio_20d` |
| **5 días** | **CatBoost** | 61.74 % | 38.26 % | `Inflacion_12M_pub_segura`, `GARCH_vol`, `price_vs_SMA_20`, `US10Y_nivel_lag1`, `Petroleo_WTI_ret_20d` |
| **20 días** | **Random Forest** | **54.97 %** | 45.03 % | `GARCH_vol`, `US10Y_nivel_lag1`, `MACD_line`, `Inflacion_12M_pub_segura`, `SP500_ret_20d` |
| **20 días** | **XGBoost** | **64.77 %** | 35.23 % | `US10Y_nivel_lag1`, `SP500_ret_20d`, `month_cos`, `SP500_vs_SMA_20`, `Oro_ret_20d` |
| **20 días** | **LightGBM** | **68.18 %** | 31.82 % | `SP500_ret_20d`, `US10Y_nivel_lag1`, `GARCH_vol`, `Oro_ret_20d`, `Cobre_vs_SMA_20` |
| **20 días** | **CatBoost** | **60.09 %** | 39.91 % | `US10Y_nivel_lag1`, `Inflacion_12M_pub_segura`, `GARCH_vol`, `Inflacion_12M_cambio_20d`, `month_cos` |

---

## 4.6 Conclusiones Principales

1. **Confirmación empírica de la hipótesis central de investigación:** Al incorporar las variables exógenas de condiciones financieras globales y *commodities* (**Cobre, VIX, DXY, Oro, Petróleo WTI, Bono del Tesoro de EE. UU. a 10 años, S&P 500 e Inflación BCRP**), los modelos de ensamble logran **reducir el Error Absoluto Medio (MAE) por debajo de la Caminata Aleatoria en los tres horizontes**:
   - En **$h = 1$ día**: **LightGBM + Exógenas** (`0.006850 S/` vs. `0.006881 S/` de la caminata aleatoria).
   - En **$h = 5$ días**: **Los 4 modelos (CatBoost `0.017262 S/`, XGBoost `0.017331 S/`, LightGBM `0.017339 S/` y Random Forest `0.017415 S/`) superan a la caminata aleatoria (`0.017458 S/`)**, con aciertos direccionales de **55.56 % a 64.81 %**.
   - En **$h = 20$ días**: **Random Forest (`0.035302 S/`), XGBoost (`0.035647 S/`) y CatBoost (`0.035868 S/`) superan a la caminata aleatoria (`0.035905 S/`)**, reduciendo el MAE hasta en `-0.005578 S/` respecto a los modelos sin exógenas y elevando el acierto direccional de `2.5 %–12.5 %` hasta **`52.5 %–60.0 %` (`24 de 40` sesiones acertadas)**.
2. **Importancia estructural de los fundamentos a 20 días:** En el horizonte mensual ($h = 20$), las variables exógenas representan entre el **55.0 % y el 68.2 %** de la ganancia total de información de los árboles, lideradas por la tasa del bono estadounidense a 10 años (`US10Y_nivel_lag1`), el apetito global por riesgo (`SP500_ret_20d`, `VIX_zscore_20d`), los metales de exportación (`Oro_ret_20d`, `Cobre_vs_SMA_20`), el petróleo (`Petroleo_WTI_ret_20d`) y la inflación doméstica (`Inflacion_12M_pub_segura`).
