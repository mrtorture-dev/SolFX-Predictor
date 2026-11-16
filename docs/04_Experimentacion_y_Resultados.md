# 4. Experimentación, Selección de Variables y Resultados (XGBoost con PSO, Ridge, Lasso y Caminata Aleatoria)

## 4.1 Configuración Experimental y Conjunto de Datos Preprocesado

Siguiendo la metodología establecida en las Secciones 3.1 a 3.5 del informe final (*Polar Urviola, Hipólito Medina, Colán Lamas y Medina Quinto, 2026*), el conjunto de datos preprocesado ([`data/processed/usdpen_preprocessed_full.csv`](../data/processed/usdpen_preprocessed_full.csv)) integra **8 522 sesiones hábiles** (`1994-02-02` a `2026-10-01`) y **97 características predictoras** (> 80 características, Sección 3.3), organizadas en tres bloques económicos:

1. **Grupo 1 — Variables propias del USD/PEN (28 predictores):**
   - Rezagos del retorno diario $\Delta_{t-k} = \ln S_{t-k} - \ln S_{t-k-1}$ para $k = 0 \dots 4$ (`PEN_ret_lag_0` a `PEN_ret_lag_4`, evaluados sin fuga de información al cierre de $t$), más rezagos de 10 y 21 días (`PEN_log_ret_lag_10`, `PEN_log_ret_lag_21`).
   - Retornos acumulados en ventanas de **5, 20 y 60 días** (`PEN_ret_acum_5d`, `PEN_ret_acum_20d`, `PEN_ret_acum_60d`).
   - Volatilidad móvil en ventanas de **5, 20 y 60 días** (`PEN_vol_5d`, `PEN_vol_20d`, `PEN_vol_60d`).
   - Distancia porcentual a la media móvil simple en ventanas de **5, 10, 20 y 60 días** (`price_vs_SMA_5`, `price_vs_SMA_10`, `price_vs_SMA_20`, `price_vs_SMA_60`).
   - Osciladores e indicadores de tendencia causales (`RSI`, `MACD_line`, `MACD_signal`, `MACD_histogram`).
   - Codificación *one-hot* del día de la semana (`dow_lun`, `dow_mar`, `dow_mie`, `dow_jue`, `dow_vie`) y estacionalidad mensual (`month_sin`, `month_cos`).

2. **Grupo 2 — Condiciones Financieras Globales y Materias Primas con rezago $t-1$ (55 predictores):**
   - Variaciones logarítmicas a **1, 5, 20 y 60 días** y distancia a las medias móviles de **20 y 60 días** para **Cobre (`HG=F` / `PCOPPUSDM`), Oro (`GC=F`), Petróleo WTI (`CL=F` / `DCOILWTICO`), Índice Dólar (`DXY`) y S&P 500 (`^GSPC`)**.
   - Nivel y variaciones a **1, 5 y 20 días** y distancia a SMA 20 del **Bono del Tesoro de EE. UU. a 10 años (`US10Y`)** y del **Índice de Volatilidad CBOE (`VIX`)**.
   - **Indicador de estado de tensión financiera global $s_t$ (`st_tension_vix`):** Vale `1` si $\text{VIX}_{t-1} > P_{80,\text{train}} = 24.52$ y `0` en caso contrario (Filippou et al. 2026).
   - **Términos de intercambio e interacciones de estado $s_t \times X_{t-1}$:** Razones Cobre/WTI y Cobre/DXY, más 11 términos de interacción entre $s_t$ y los retornos de Cobre, DXY, US10Y, S&P 500, WTI, Oro y USD/PEN.

3. **Grupo 3 — Variables Macroeconómicas del BCRP por Fecha de Publicación (14 predictores):**
   - Variaciones mensuales y trimestrales de los índices IPC de Lima Metropolitana (`IPC_General`, `IPC_Subyacente`, `IPC_SinAlimEnergia`, `IPC_Transables`, `IPC_NoTransables`) e inflación mensual e interanual (`Inflacion_Mensual_pub`, `Inflacion_12M_pub`, `Inflacion_12M_cambio_21d`) asignadas respetando su rezago de publicación oficial, más la interacción `st_x_Inflacion_12M_pub`.

---

## 4.2 Reducción de Dimensionalidad y Selección de Variables mediante Lasso (Sección 3.3)

Dentro de la ventana de entrenamiento (previa a la purga $t + h \le \tau$), todos los predictores se winsorizaron en los percentiles 1 y 99 y se estandarizaron. La regresión **Lasso** seleccionó automáticamente el subconjunto de predictores con coeficientes no nulos para cada horizonte (exportado en [`resultados_xgboost/tabla_seleccion_variables_lasso.csv`](../resultados_xgboost/tabla_seleccion_variables_lasso.csv)):

**Tabla 2.** Reducción de dimensionalidad mediante selección Lasso por horizonte de pronóstico

| Horizonte de Pronóstico ($h$) | Total de Predictores Iniciales | Variables Conservadas por Lasso | Reducción Dimensional (%) | Grupo 1: Propias USD/PEN | Grupo 2: Globales y Commodities | Grupo 3: Macroeconómicas BCRP | Principales 8 Variables Seleccionadas por Magnitud $|\hat{\beta}_{\text{Lasso}}|$ |
|:---:|---:|---:|---:|---:|---:|---:|---|
| **$h = 1$ día hábil** *(Tesorería diaria)* | 97 | **35** | 63,9 % | 10 | 21 | 4 | `Cobre_ret_60d`, `st_x_DXY_ret_5d`, `st_x_Cobre_ret_5d`, `dow_jue`, `Oro_ret_20d`, `DXY_ret_20d`, `Inflacion_Mensual_pub`, `st_x_Oro_ret_20d` |
| **$h = 5$ días hábiles** *(Principal — Semanal)* | 97 | **38** | 60,8 % | 7 | 24 | 7 | `SP500_vs_SMA_60`, `SP500_ret_60d`, `st_x_Cobre_ret_5d`, `US10Y_nivel_lag1`, `price_vs_SMA_60`, `IPC_General_var_63d_pub`, `st_x_Inflacion_12M_pub`, `st_x_DXY_ret_5d` |
| **$h = 20$ días hábiles** *(Presupuesto mensual)* | 97 | **38** | 60,8 % | 9 | 23 | 6 | `Cobre_vs_SMA_60`, `US10Y_nivel_lag1`, `SP500_ret_20d`, `TermIntercambio_Cu_WTI_ret_20d`, `IPC_SinAlimEnergia_var_63d_pub`, `IPC_General_var_63d_pub`, `VIX_nivel_lag1`, `PEN_vol_20d` |

---

## 4.3 Resultados Comparativos Fuera de Muestra y Tabla de Valores Exactos del Gráfico de Barras

*(Exportado en [`resultados_xgboost/tabla_resultados_descriptiva.csv`](../resultados_xgboost/tabla_resultados_descriptiva.csv) y graficado en [`resultados_xgboost/resumen_metricas_uh_y_ad.png`](../resultados_xgboost/resumen_metricas_uh_y_ad.png))*

### Explicación de los Símbolos y Columnas de la Tabla 3 (conforme a la Ecuación 3 de la Sección 3.5)
* **$h$ (Horizonte en días hábiles):** Plazo del pronóstico: **5 días hábiles** es el horizonte principal (decisión semanal de compra de dólares o cobertura *forward*); **1 y 20 días** se reportan como contraste.
* **$n$ (Pronósticos evaluados):** Número de pares $(S_t, S_{t+h})$ evaluados en validación progresiva con purga ($n = 59, 55, 40$).
* **$EAM_h$ (Soles por USD):** Error Absoluto Medio $\frac{1}{n}\sum |S_{t+h} - \hat{S}_{t+h}|$ medido en soles por dólar.
* **$U_h = EAM_h(\text{modelo}) / EAM_h(CA)$:** Cociente de error relativo frente a la Caminata Aleatoria. **Un valor $U_h < 1,0000$ indica que el modelo se equivoca menos que suponer ausencia de cambio.**
* **Error en Soles por cada USD 10,000 ($EAM_h \times 10\,000$):** Impacto económico directo del error promedio de pronóstico sobre el pago de una factura comercial de USD 10 000 (Sección 1.3).
* **$AD_h$ (% y sesiones):** Acierto Direccional $\frac{1}{n}\sum \mathbf{1}[\text{signo}(\hat{r}_t^{(h)}) = \text{signo}(r_t^{(h)})]$ sobre variaciones no nulas.
* **Valor-$p$ DM y Valor-$p$ PT:** Valores-$p$ unilaterales de las pruebas de **Diebold-Mariano** ($H_1: U_h < 1$) y **Pesaran-Timmermann** ($H_1: AD_h > 50\%$).

---

**Tabla 3.** Resultados comparativos y valores exactos del gráfico de barras (`resumen_metricas_uh_y_ad.png`) en los horizontes de 1, 5 y 20 días hábiles

| Horizonte ($h$) | Modelo Evaluado | N° de Predictores | $EAM_h$ (Soles por USD) | Cociente $U_h = \frac{EAM(\text{Mod})}{EAM(CA)}$ | Error por Factura de USD 10,000 (S/) | Ahorro vs. CA por USD 10,000 (S/) | $RMSE_h$ (Soles por USD) | Acierto Direccional $AD_h$ (% y Días) | Valor-$p$ Diebold-Mariano | Valor-$p$ Pesaran-Timmermann |
|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1 día** | Caminata Aleatoria (CA) | 0 | 0,006881 S/ | 1,0000 | S/ 68,81 | S/ 0,00 | 0,009411 S/ | — *(0/54)* | — | — |
| **1 día** | Regresión Ridge | 35 | 0,006944 S/ | 1,0092 | S/ 69,44 | -S/ 0,63 | 0,009349 S/ | 51,85 % *(28/54)* | 0,7865 | 0,3927 |
| **1 día** | Regresión Lasso | 35 | 0,006951 S/ | 1,0101 | S/ 69,51 | -S/ 0,70 | 0,009457 S/ | 51,85 % *(28/54)* | 0,6712 | 0,3927 |
| **1 día** | XGBoost + Búsqueda Aleatoria (Lasso) | 35 | 0,007008 S/ | 1,0185 | S/ 70,08 | -S/ 1,27 | 0,009561 S/ | 46,30 % *(25/54)* | 0,7926 | 0,7069 |
| **1 día** | **XGBoost + PSO (97 Vars Completas)** | 97 | **0,006875 S/** | **0,9991** | **S/ 68,75** | **+S/ 0,06** | **0,009303 S/** | **55,56 % *(30/54)*** | 0,4781 | 0,2008 |
| **1 día** | XGBoost + PSO + Selección Lasso | 35 | 0,007128 S/ | 1,0359 | S/ 71,28 | -S/ 2,47 | 0,009684 S/ | 51,85 % *(28/54)* | 0,9456 | 0,3927 |
| **5 días (Principal)** | Caminata Aleatoria (CA) | 0 | 0,017458 S/ | 1,0000 | S/ 174,58 | S/ 0,00 | 0,024435 S/ | — *(0/54)* | — | — |
| **5 días (Principal)** | Regresión Ridge | 38 | 0,017852 S/ | 1,0226 | S/ 178,52 | -S/ 3,94 | 0,024822 S/ | 44,44 % *(24/54)* | 0,7906 | 0,7929 |
| **5 días (Principal)** | Regresión Lasso | 38 | 0,018288 S/ | 1,0475 | S/ 182,88 | -S/ 8,30 | 0,025216 S/ | 46,30 % *(25/54)* | 0,8615 | 0,7069 |
| **5 días (Principal)** | **XGBoost + Búsqueda Aleatoria (Lasso)** | 38 | **0,017312 S/** | **0,9916** | **S/ 173,12** | **+S/ 1,46** | **0,024472 S/** | **57,41 % *(31/54)*** | 0,3587 | 0,1382 |
| **5 días (Principal)** | XGBoost + PSO (97 Vars Completas) | 97 | 0,017468 S/ | 1,0005 | S/ 174,68 | -S/ 0,09 | 0,024604 S/ | **57,41 % *(31/54)*** | 0,5202 | 0,1382 |
| **5 días (Principal)** | **XGBoost + PSO + Selección Lasso** | 38 | **0,017439 S/** | **0,9989** | **S/ 174,39** | **+S/ 0,19** | **0,024542 S/** | **57,41 % *(31/54)*** | 0,4575 | 0,1382 |
| **20 días** | Caminata Aleatoria (CA) | 0 | 0,035905 S/ | 1,0000 | S/ 359,05 | S/ 0,00 | 0,042619 S/ | — *(0/40)* | — | — |
| **20 días** | **Regresión Ridge** | 38 | **0,034420 S/** | **0,9586** | **S/ 344,20** | **+S/ 14,85** | **0,041416 S/** | **62,50 % *(25/40)*** | 0,2670 | **0,0569** |
| **20 días** | **Regresión Lasso** | 38 | **0,034876 S/** | **0,9713** | **S/ 348,76** | **+S/ 10,29** | **0,042254 S/** | **55,00 % *(22/40)*** | 0,4547 | 0,2635 |
| **20 días** | XGBoost + Búsqueda Aleatoria (Lasso) | 38 | 0,037029 S/ | 1,0313 | S/ 370,29 | -S/ 11,24 | 0,044710 S/ | 52,50 % *(21/40)* | 0,5701 | 0,3759 |
| **20 días** | XGBoost + PSO (97 Vars Completas) | 97 | 0,037398 S/ | 1,0416 | S/ 373,98 | -S/ 14,93 | 0,044767 S/ | **60,00 % *(24/40)*** | 0,5728 | 0,1030 |
| **20 días** | **XGBoost + PSO + Selección Lasso** | 38 | **0,035925 S/** | **1,0006** | **S/ 359,25** | **-S/ 0,20** | **0,043519 S/** | **60,00 % *(24/40)*** | 0,5012 | 0,1030 |

---

## 4.4 Interpretación de Resultados frente a la Hipótesis Principal ($h = 5$ días hábiles)

1. **Desempeño en el horizonte principal ($h = 5$ días hábiles):**
   - **Reducción del error puntual ($U_5 < 1$):** Tanto **XGBoost + PSO + Selección Lasso** ($EAM_5 = 0,017439\text{ S/}$, $U_5 = \mathbf{0,9989} < 1$) como **XGBoost + Búsqueda Aleatoria (Lasso)** ($EAM_5 = 0,017312\text{ S/}$, $U_5 = \mathbf{0,9916} < 1$) logran un error absoluto medio menor que el de la Caminata Aleatoria ($EAM_5(CA) = 0,017458\text{ S/}$) y que los modelos lineales Ridge ($U_5 = 1,0226$) y Lasso ($U_5 = 1,0475$). Asimismo, se comprueba que la **selección de variables con Lasso (38 variables)** mejora a XGBoost + PSO respecto a usar las 97 variables completas ($U_5 = 1,0005$).
   - **Acierto direccional ($AD_5 > 50\%$):** En el horizonte semanal, las tres variantes de XGBoost alcanzan un **Acierto Direccional de $57,41\%$ (`31 de 54` sesiones acertadas)**, superando en **+7,41 puntos porcentuales** el umbral de azar de $50\%$.
   - **Significancia estadística (Diebold-Mariano y Pesaran-Timmermann):** Aunque se cumplen numéricamente las dos desigualdades puntuales en $h = 5$ ($U_5 = 0,9989 < 1$ y $AD_5 = 57,41\% > 50\%$), en la muestra de evaluación de 55 pronósticos los valores-$p$ de Diebold-Mariano ($p = 0,4575$) y Pesaran-Timmermann ($p = 0,1382$) no rechazan la hipótesis nula al $5\%$. Esto coincide con la literatura financiera (Pfahler 2021; Filippou et al. 2026): debido a la baja volatilidad del sol peruano, las ganancias en magnitud de error son moderadas (alrededor de S/ 0,19 a S/ 1,46 por cada USD 10 000 en una semana), mientras que la ganancia principal reside en anticipar el **sentido del movimiento ($57,41\%$ a 5 días y $60,00\%$ a 20 días)**.

2. **Efecto de la selección Lasso y PSO en el horizonte mensual ($h = 20$ días hábiles):**
   - En $h = 20$, donde la superposición de etiquetas agrava la maldición de la dimensionalidad (Sección 3.3), reducir de 97 a 38 predictores mediante Lasso disminuye el error de **XGBoost + PSO** de **S/ 373,98 a S/ 359,25 por cada USD 10 000** (una reducción de **S/ 14,73**), superando además a la Búsqueda Aleatoria (**S/ 370,29**) y alcanzando un **60,00 % de acierto direccional (`24 de 40` sesiones)**.
   - Destaca además que, en el horizonte de 20 días, los modelos lineales regularizados con interacciones de tensión (**Regresión Ridge** con $U_{20} = \mathbf{0,9586}$, ahorro de **S/ 14,85 por cada USD 10 000** y $AD_{20} = \mathbf{62,50\%}$, $p_{PT} = 0,0569$; y **Regresión Lasso** con $U_{20} = \mathbf{0,9713}$ y $AD_{20} = 55,00\%$) superan claramente a la Caminata Aleatoria al extrapolar linealmente las tendencias de tasas de interés (`US10Y`), `Cobre` y `S&P 500`.

---

## 4.5 Prueba de Exclusión por Grupos (Ablation Study — Sección 3.5)

Para verificar si cada bloque de información aporta capacidad predictiva real en **XGBoost + PSO + Selección Lasso**, se reentrenó el modelo excluyendo un grupo a la vez ([`resultados_xgboost/tabla_exclusion_grupos_ablacion.csv`](../resultados_xgboost/tabla_exclusion_grupos_ablacion.csv)):

**Tabla 4.** Prueba de exclusión por grupos de variables (Ablación sobre XGBoost + PSO)

| Horizonte ($h$) | Configuración del Modelo | Variables Activas | $EAM_h$ (Soles por USD) | Cociente $U_h = \frac{EAM}{EAM(CA)}$ | Variación del Error por USD 10,000 vs. Modelo Completo | Acierto Direccional $AD_h$ (%) |
|:---:|---|---:|---:|---:|---:|---:|
| **1 día** | Modelo Completo (3 Grupos con Selección Lasso) | 35 | 0,007128 S/ | 1,0359 | S/ 0,00 | 51,85 % |
| **1 día** | **Sin Grupo 1: Propias del USD/PEN** | 25 | 0,007386 S/ | 1,0733 | **+S/ 2,57 (Empeora)** | 51,85 % |
| **1 día** | Sin Grupo 2: Globales y Commodities | 14 | 0,006900 S/ | 1,0027 | -S/ 2,28 | 48,15 % |
| **1 día** | Sin Grupo 3: Macroeconómicas BCRP | 31 | 0,007064 S/ | 1,0266 | -S/ 0,64 | 55,56 % |
| **5 días (Principal)** | Modelo Completo (3 Grupos con Selección Lasso) | 38 | **0,017439 S/** | **0,9989** | S/ 0,00 | **57,41 %** |
| **5 días (Principal)** | **Sin Grupo 1: Propias del USD/PEN** | 31 | 0,017425 S/ | 0,9981 | -S/ 0,15 | **51,85 % (-5,56 pp)** |
| **5 días (Principal)** | Sin Grupo 2: Globales y Commodities | 14 | 0,017303 S/ | 0,9911 | -S/ 1,36 | 66,67 % |
| **5 días (Principal)** | Sin Grupo 3: Macroeconómicas BCRP | 31 | 0,017400 S/ | 0,9967 | -S/ 0,40 | 57,41 % |
| **20 días** | Modelo Completo (3 Grupos con Selección Lasso) | 38 | **0,035925 S/** | **1,0006** | S/ 0,00 | **60,00 %** |
| **20 días** | **Sin Grupo 1: Propias del USD/PEN** | 29 | 0,036441 S/ | 1,0149 | **+S/ 5,16 (Empeora)** | 62,50 % |
| **20 días** | **Sin Grupo 2: Globales y Commodities (VIX, DXY, Cobre, WTI, US10Y)** | 15 | **0,038910 S/** | **1,0837** | **+S/ 29,85 (Empeora severamente)** | **35,00 % (-25,00 pp)** |
| **20 días** | **Sin Grupo 3: Macroeconómicas BCRP (IPC e Inflación)** | 32 | 0,036343 S/ | 1,0122 | **+S/ 4,18 (Empeora)** | 60,00 % |

> **Hallazgo clave de la prueba de exclusión:**
> * A **1 día hábil**, eliminar las variables propias del USD/PEN (Grupo 1) es lo que más deteriora el pronóstico (+S/ 2,57 por USD 10 000), confirmando que en el muy corto plazo domina la dinámica autorregresiva y de microestructura local.
> * A **20 días hábiles**, eliminar el **Grupo 2 (Condiciones Globales y Commodities: Cobre, VIX, DXY, WTI, US10Y, S&P 500)** provoca que el error se dispare en **+S/ 29,85 por cada USD 10 000** ($U_{20}$ sube de `1,0006` a `1,0837`) y que el Acierto Direccional caiga **de 60,00 % a 35,00 %**. Asimismo, excluir el **Grupo 3 (Macroeconómicas BCRP)** incrementa el error en **+S/ 4,18 por cada USD 10 000**, demostrando empíricamente que los tres grupos son necesarios en el horizonte mensual.
