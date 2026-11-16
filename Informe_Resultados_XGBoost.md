# Propuesta de redacción para las Secciones 4, 5 y 6 del Informe Final

## 4. Experimentación y Resultados

### 4.1 Configuración experimental y selección de variables con Lasso
Se construyó una matriz preprocesada de **8 522 sesiones hábiles** (`1994-02-02` a `2026-10-01`) con **97 características predictoras** (> 80 características, Sección 3.3), agrupadas en tres bloques: (1) **28 variables propias del USD/PEN** (rezagos diarios $k=0\dots 4$, rezagos a 10 y 21 días, retornos acumulados y volatilidades móviles a 5, 20 y 60 días, distancias a medias móviles, RSI, MACD y días de la semana en *one-hot*); (2) **55 variables financieras globales y de materias primas** con rezago $t-1$ (variaciones y brechas a medias móviles de **Cobre, Oro, Petróleo WTI, DXY, Bono del Tesoro EE. UU. a 10 años, S&P 500 y VIX**, más el indicador de tensión $s_t = \mathbf{1}[\text{VIX}_{t-1} > P_{80} = 24,52]$ y sus interacciones); y (3) **14 variables macroeconómicas del BCRP** asignadas por su fecha de publicación (variaciones del IPC General, Subyacente, Sin Alimentos y Energía, Transables, No Transables e inflación mensual y a 12 meses).

En cada ventana de entrenamiento, los predictores fueron winsorizados en los percentiles 1 y 99 y estandarizados. Para mitigar la maldición de la dimensionalidad, la regresión **Lasso** seleccionó **35 variables** en $h=1$ (reducción de 63,9 %) y **38 variables** en $h=5$ y $h=20$ (reducción de 60,8 %), conservando representantes de los tres grupos económicos.

### 4.2 Comparación de modelos en el horizonte principal ($h = 5$ días hábiles) y de contraste ($h = 1$ y $h = 20$)

| Horizonte ($h$) | Modelo | Predictores | $EAM_h$ (Soles/USD) | $U_h = \frac{EAM(\text{Mod})}{EAM(CA)}$ | Error por USD 10,000 (S/) | $AD_h$ (% y Días) | Valor-$p$ DM | Valor-$p$ PT |
|:---:|---|---:|---:|---:|---:|---:|---:|---:|
| **1 día** | Caminata Aleatoria (CA) | 0 | 0,006881 | 1,0000 | S/ 68,81 | — | — | — |
| **1 día** | Regresión Ridge | 35 | 0,006944 | 1,0092 | S/ 69,44 | 51,85 % *(28/54)* | 0,7865 | 0,3927 |
| **1 día** | Regresión Lasso | 35 | 0,006951 | 1,0101 | S/ 69,51 | 51,85 % *(28/54)* | 0,6712 | 0,3927 |
| **1 día** | XGBoost + Búsqueda Aleatoria (Lasso) | 35 | 0,007008 | 1,0185 | S/ 70,08 | 46,30 % *(25/54)* | 0,7926 | 0,7069 |
| **1 día** | **XGBoost + PSO (97 Vars Completas)** | 97 | **0,006875** | **0,9991** | **S/ 68,75** | **55,56 % *(30/54)*** | 0,4781 | 0,2008 |
| **1 día** | XGBoost + PSO + Selección Lasso | 35 | 0,007128 | 1,0359 | S/ 71,28 | 51,85 % *(28/54)* | 0,9456 | 0,3927 |
| **5 días (Principal)** | Caminata Aleatoria (CA) | 0 | 0,017458 | 1,0000 | S/ 174,58 | — | — | — |
| **5 días (Principal)** | Regresión Ridge | 38 | 0,017852 | 1,0226 | S/ 178,52 | 44,44 % *(24/54)* | 0,7906 | 0,7929 |
| **5 días (Principal)** | Regresión Lasso | 38 | 0,018288 | 1,0475 | S/ 182,88 | 46,30 % *(25/54)* | 0,8615 | 0,7069 |
| **5 días (Principal)** | **XGBoost + Búsqueda Aleatoria (Lasso)** | 38 | **0,017312** | **0,9916** | **S/ 173,12** | **57,41 % *(31/54)*** | 0,3587 | 0,1382 |
| **5 días (Principal)** | XGBoost + PSO (97 Vars Completas) | 97 | 0,017468 | 1,0005 | S/ 174,68 | **57,41 % *(31/54)*** | 0,5202 | 0,1382 |
| **5 días (Principal)** | **XGBoost + PSO + Selección Lasso** | 38 | **0,017439** | **0,9989** | **S/ 174,39** | **57,41 % *(31/54)*** | 0,4575 | 0,1382 |
| **20 días** | Caminata Aleatoria (CA) | 0 | 0,035905 | 1,0000 | S/ 359,05 | — | — | — |
| **20 días** | **Regresión Ridge** | 38 | **0,034420** | **0,9586** | **S/ 344,20** | **62,50 % *(25/40)*** | 0,2670 | **0,0569** |
| **20 días** | **Regresión Lasso** | 38 | **0,034876** | **0,9713** | **S/ 348,76** | **55,00 % *(22/40)*** | 0,4547 | 0,2635 |
| **20 días** | XGBoost + Búsqueda Aleatoria (Lasso) | 38 | 0,037029 | 1,0313 | S/ 370,29 | 52,50 % *(21/40)* | 0,5701 | 0,3759 |
| **20 días** | XGBoost + PSO (97 Vars Completas) | 97 | 0,037398 | 1,0416 | S/ 373,98 | **60,00 % *(24/40)*** | 0,5728 | 0,1030 |
| **20 días** | **XGBoost + PSO + Selección Lasso** | 38 | **0,035925** | **1,0006** | **S/ 359,25** | **60,00 % *(24/40)*** | 0,5012 | 0,1030 |

### 4.3 Análisis de SHAP y Prueba de Exclusión por Grupos
- **Atribución SHAP:** En $h = 5$ y $h = 20$, los valores TreeSHAP muestran que las variables con mayor impacto en las decisiones de XGBoost son el nivel de la tasa del Tesoro de EE. UU. a 10 años (`US10Y_nivel_lag1`), las variaciones y brechas de tendencia del **Cobre (`Cobre_vs_SMA_60`, `st_x_Cobre_ret_5d`)**, el **S&P 500 (`SP500_ret_20d`, `SP500_vs_SMA_60`)**, la **volatilidad del sol (`PEN_vol_20d`)** y las variaciones de **inflación e IPC del BCRP (`IPC_General_var_63d_pub`, `st_x_Inflacion_12M_pub`)**.
- **Exclusión por grupos (Ablación):** Al reentrenar XGBoost + PSO a 20 días eliminando el grupo de **variables globales y materias primas (Cobre, VIX, DXY, WTI, US10Y)**, el error por cada USD 10 000 empeora en **+S/ 29,85** ($U_{20}$ sube de `1,0006` a `1,0837`) y el acierto direccional cae de **60,00 % a 35,00 %**. Al excluir las variables propias del USD/PEN o las macroeconómicas del BCRP, el error también aumenta en **+S/ 5,16** y **+S/ 4,18 por cada USD 10 000**, respectivamente, confirmando que los tres bloques aportan información complementaria.

---

## 5. Conclusiones

1. **Verificación sobre la hipótesis a 5 días hábiles:** En el horizonte semanal ($h = 5$), **XGBoost ajustado con PSO y selección de variables Lasso** obtuvo un error absoluto medio menor que el de la caminata aleatoria ($U_5 = 0,9989 < 1$, equivalente a un error de S/ 174,39 frente a S/ 174,58 por cada USD 10 000) y un acierto direccional de **$57,41\%$ (`31 de 54` sesiones)**, superando también a Ridge ($U_5 = 1,0226$) y Lasso ($U_5 = 1,0475$). No obstante, dado que los valores-$p$ de Diebold-Mariano ($p = 0,4575$) y Pesaran-Timmermann ($p = 0,1382$) no resultaron estadísticamente significativos al $5\%$ en el tramo evaluado, la mejora en magnitud de error debe considerarse marginal, siendo el principal aporte práctico su orientación direccional para decidir el momento de compra dentro de la semana.
2. **Aporte de la reducción de dimensionalidad con Lasso:** Filtrar el vector de 97 predictores a 38 variables mediante Lasso redujo el sobreajuste de XGBoost + PSO tanto en $h = 5$ ($U_5$ baja de `1,0005` a `0,9989`) como en $h = 20$ ($U_{20}$ baja de `1,0416` a `1,0006`, ahorrando S/ 14,73 de error por cada USD 10 000).
3. **Complementariedad entre modelos lineales y no lineales:** Mientras que XGBoost lidera en los horizontes de 1 y 5 días hábiles ($U_1 = 0,9991$ y $U_5 = 0,9916\text{–}0,9989$), en el horizonte mensual ($h = 20$) la **Regresión Ridge** con interacciones de tensión financiera alcanza el menor error ($U_{20} = 0,9586$, ahorro de S/ 14,85 por cada USD 10 000) y un acierto direccional de **$62,50\%$** ($p_{PT} = 0,0569$).

---

## 6. Sugerencias de Trabajos Futuros

1. **Evaluación de otros algoritmos de ensamble y modelos profundos:** Extender el esquema de validación progresiva y selección Lasso para comparar formalmente **Random Forest, LightGBM y CatBoost**, así como arquitecturas secuenciales (LSTM, Temporal Fusion Transformers) y modelos fundacionales de series de tiempo.
2. **Ampliación del periodo de validación progresiva:** Evaluar el reajuste anual de PSO a lo largo de múltiples años fuera de muestra (p. ej., 2020–2026) para incrementar la potencia estadística de las pruebas de Diebold-Mariano y Pesaran-Timmermann.
3. **Regla de decisión financiera con costos de transacción y puntos *forward*:** Integrar los diferenciales de tasas de interés implícitos en los contratos *forward* bancarios locales y el *spread* de compra-venta para simular el ahorro neto efectivo en tesorería empresarial.
