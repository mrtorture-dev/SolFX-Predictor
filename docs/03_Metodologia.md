# 3. Metodología

## 3.1 Formulación del problema y datos de entrada
Se predice el retorno logarítmico acumulado del USD/PEN para horizontes de 1, 5 y 20 días. Variables locales (BCRPData) y globales (DXY, VIX, etc.) son utilizadas.

## 3.2 Preprocesamiento e Imputación (Palma & Salazar, 2025)
Siguiendo las directrices del Capítulo IV (Secciones 4.1 y 4.2) y el Algoritmo 2 de Palma y Salazar (2025):
- **Variable Objetivo ($r_t$):** Retorno logarítmico continuo diario del tipo de cambio spot interbancario USD/PEN:
  $$r_t = \ln(P_t) - \ln(P_{t-1})$$
  Se confirma la estacionariedad de la serie mediante pruebas estadísticas de Augmented Dickey-Fuller (ADF) y Kwiatkowski-Phillips-Schmidt-Shin (KPSS).
- **Alineación de Calendario e Imputación de Faltantes:**
  - Se homogeniza la serie sobre el calendario hábil bursátil (`freq='B'`).
  - Para feriados locales no coincidentes con festividades internacionales (Nueva York, Londres), se aplica **Forward Fill / Last Observation Carried Forward (LOCF)**, preservando el último precio disponible sin introducir fuga temporal hacia el futuro (*lookahead bias*).
  - Series de frecuencias mixtas o publicación mensual (ej. subíndices de IPC de Lima) se proyectan continuamente hacia adelante con propagación causal (*ffill*).
- **Tratamiento de Inconsistencias y Valores Atípicos:**
  - Depuración de errores de celda propios de terminales Bloomberg (`#NAME?`).
  - Detección de valores atípicos mediante el rango intercuartílico ($IQR$) y escalamiento robusto (`RobustScaler`) basado en mediana y rango intercuartílico.

## 3.3 Ingeniería de Características con Control Anti-Leakage
Conforme al protocolo de la Tesis (Algoritmo 2):
1. **Regla de Desplazamiento Causal:** Toda covariable exógena se rezaga un período (`shift(1)`) antes de calcular cualquier transformación o indicador derivado.
2. **Rezagos Autorregresivos (AR):** Lags de la variable objetivo $r_{t-1}, r_{t-2}, r_{t-3}, r_{t-5}, r_{t-10}, r_{t-21}$.
3. **Indicadores Técnicos Causalmente Seguros:**
   - *RSI(14)* calculado sobre $P_{t-1}$ y normalizado en el rango $[-1, 1]$.
   - *MACD(12, 26, 9)* normalizado por el precio rezagado $P_{t-1}$.
   - *Volatilidad Histórica (proxy GARCH)*: desviación estándar móvil de 21 días bursátiles sobre $r_{t-1}$.
   - *Distancia a Medias Móviles Simples (SMA 5, 10, 20)*: $(P_{t-1} - \text{SMA}_w) / \text{SMA}_w$.
4. **Características Estacionales / Cíclicas:** Codificación periódica mediante funciones $\sin$ y $\cos$ para el mes (período 12) y día de la semana (período 7).

## 3.4 Partición Temporal Estricta
- **Holdout Final:** Las últimas 60 observaciones de mercado ($N_{holdout} = 60$) se reservan exclusivamente para la evaluación final fuera de muestra.
- **Entrenamiento y Validación:** El resto de la serie histórica se utiliza para entrenamiento y validación cruzada temporal expandente (*Walk-Forward Validation*, `STEP_SIZE = 21`, `MIN_TRAIN = 252`).

## 3.4 Modelos y optimización de hiperparámetros (PSO)
Comparación de Ridge, Lasso y caminata aleatoria con ensambles: Random Forest, XGBoost, LightGBM y CatBoost. Optimización realizada con Enjambre de Partículas (PSO).

## 3.5 Validación temporal, métricas e interpretabilidad
Validación de tipo expansiva (walk-forward). Métricas: Error Absoluto Medio (MAE), Theil's U, Acierto Direccional (AD). Interpretabilidad con valores SHAP.
