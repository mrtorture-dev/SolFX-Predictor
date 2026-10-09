# 3. Metodología

## 3.1 Formulación del problema y datos de entrada
Se predice el retorno logarítmico acumulado del USD/PEN para horizontes de 1, 5 y 20 días. Variables locales (BCRPData) y globales (DXY, VIX, etc.) son utilizadas.

## 3.2 Preprocesamiento
Tratamiento de datos faltantes, atípicos, escalamiento y codificación de variables categóricas calculadas exclusivamente con la ventana de entrenamiento.

## 3.3 Características y reducción de dimensionalidad
Uso de rezagos, volatilidades móviles, y características de calendario. Reducción mediante PCA por grupos y selección con Lasso.

## 3.4 Modelos y optimización de hiperparámetros (PSO)
Comparación de Ridge, Lasso y caminata aleatoria con ensambles: Random Forest, XGBoost, LightGBM y CatBoost. Optimización realizada con Enjambre de Partículas (PSO).

## 3.5 Validación temporal, métricas e interpretabilidad
Validación de tipo expansiva (walk-forward). Métricas: Error Absoluto Medio (MAE), Theil's U, Acierto Direccional (AD). Interpretabilidad con valores SHAP.
