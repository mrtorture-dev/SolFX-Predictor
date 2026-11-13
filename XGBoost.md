# Guía de ejecución local — SolFX Modelos de Ensamble (XGBoost, LightGBM, Random Forest y CatBoost)

## 1. Qué contiene esta entrega

- `Entrenando_el_XGBoost.py`: script reproducible para XGBoost (`resultados_xgboost/`).
- `Entrenando_el_LightGBM.py`: script reproducible para LightGBM (`resultados_lightgbm/`).
- `Entrenando_el_RandomForest.py`: script reproducible para Random Forest (`resultados_random_forest/`).
- `Entrenando_el_CatBoost.py`: script reproducible para CatBoost (`resultados_catboost/`).
- `Comparar_Modelos_Ensamble.py`: consolidador de métricas, gráficos comparativos multi-modelo (`resultados_comparativos/`) y sincronización de artefactos hacia `models/`.
- `requirements.txt` (y `requisitos_xgboost.txt`): bibliotecas necesarias.
- `data/processed/usdpen_train.csv` y `data/processed/usdpen_holdout.csv`: datos preprocesados.
- `docs/04_Resultados_Ensambles.md` e `Informe_Resultados_XGBoost.md`: redacción académica de metodología y resultados.

## 2. Cómo abrirlo y ejecutarlo en Python (Windows + Visual Studio Code)

1. Abrir **Visual Studio Code** y elegir **Archivo > Abrir carpeta** seleccionando `SolFX-Predictor`.
2. Abrir **Terminal > Nuevo terminal** y ejecutar:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe Entrenando_el_XGBoost.py
.\.venv\Scripts\python.exe Entrenando_el_LightGBM.py
.\.venv\Scripts\python.exe Entrenando_el_RandomForest.py
.\.venv\Scripts\python.exe Entrenando_el_CatBoost.py
.\.venv\Scripts\python.exe Comparar_Modelos_Ensamble.py
```

## 3. Qué se verá al terminar

En cada carpeta `resultados_<modelo>/` se guardan:

- `metricas_comparativas.csv`: resumen de MAE, RMSE, Theil U, R² y acierto direccional.
- `predicciones_h1.csv`, `predicciones_h5.csv`, `predicciones_h20.csv`: cada pronóstico, fecha de origen y valor real posterior.
- `grafico_h1.png`, `grafico_h5.png`, `grafico_h20.png`: curvas reales y estimadas, más caminata aleatoria.
- `importancias_h*.png` / `.csv`: importancia relativa de los indicadores en cada modelo.
- `modelo_<modelo>_h*`: modelos entrenados para reutilización futura (también copiados en `models/`).
- `validacion_temporal.csv`: comparación de 3 configuraciones por horizonte en cortes cronológicos.
- `proyeccion_ilustrativa_ultima_fecha.csv`: predicciones calculadas desde el **1 de octubre de 2026**.

Además, en `resultados_comparativos/` se generan:
- `tabla_comparativa_ensambles.csv`: tabla consolidada de los 4 modelos en los 3 horizontes.
- `comparacion_ensambles_h1.png`, `comparacion_ensambles_h5.png`, `comparacion_ensambles_h20.png`: curvas superpuestas de los 4 modelos frente al valor real y la caminata aleatoria.
- `resumen_mae_y_acierto_direccional.png`: gráfico de barras comparativo de MAE y acierto direccional.
- `proyecciones_ilustrativas_consolidadas.csv`: proyecciones desde el último registro para los 4 algoritmos.

## 4. Diseño de evaluación y precauciones

- La etiqueta de entrenamiento es `ln(USD_PEN[t+h] / USD_PEN[t]) * 10000` (puntos básicos, pb).
- Para predecir desde el cierre de `t`, solo se usan indicadores precalculados con rezagos y calendario. Se excluyen datos contemporáneos del objetivo y los IPC de fecha de publicación no auditada.
- El último tramo de 60 sesiones queda reservado para probar los modelos; no interviene en el ajuste de parámetros.
- Para impedir que las etiquetas de entrenamiento alcancen el periodo de validación, se usa `TimeSeriesSplit` con `gap=h`.
- La evaluación es **rolling origin**, no un pronóstico hecho desde el inicio de los 60 días. Cada predicción usa la información conocida al terminar su fecha de origen.
