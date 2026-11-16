# Guía de ejecución local — SolFX XGBoost + PSO y Selección Lasso

## 1. Qué contiene esta entrega (Alineada al Informe Final)

- `Entrenando_el_XGBoost.py`: script completo de experimentación que ejecuta el preprocesamiento por ventana (winsorización P1–P99 y estandarización), la **selección de variables con Lasso** sobre las 97 características, los modelos de referencia (**Caminata Aleatoria, Ridge y Lasso**), el ajuste de **XGBoost con PSO vs. Búsqueda Aleatoria**, las pruebas de **Diebold-Mariano y Pesaran-Timmermann**, los **valores SHAP** y la **prueba de exclusión por grupos (ablación)**.
- `src/exogenous_loader.py`: integrador de las fuentes de la Tabla 1 (Bloomberg, BCRPData, Yahoo Finance y FRED) para construir los 97 predictores (> 80 características).
- `requirements.txt` y `requisitos_xgboost.txt`: bibliotecas necesarias.
- `data/processed/usdpen_train.csv`, `data/processed/usdpen_holdout.csv` y `data/processed/catalogo_84_predictores.json`: datos preprocesados y catálogo de los 3 grupos de variables.
- `resultados_xgboost/`: tablas descriptivas, valores exactos de los gráficos de barras, predicciones, importancias SHAP y estudio de ablación.
- `docs/04_Experimentacion_y_Resultados.md` e `Informe_Resultados_XGBoost.md`: redacción lista para las secciones 4, 5 y 6 del informe.

## 2. Cómo ejecutarlo en Python (Windows + Visual Studio Code)

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\exogenous_loader.py
.\.venv\Scripts\python.exe Entrenando_el_XGBoost.py
```

## 3. Archivos generados en `resultados_xgboost/`

- `tabla_resultados_descriptiva.csv` y `metricas_comparativas.csv`: métricas completas ($EAM_h$, $U_h$, error en soles por cada USD 10,000, $AD_h$, estadísticos y valores-$p$ de Diebold-Mariano y Pesaran-Timmermann).
- `tabla_seleccion_variables_lasso.csv`: conteo y variables seleccionadas por Lasso en cada horizonte ($h = 1, 5, 20$).
- `tabla_exclusion_grupos_ablacion.csv`: resultados de reentrenar XGBoost + PSO sin cada uno de los tres grupos de variables (Propias USD/PEN, Globales/Commodities y Macroeconómicas BCRP).
- `tabla_hiperparametros_pso_vs_random_search.csv` y `convergencia_pso_vs_rs_h*.csv`: hiperparámetros óptimos y curva de aptitud de PSO frente a Búsqueda Aleatoria.
- `importancias_shap_h1.png`, `importancias_shap_h5.png`, `importancias_shap_h20.png` (y `.csv`): atribución exacta TreeSHAP en puntos básicos.
- `grafico_h1.png`, `grafico_h5.png`, `grafico_h20.png`: curvas observadas y pronosticadas.
- `resumen_metricas_uh_y_ad.png`: gráfico de barras comparativo de $U_h$ y $AD_h$ con etiquetas numéricas exactas sobre cada barra.
