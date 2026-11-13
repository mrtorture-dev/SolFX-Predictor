# Propuesta de redacción del informe: XGBoost en la predicción USD/PEN

## 1. Implementación del método XGBoost

Se desarrolló un modelo de regresión supervisada con XGBoost para estimar el retorno logarítmico acumulado del tipo de cambio USD/PEN a 1, 5 y 20 días hábiles. Se utilizaron los conjuntos de datos ya preprocesados del repositorio SolFX-Predictor, correspondientes a 8 462 registros de entrenamiento y 60 registros reservados para evaluación. En esta primera implementación se seleccionaron 18 variables técnicas y de calendario, entre ellas retornos rezagados, distancias respecto de medias móviles, índice de fuerza relativa (RSI), volatilidad histórica, componentes MACD e indicadores cíclicos del calendario.

Para el horizonte h, se definió como variable objetivo el retorno futuro en puntos básicos: 10 000 × ln(P[t+h] / P[t]), donde P[t] es el número de soles por dólar al cierre de la fecha de origen. El modelo genera un pronóstico de ese retorno, que luego se transforma a tipo de cambio mediante P_estimado[t+h] = P[t] × exp(retorno_estimado_pb / 10 000).

Se compararon tres configuraciones de hiperparámetros mediante validación temporal expansiva de tres cortes, con ventanas de evaluación de 252 sesiones y separación (gap) de h observaciones para reducir la fuga temporal. Una vez elegida la configuración por el error absoluto medio de validación, se entrenó el modelo con la información de entrenamiento y se evaluó en el periodo reservado. Como referencia se consideró el pronóstico de no cambio (caminata aleatoria sin deriva), en el cual el valor pronosticado para t+h es igual a P[t]. El experimento no incluye aún optimización mediante enjambre de partículas (PSO) ni explicaciones con SHAP.

## 2. Resultados obtenidos (primera ejecución)

| Horizonte | Predicciones evaluadas | MAE XGBoost (soles/USD) | MAE caminata aleatoria (soles/USD) | Acierto direccional XGBoost |
|---|---:|---:|---:|---:|
| 1 día hábil | 59 | 0.00697 | 0.00688 | 42.6 % |
| 5 días hábiles | 55 | 0.01760 | 0.01746 | 50.0 % |
| 20 días hábiles | 40 | 0.03912 | 0.03591 | 5.0 % |

Los pronósticos se evaluaron con origen móvil: en cada fecha de origen se contó con la información histórica observada hasta ese momento. El error absoluto medio (MAE) refleja la diferencia media entre la cotización real y la estimada. El acierto direccional indica en qué proporción el modelo anticipó correctamente una subida o bajada del precio; se excluyeron los casos sin variación real.

Los valores indican que XGBoost produjo estimaciones en los tres horizontes, aunque no redujo el MAE frente a la caminata aleatoria en este experimento. La diferencia es pequeña en 1 y 5 días, y mayor a 20 días. Estos resultados no confirman la hipótesis de superioridad predictiva para el conjunto de variables y configuración utilizados. El tamaño del periodo de prueba, especialmente a 20 días, y el solapamiento entre pronósticos limitan las conclusiones generales.

## 3. Conclusiones

La implementación de XGBoost permitió construir y evaluar un procedimiento automatizado de predicción del tipo de cambio USD/PEN a uno, cinco y veinte días hábiles. A partir de los datos preprocesados se prepararon variables predictoras, se entrenaron modelos de regresión y se generaron pronósticos comparables con los valores observados. Por lo tanto, se logró comprobar la viabilidad técnica del enfoque de aprendizaje supervisado para este problema.

No obstante, los resultados iniciales mostraron que XGBoost no superó a la caminata aleatoria en términos del error absoluto medio en ninguno de los tres horizontes. Esto evidencia que la complejidad de un modelo de inteligencia artificial no garantiza una mayor precisión frente a alternativas sencillas, especialmente en una serie financiera que presenta variaciones difíciles de anticipar. En consecuencia, el modelo desarrollado debe considerarse un punto de partida experimental y no una herramienta validada para la toma de decisiones cambiarias.

## 4. Resultados esperados a futuro

Como etapa posterior, se espera ampliar la base predictora mediante la incorporación de indicadores financieros y macroeconómicos relevantes, como el índice del dólar (DXY), el índice de volatilidad VIX, los precios internacionales de materias primas y las tasas de interés. Para evitar información futura en el entrenamiento, deberán respetarse las fechas efectivas de publicación y disponibilidad de cada variable, especialmente en los indicadores mensuales del BCRP.

También se propone ampliar la búsqueda de hiperparámetros, aplicar el esquema completo de validación temporal walk-forward previsto en el proyecto, analizar la importancia de las variables mediante SHAP y evaluar el desempeño en nuevos periodos fuera de muestra. Se espera que estas mejoras permitan identificar si XGBoost puede ofrecer ventajas predictivas consistentes para alguno de los horizontes estudiados, sin asumir de antemano que se alcanzará una precisión superior.

Finalmente, el desarrollo podría convertirse en un prototipo que actualice los datos, genere pronósticos periódicos, visualice sus errores y señale la incertidumbre de las estimaciones. Para ello se requerirían nuevas pruebas de robustez y validación antes de recomendar su uso en decisiones económicas reales.

## 5. Limitaciones metodológicas para consignar si el informe las solicita

- Los 60 registros reservados abarcan del 10/07/2026 al 01/10/2026; los pronósticos evaluables dentro del bloque son 59, 55 y 40 para horizontes de 1, 5 y 20 días, respectivamente.
- El código contiene 18 características técnicas y estacionales; no se añadieron las variables exógenas globales previstas en el documento original.
- El archivo mensual de IPC se alinea por mes de referencia. Esa alineación y un retraso de un día no garantizan que el dato estuviera realmente publicado; por prudencia se excluyó de esta versión.
- Se emplearon tres configuraciones de hiperparámetros y una validación temporal simplificada, no la optimización PSO ni el esquema exhaustivo con paso de 21 sesiones del diseño metodológico.
- Los pronósticos de 5 y 20 días están solapados temporalmente, por lo que sus errores no son observaciones independientes.
- Este ejercicio es una evaluación histórica: los datos terminan el 01/10/2026 y deben actualizarse antes de obtener pronósticos vigentes.
