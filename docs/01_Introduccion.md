# 1. Introducción

## 1.1 Descripción del problema
Actualmente, la variación del tipo de cambio USD/PEN modifica el costo en soles de las importaciones, el valor de los ingresos por exportaciones y las obligaciones denominadas en dólares en el Perú. Para las empresas e instituciones con exposición a ambas monedas, anticipar la magnitud y el sentido de estos movimientos puede apoyar la planificación presupuestaria y la evaluación de coberturas.

## 1.2 Hipótesis y Pregunta a abordar
- **Pregunta:** ¿La incorporación de indicadores económicos y características temporales en modelos de aprendizaje supervisado mejora el pronóstico del tipo de cambio USD/PEN a 1, 5 y 20 días hábiles frente a una caminata aleatoria y una regresión regularizada?
- **Hipótesis:** Al menos, un modelo de ensamble con variables económicas y características temporales puede reducir el error absoluto medio fuera de muestra frente a ambos modelos de referencia en alguno de los horizontes.

## 1.3 Objetivos
- Diseñar, implementar y evaluar un sistema predictivo multivariable del tipo de cambio USD/PEN.
- Estructurar un conjunto de datos trazable.
- Comparar Ridge, Lasso, Random Forest, XGBoost, LightGBM y CatBoost.
