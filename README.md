# 📈 SolFX-Predictor
**Predicción del tipo de cambio USD/PEN mediante Machine Learning**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Ensembles-orange.svg)
![Status](https://img.shields.io/badge/Estado-En%20Desarrollo-green.svg)

SolFX-Predictor es un sistema de aprendizaje supervisado diseñado para pronosticar la cotización del dólar estadounidense en soles peruanos (USD/PEN) en múltiples horizontes temporales (1, 5 y 20 días hábiles).

---

## 🎯 Resumen y Propuesta de Valor
Las fluctuaciones actuales del tipo de cambio monetario dificultan la planificación financiera y la gestión del riesgo cambiario en el Perú. 
Para resolver esto, el proyecto propone una arquitectura analítica que combina:
*   **Fundamentos Económicos:** Tasas de interés, Reservas Internacionales, precios de materias primas (Cobre, Oro, WTI), e índices financieros globales (DXY, VIX).
*   **Machine Learning Avanzado:** Evaluación temporal expansiva (walk-forward) comparando regresiones regularizadas contra algoritmos del estado del arte (*Random Forest, XGBoost, LightGBM, CatBoost*).
*   **Interpretabilidad:** Uso de valores *SHAP* para entender el peso de cada variable predictora en la toma de decisiones.

## 📂 Arquitectura del Repositorio

`	ext
SolFX-Predictor/
├── data/               # Datasets: Crudos (Bloomberg, BCRPData) y Procesados
├── docs/               # Documentación de la investigación desglosada por capítulos
├── models/             # Artefactos y modelos entrenados listos para despliegue
├── notebooks/          # Jupyter Notebooks para Análisis Exploratorio de Datos (EDA)
├── src/                # Scripts fuente (procesamiento, entrenamiento y métricas)
└── README.md           # Panel principal de navegación
`

## 🛠️ Enfoque Metodológico
1.  **Ingeniería de Características:** Creación de variables técnicas (rezagos, distancias a medias móviles, indicadores de tensión financiera global).
2.  **Selección y Reducción de Dimensionalidad:** Aplicación de PCA por grupos correlacionados y selección automática a través de regresiones *Lasso*.
3.  **Optimización Estocástica:** Búsqueda y ajuste de hiperparámetros mediante la técnica de enjambre de partículas (*Particle Swarm Optimization - PSO*).

## 👥 Equipo de Investigación (Grupo 3)
*   **Fredy Alberto Polar Urviola** (20221300)
*   **Renzo Raúl Hipólito Medina** (20190213)
*   **Ignacio Alonso Colán Lamas** (20221193)
*   **Dany Denilson Medina Quinto** (20192730)

---
*Desarrollado como proyecto aplicativo en ciencia de datos e investigación macroeconómica.*
