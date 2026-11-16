# 3. Metodología

## 3.1 Formulación y datos
Sea $S_t$ el tipo de cambio spot USD/PEN (soles por dólar) al cierre del día hábil, obtenido de Bloomberg con el código `PEN REGN Curncy` (Bloomberg L. P. 2026). Para cada horizonte $h \in \{1, 5, 20\}$ días hábiles se predice el retorno logarítmico:

$$r_t^{(h)} = \ln S_{t+h} - \ln S_t \quad (1)$$

El nivel se recupera como $\hat{S}_{t+h} = S_t \cdot \exp(\hat{r}_t^{(h)})$. Se predice el retorno porque el nivel no es estacionario y los árboles no extrapolan fuera del rango de entrenamiento. Se entrena un modelo independiente por horizonte, y la caminata aleatoria (CA) equivale a predecir un retorno nulo.

La **Tabla 1** resume qué información se obtiene de cada fuente. Para evitar fuga de información, las variables de Nueva York entran con rezago $t - 1$, porque ese mercado cierra después que Lima, y las series mensuales del BCRP se asignan por su fecha de publicación, como hacen Filippou y otros (2026) con los rezagos de publicación.

**Tabla 1.** Fuentes de datos y uso de cada una en el modelo

| Fuente | Información que se obtiene | Uso en el modelo |
| :--- | :--- | :--- |
| **Bloomberg** | Tipo de cambio spot USD/PEN al cierre (`PEN REGN Curncy`). Diaria, 1994-2026 | Variable objetivo; sus rezagos, medias y volatilidad móviles |
| **BCRPData** | IPC de Lima Metropolitana; inflación mensual y a 12 meses. Mensual, 1994-2026 | Predictores macroeconómicos, por fecha de publicación |
| **Yahoo Finance** | VIX, índice dólar (DXY), Tesoro a 10 años, S&P 500 y futuros de cobre, oro y petróleo. Diaria, 1994-2026 | Predictores globales, con rezago $t - 1$ |
| **FRED** | Cobre (mensual `PCOPPUSDM`) y petróleo WTI (diaria `DCOILWTICO`) | Completar cobre y petróleo antes del 2000 |

## 3.2 Preprocesamiento y características
Todos los parámetros se calculan solo con la ventana de entrenamiento. Los faltantes de calendario se completan con el último valor publicado (máximo 5 días), sin interpolar. Los retornos atípicos (puntaje robusto basado en la mediana mayor que 3,5) se conservan porque son eventos reales, pero los predictores se winsorizan en los percentiles 1 y 99; la variable objetivo no se modifica. Los predictores se estandarizan para los modelos lineales y el día de la semana se codifica con *one-hot*.

Con el retorno diario $\Delta_t = \ln S_t - \ln S_{t-1}$ se construyen rezagos ($k = 0 \dots 4$), retornos acumulados, volatilidad móvil y distancia a la media móvil en ventanas de 5, 20 y 60 días. Se añaden las variaciones de las variables globales, y un indicador de tensión $s_t$, que vale 1 si $\text{VIX}_{t-1}$ supera su percentil 80. Siguiendo la idea de Filippou y otros (2026), que usan la volatilidad cambiaria global como variable de estado, este indicador interactúa con los predictores en los modelos lineales.

## 3.3 Reducción de dimensionalidad
El vector de predictores supera las 80 características (**97 características** en el dataset preprocesado final). Para $h = 20$, la superposición de etiquetas reduce las observaciones independientes a cerca de $n/20$, lo que eleva el riesgo de sobreajuste (maldición de la dimensionalidad). Por eso, además de limitar la profundidad de los árboles, se compara el conjunto completo con el subconjunto de variables que **Lasso** selecciona en cada ventana de entrenamiento. No se usan componentes principales, para mantener un alcance realista.

## 3.4 Modelos y optimización con PSO
Las referencias son la **Caminata Aleatoria (CA)**, **Ridge** y **Lasso**, modelos lineales vistos en el curso. El único modelo candidato es **XGBoost**, elegido porque es el que Pebrianti y otros (2023) ajustan con PSO y el que Pfahler (2021) evalúa en tipos de cambio; así, la adaptación algorítmica se concentra en un solo modelo. Random Forest, LightGBM y CatBoost quedan como trabajo futuro. Por la baja señal de los retornos, la profundidad de los árboles se limita a entre 3 y 6 y se usa parada temprana.

Los hiperparámetros se eligen mediante optimización por enjambre de partículas (**PSO**). Cada partícula codifica un vector de hiperparámetros, con los enteros redondeados y las posiciones recortadas a sus límites. Su aptitud es el error absoluto medio en el último 20 % de la ventana de entrenamiento, lo que respeta el orden temporal:

$$v_i \leftarrow \omega \cdot v_i + c_1 r_1 (p_i - x_i) + c_2 r_2 (g - x_i) \quad ; \quad x_i \leftarrow x_i + v_i \quad (2)$$

Donde $\omega = 0,7$ es la inercia; $c_1 = c_2 = 1,5$ son los coeficientes de aceleración; $r_1$ y $r_2$ son números aleatorios en $[0, 1]$; $p_i$ es la mejor posición de la partícula $i$ y $g$ la mejor del enjambre. PSO se compara con una **búsqueda aleatoria** con el mismo presupuesto de evaluaciones, para medir si el enjambre aporta frente a una alternativa más simple.

## 3.5 Validación y métricas
Se usa validación progresiva con ventana expansiva: en cada origen $\tau$ se entrena con los datos disponibles hasta $\tau$, se pronostican los 20 días siguientes y el origen avanza 20 días. Solo se usan etiquetas con $t + h \le \tau$ (purga), para que el modelo no vea precios del período de prueba. Las métricas son:

$$EAM_h = \frac{1}{n} \sum_{t} \left| S_{t+h} - \hat{S}_{t+h} \right| \quad ; \quad U_h = \frac{EAM_h(\text{modelo})}{EAM_h(CA)}$$

$$AD_h = \frac{1}{n} \sum_{t} \mathbf{1}\left[ \text{signo}\left(\hat{r}_t^{(h)}\right) = \text{signo}\left(r_t^{(h)}\right) \right] \quad (3)$$

Donde $EAM$ es el error absoluto medio, $U$ es el cociente entre el $EAM$ del modelo y el de la $CA$, $n$ es el número de pronósticos, $\mathbf{1}[\cdot]$ vale 1 si se cumple la condición y 0 si no, y $AD$ es el acierto direccional. Un valor $U_h < 1$ indica mejora sobre la $CA$ y se contrasta con la **prueba de Diebold-Mariano**; el $AD$ se contrasta con la **prueba de Pesaran-Timmermann**.

Finalmente, **SHAP** identifica las variables más influyentes de XGBoost, y una **prueba de exclusión por grupos** (reentrenar sin cada grupo de variables: propias del USD/PEN, globales y macroeconómicas) mide si cada grupo aporta capacidad predictiva real.
