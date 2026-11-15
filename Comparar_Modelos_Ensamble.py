"""
SolFX-Predictor | Comparacion Consolidada de Modelos de Ensamble (USD/PEN)
Compara Random Forest, XGBoost, LightGBM y CatBoost frente a Caminata Aleatoria
en los horizontes de 1, 5 y 20 dias habiles.

Genera:
  - tabla_comparativa_ensambles.csv (nombres tecnicos estandarizados)
  - tabla_comparativa_descriptiva.csv (nombres de columnas descriptivos en espanol)
  - tabla_valores_exactos_grafico_barras.csv (valores exactos de cada barra del grafico resumen)
  - diccionario_variables_y_metricas.csv (explicacion detallada de las 18 variables X y metricas)
  - comparacion_ensambles_h{1,5,20}.png
  - resumen_mae_y_acierto_direccional.png (con etiquetas de valores exactos)

Uso desde la carpeta raiz del repositorio:
    python Comparar_Modelos_Ensamble.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
SALIDA = BASE / 'resultados_comparativos'
MODELOS_DIR = BASE / 'models'
SALIDA.mkdir(exist_ok=True)
MODELOS_DIR.mkdir(exist_ok=True)

HORIZONTES = [1, 5, 20]

MODELOS_CONFIG = [
    {
        'id': 'xgboost',
        'nombre': 'XGBoost',
        'carpeta': BASE / 'resultados_xgboost',
        'col_pred_precio': 'tipo_cambio_xgboost',
        'col_pred_ret': 'retorno_xgboost_bp',
        'col_mae_ret': 'mae_ret_xgb_bp',
        'col_rmse_ret': 'rmse_ret_xgb_bp',
        'col_r2_ret': 'r2_ret_xgb',
        'col_mae_precio': 'mae_precio_xgb_pen',
        'col_rmse_precio': 'rmse_precio_xgb_pen',
        'col_mae_cv': 'mae_cv_xgb_bp',
        'patron_modelo': 'modelo_xgboost_h{h}.json',
    },
    {
        'id': 'lightgbm',
        'nombre': 'LightGBM',
        'carpeta': BASE / 'resultados_lightgbm',
        'col_pred_precio': 'tipo_cambio_lightgbm',
        'col_pred_ret': 'retorno_lightgbm_bp',
        'col_mae_ret': 'mae_ret_lightgbm_bp',
        'col_rmse_ret': 'rmse_ret_lightgbm_bp',
        'col_r2_ret': 'r2_ret_lightgbm',
        'col_mae_precio': 'mae_precio_lightgbm_pen',
        'col_rmse_precio': 'rmse_precio_lightgbm_pen',
        'col_mae_cv': 'mae_cv_lightgbm_bp',
        'patron_modelo': 'modelo_lightgbm_h{h}.txt',
    },
    {
        'id': 'random_forest',
        'nombre': 'Random Forest',
        'carpeta': BASE / 'resultados_random_forest',
        'col_pred_precio': 'tipo_cambio_random_forest',
        'col_pred_ret': 'retorno_random_forest_bp',
        'col_mae_ret': 'mae_ret_random_forest_bp',
        'col_rmse_ret': 'rmse_ret_random_forest_bp',
        'col_r2_ret': 'r2_ret_random_forest',
        'col_mae_precio': 'mae_precio_random_forest_pen',
        'col_rmse_precio': 'rmse_precio_random_forest_pen',
        'col_mae_cv': 'mae_cv_random_forest_bp',
        'patron_modelo': 'modelo_random_forest_h{h}.joblib',
    },
    {
        'id': 'catboost',
        'nombre': 'CatBoost',
        'carpeta': BASE / 'resultados_catboost',
        'col_pred_precio': 'tipo_cambio_catboost',
        'col_pred_ret': 'retorno_catboost_bp',
        'col_mae_ret': 'mae_ret_catboost_bp',
        'col_rmse_ret': 'rmse_ret_catboost_bp',
        'col_r2_ret': 'r2_ret_catboost',
        'col_mae_precio': 'mae_precio_catboost_pen',
        'col_rmse_precio': 'rmse_precio_catboost_pen',
        'col_mae_cv': 'mae_cv_catboost_bp',
        'patron_modelo': 'modelo_catboost_h{h}.cbm',
    },
]


def consolidar_metricas() -> pd.DataFrame:
    filas = []
    for cfg in MODELOS_CONFIG:
        ruta_m = cfg['carpeta'] / 'metricas_comparativas.csv'
        if not ruta_m.exists():
            raise FileNotFoundError(f'No se encontro {ruta_m}. Ejecuta primero el script del modelo {cfg["nombre"]}.')
        df_m = pd.read_csv(ruta_m)
        for _, row in df_m.iterrows():
            h = int(row['horizonte_dias_habiles'])
            n_no_nulos = int(row['n_cambios_no_nulos'])
            hit_rate = float(row['acierto_direccional_sin_ceros'])
            n_aciertos = int(round(hit_rate * n_no_nulos))
            mae_mod = float(row[cfg['col_mae_precio']])
            mae_rw = float(row['mae_precio_baseline_pen'])

            filas.append({
                'modelo': cfg['nombre'],
                'horizonte_dias_habiles': h,
                'n_pronosticos': int(row['n_pronosticos']),
                'n_cambios_no_nulos': n_no_nulos,
                'n_aciertos_direccionales': n_aciertos,
                'mae_precio_modelo_pen': mae_mod,
                'mae_precio_baseline_pen': mae_rw,
                'diferencia_mae_vs_baseline_pen': mae_mod - mae_rw,
                'rmse_precio_modelo_pen': float(row[cfg['col_rmse_precio']]),
                'rmse_precio_baseline_pen': float(row['rmse_precio_baseline_pen']),
                'mae_ret_modelo_bp': float(row[cfg['col_mae_ret']]),
                'mae_ret_baseline_bp': float(row['mae_ret_baseline_bp']),
                'rmse_ret_modelo_bp': float(row[cfg['col_rmse_ret']]),
                'rmse_ret_baseline_bp': float(row['rmse_ret_baseline_bp']),
                'theil_u_rmse': float(row['theil_u_rmse']),
                'r2_ret': float(row[cfg['col_r2_ret']]),
                'acierto_direccional_sin_ceros': hit_rate,
                'acierto_direccional_porcentaje': hit_rate * 100.0,
                'mae_cv_bp': float(row[cfg['col_mae_cv']]),
                'mae_cv_baseline_bp': float(row['mae_cv_baseline_bp']),
                'mejor_candidato': int(row['mejor_candidato']),
            })

            # Sincronizar artefactos de modelos entrenados hacia la carpeta models/
            archivo_modelo = cfg['carpeta'] / cfg['patron_modelo'].format(h=h)
            if archivo_modelo.exists():
                shutil.copy2(archivo_modelo, MODELOS_DIR / archivo_modelo.name)

    df_consolidado = pd.DataFrame(filas).sort_values(
        ['horizonte_dias_habiles', 'mae_precio_modelo_pen']
    ).reset_index(drop=True)
    df_consolidado.to_csv(SALIDA / 'tabla_comparativa_ensambles.csv', index=False)

    # Version con encabezados descriptivos en espanol
    mapa_descriptivo = {
        'modelo': 'Algoritmo de Ensamble',
        'horizonte_dias_habiles': 'Horizonte de Pronostico (Dias Habiles)',
        'n_pronosticos': 'Pronosticos Evaluados en Holdout (N)',
        'n_cambios_no_nulos': 'Sesiones con Movimiento Real No Nulo',
        'n_aciertos_direccionales': 'Sesiones con Direccion Acertada',
        'mae_precio_modelo_pen': 'MAE del Modelo en Nivel (Soles por USD)',
        'mae_precio_baseline_pen': 'MAE Caminata Aleatoria en Nivel (Soles por USD)',
        'diferencia_mae_vs_baseline_pen': 'Diferencia MAE Modelo menos Caminata Aleatoria (Soles por USD)',
        'rmse_precio_modelo_pen': 'RMSE del Modelo en Nivel (Soles por USD)',
        'rmse_precio_baseline_pen': 'RMSE Caminata Aleatoria en Nivel (Soles por USD)',
        'mae_ret_modelo_bp': 'MAE del Modelo en Retornos (Puntos Basicos)',
        'mae_ret_baseline_bp': 'MAE Caminata Aleatoria en Retornos (Puntos Basicos)',
        'rmse_ret_modelo_bp': 'RMSE del Modelo en Retornos (Puntos Basicos)',
        'rmse_ret_baseline_bp': 'RMSE Caminata Aleatoria en Retornos (Puntos Basicos)',
        'theil_u_rmse': 'Coeficiente U de Theil (Ratio RMSE Modelo / Caminata Aleatoria)',
        'r2_ret': 'Coeficiente de Determinacion R2 sobre Retornos',
        'acierto_direccional_porcentaje': 'Acierto Direccional de Subida/Bajada (%)',
        'mae_cv_bp': 'MAE en Validacion Cruzada Temporal (Puntos Basicos)',
        'mae_cv_baseline_bp': 'MAE Caminata Aleatoria en Validacion Cruzada (Puntos Basicos)',
        'mejor_candidato': 'Configuracion Hiperparametros Seleccionada (ID)',
    }
    cols_export = [c for c in mapa_descriptivo.keys() if c in df_consolidado.columns]
    df_descriptivo = df_consolidado[cols_export].rename(columns=mapa_descriptivo)
    df_descriptivo.to_csv(SALIDA / 'tabla_comparativa_descriptiva.csv', index=False)

    return df_consolidado


def generar_tabla_valores_exactos_barras(df_consolidado: pd.DataFrame) -> pd.DataFrame:
    """
    Construye la tabla con los valores exactos graficados en los dos paneles
    de resumen_mae_y_acierto_direccional.png (Panel 1: MAE en Soles/USD;
    Panel 2: Acierto Direccional en % y conteo de sesiones).
    """
    filas_barras = []
    for h in HORIZONTES:
        sub = df_consolidado[df_consolidado['horizonte_dias_habiles'] == h].set_index('modelo')
        rw_mae_pen = float(sub['mae_precio_baseline_pen'].iloc[0])
        rw_mae_bp = float(sub['mae_ret_baseline_bp'].iloc[0])
        n_eval = int(sub['n_pronosticos'].iloc[0])
        n_mov = int(sub['n_cambios_no_nulos'].iloc[0])

        filas_barras.append({
            'Horizonte (Dias Habiles)': h,
            'Pronosticos Evaluados (N)': n_eval,
            'Movimientos No Nulos': n_mov,
            # Panel Izquierdo: MAE en Soles por USD
            'MAE Caminata Aleatoria (Soles/USD)': round(rw_mae_pen, 6),
            'MAE Random Forest (Soles/USD)': round(float(sub.loc['Random Forest', 'mae_precio_modelo_pen']), 6),
            'MAE CatBoost (Soles/USD)': round(float(sub.loc['CatBoost', 'mae_precio_modelo_pen']), 6),
            'MAE XGBoost (Soles/USD)': round(float(sub.loc['XGBoost', 'mae_precio_modelo_pen']), 6),
            'MAE LightGBM (Soles/USD)': round(float(sub.loc['LightGBM', 'mae_precio_modelo_pen']), 6),
            # Equivalente en Puntos Basicos de retorno
            'MAE Caminata Aleatoria (pb)': round(rw_mae_bp, 2),
            'MAE Random Forest (pb)': round(float(sub.loc['Random Forest', 'mae_ret_modelo_bp']), 2),
            'MAE CatBoost (pb)': round(float(sub.loc['CatBoost', 'mae_ret_modelo_bp']), 2),
            'MAE XGBoost (pb)': round(float(sub.loc['XGBoost', 'mae_ret_modelo_bp']), 2),
            'MAE LightGBM (pb)': round(float(sub.loc['LightGBM', 'mae_ret_modelo_bp']), 2),
            # Panel Derecho: Acierto Direccional (%)
            'Acierto Direccional Random Forest (%)': round(float(sub.loc['Random Forest', 'acierto_direccional_porcentaje']), 2),
            'Acierto Direccional CatBoost (%)': round(float(sub.loc['CatBoost', 'acierto_direccional_porcentaje']), 2),
            'Acierto Direccional XGBoost (%)': round(float(sub.loc['XGBoost', 'acierto_direccional_porcentaje']), 2),
            'Acierto Direccional LightGBM (%)': round(float(sub.loc['LightGBM', 'acierto_direccional_porcentaje']), 2),
            'Referencia Azar Direccional (%)': 50.00,
            # Conteo exacto de aciertos / total movimientos
            'Aciertos Exactos Random Forest': f"{int(sub.loc['Random Forest', 'n_aciertos_direccionales'])}/{n_mov}",
            'Aciertos Exactos CatBoost': f"{int(sub.loc['CatBoost', 'n_aciertos_direccionales'])}/{n_mov}",
            'Aciertos Exactos XGBoost': f"{int(sub.loc['XGBoost', 'n_aciertos_direccionales'])}/{n_mov}",
            'Aciertos Exactos LightGBM': f"{int(sub.loc['LightGBM', 'n_aciertos_direccionales'])}/{n_mov}",
        })

    df_barras = pd.DataFrame(filas_barras)
    df_barras.to_csv(SALIDA / 'tabla_valores_exactos_grafico_barras.csv', index=False)
    return df_barras


def generar_diccionario_variables_y_metricas() -> pd.DataFrame:
    registros = [
        # 18 Variables predictoras (COLUMNAS_X)
        ('Variable Predictora (X)', 'PEN_log_ret_lag_1', 'Retorno logaritmico rezagado 1 dia habil', 'ln(P_{t-1} / P_{t-2})', 'Captura autocorrelacion de primer orden y reversion inmediata del tipo de cambio.'),
        ('Variable Predictora (X)', 'PEN_log_ret_lag_2', 'Retorno logaritmico rezagado 2 dias habiles', 'ln(P_{t-2} / P_{t-3})', 'Mide el impulso o correccion de dos sesiones previas.'),
        ('Variable Predictora (X)', 'PEN_log_ret_lag_3', 'Retorno logaritmico rezagado 3 dias habiles', 'ln(P_{t-3} / P_{t-4})', 'Refleja la dinamica intrasemanal de corto plazo (3 sesiones).'),
        ('Variable Predictora (X)', 'PEN_log_ret_lag_5', 'Retorno logaritmico rezagado 5 dias habiles (1 semana)', 'ln(P_{t-5} / P_{t-6})', 'Captura efectos de memoria de una semana bursatil.'),
        ('Variable Predictora (X)', 'PEN_log_ret_lag_10', 'Retorno logaritmico rezagado 10 dias habiles (2 semanas)', 'ln(P_{t-10} / P_{t-11})', 'Mide persistencia quincenal en las variaciones del sol.'),
        ('Variable Predictora (X)', 'PEN_log_ret_lag_21', 'Retorno logaritmico rezagado 21 dias habiles (1 mes)', 'ln(P_{t-21} / P_{t-22})', 'Captura ciclos mensuales de liquidez y cierre contable.'),
        ('Variable Predictora (X)', 'price_vs_SMA_5', 'Distancia relativa a la Media Movil Simple de 5 dias', '(P_{t-1} - SMA_5(P_{t-1})) / SMA_5(P_{t-1})', 'Mide sobrecompra/sobreventa frente al promedio de la ultima semana.'),
        ('Variable Predictora (X)', 'price_vs_SMA_10', 'Distancia relativa a la Media Movil Simple de 10 dias', '(P_{t-1} - SMA_10(P_{t-1})) / SMA_10(P_{t-1})', 'Desviacion porcentual frente a la tendencia de dos semanas bursatiles.'),
        ('Variable Predictora (X)', 'price_vs_SMA_20', 'Distancia relativa a la Media Movil Simple de 20 dias', '(P_{t-1} - SMA_20(P_{t-1})) / SMA_20(P_{t-1})', 'Desviacion porcentual frente al equilibrio promedio mensual.'),
        ('Variable Predictora (X)', 'RSI', 'Indice de Fuerza Relativa de 14 sesiones normalizado', '(RSI_14(P_{t-1}) - 50) / 50 en [-1, 1]', 'Oscilador de velocidad y agotamiento de tendencia; valores cercanos a +1 indican sobrecompra y a -1 sobreventa.'),
        ('Variable Predictora (X)', 'GARCH_vol', 'Volatilidad historica movil de 21 sesiones (Proxy GARCH)', 'Desv. Estandar movil de 21 dias sobre r_{t-1}', 'Cuantifica el regimen de incertidumbre y tension cambiaria reciente.'),
        ('Variable Predictora (X)', 'MACD_line', 'Linea MACD normalizada (12 vs 26 sesiones)', '(EMA_12(P_{t-1}) - EMA_26(P_{t-1})) / P_{t-1}', 'Diferencial de medias exponenciales rapida y lenta relativo al nivel del tipo de cambio.'),
        ('Variable Predictora (X)', 'MACD_signal', 'Linea de Senal del MACD (EMA de 9 sesiones)', 'EMA_9(MACD_line)', 'Suavizamiento exponencial de la linea MACD que actua como umbral de cambio de tendencia.'),
        ('Variable Predictora (X)', 'MACD_histogram', 'Histograma de divergencia MACD', 'MACD_line - MACD_signal', 'Mide la aceleracion del momento alcista o bajista del dolar.'),
        ('Variable Predictora (X)', 'month_sin', 'Componente Seno del mes calendario', 'sin(2 * pi * mes / 12)', 'Codifica la estacionalidad anual (ej. pago de impuestos en marzo/abril, gratificaciones en julio/diciembre).'),
        ('Variable Predictora (X)', 'month_cos', 'Componente Coseno del mes calendario', 'cos(2 * pi * mes / 12)', 'Complemento ortogonal del seno mensual para preservar continuidad entre diciembre y enero.'),
        ('Variable Predictora (X)', 'day_of_week_sin', 'Componente Seno del dia de la semana', 'sin(2 * pi * dia_semana / 7)', 'Captura patrones de liquidez intrasemanal (apertura de lunes vs cierre de viernes).'),
        ('Variable Predictora (X)', 'day_of_week_cos', 'Componente Coseno del dia de la semana', 'cos(2 * pi * dia_semana / 7)', 'Complemento ortogonal para representar ciclicamente el dia de la semana.'),
        # Metricas de las tablas
        ('Metrica de Evaluacion', 'horizonte_dias_habiles', 'Horizonte de pronostico (h)', 'h in {1, 5, 20} sesiones bursatiles', 'Plazo hacia adelante: 1 dia (diario), 5 dias (semanal) y 20 dias (mensual).'),
        ('Metrica de Evaluacion', 'n_pronosticos', 'Numero de pronosticos evaluados en Holdout', '60 - h observaciones (59, 55 y 40)', 'Cantidad de pares (origen t, destino t+h) completamente contenidos en los 60 dias reservados.'),
        ('Metrica de Evaluacion', 'mae_cv_bp', 'MAE en Validacion Cruzada Temporal (pb)', 'Promedio de |y - y_hat| en 3 cortes de 252 dias', 'Error absoluto medio en puntos basicos (1 pb = 0.01%) usado para elegir hiperparametros sin tocar el holdout.'),
        ('Metrica de Evaluacion', 'mae_precio_modelo_pen', 'MAE del Modelo en Nivel (Soles/USD)', 'mean(|P_{t+h} - P_hat_{t+h}|)', 'Error promedio directo en soles por dolar. Menor valor indica mayor precision cambiaria.'),
        ('Metrica de Evaluacion', 'mae_precio_baseline_pen', 'MAE de Caminata Aleatoria en Nivel (Soles/USD)', 'mean(|P_{t+h} - P_t|)', 'Error promedio de asumir que el tipo de cambio no variara entre t y t+h.'),
        ('Metrica de Evaluacion', 'rmse_precio_modelo_pen', 'RMSE del Modelo en Nivel (Soles/USD)', 'sqrt(mean((P_{t+h} - P_hat_{t+h})^2))', 'Raiz del error cuadratico medio en soles/USD; penaliza con mayor severidad errores grandes.'),
        ('Metrica de Evaluacion', 'theil_u_rmse', 'Indice U de Theil (Ratio RMSE)', 'RMSE_Modelo / RMSE_Caminata_Aleatoria', 'Si U < 1, el modelo mejora a la caminata aleatoria en error cuadratico; si U > 1, no la supera.'),
        ('Metrica de Evaluacion', 'acierto_direccional_sin_ceros', 'Acierto Direccional (Hit Rate)', 'Proporcion donde sign(retorno_real) == sign(retorno_predicho)', 'Porcentaje de veces que el modelo anticipa correctamente si el dolar sube o baja (excluyendo variaciones nulas).'),
        ('Metrica de Evaluacion', 'r2_ret', 'Coeficiente de Determinacion R2 sobre Retornos', '1 - SS_res / SS_tot', 'Fraccion de la varianza del retorno explicada fuera de muestra; valores negativos indican mayor error que la media.'),
    ]
    df_dic = pd.DataFrame(
        registros,
        columns=['Categoria', 'Codigo Variable / Columna', 'Nombre Descriptivo', 'Formula / Definicion', 'Interpretacion Economica y Estadistica'],
    )
    df_dic.to_csv(SALIDA / 'diccionario_variables_y_metricas.csv', index=False)
    return df_dic


def graficar_comparativa_por_horizonte():
    for h in HORIZONTES:
        fig, ax = plt.subplots(figsize=(11.5, 4.8))
        base_plotted = False

        for cfg in MODELOS_CONFIG:
            df_p = pd.read_csv(
                cfg['carpeta'] / f'predicciones_h{h}.csv',
                parse_dates=['fecha_origen', 'fecha_objetivo'],
            )
            if not base_plotted:
                ax.plot(
                    df_p['fecha_objetivo'],
                    df_p['tipo_cambio_real'],
                    label='USD/PEN observado',
                    color='black',
                    linewidth=2.2,
                )
                ax.plot(
                    df_p['fecha_objetivo'],
                    df_p['tipo_cambio_caminata_aleatoria'],
                    label='Referencia: caminata aleatoria',
                    color='gray',
                    linestyle='--',
                    linewidth=1.2,
                    alpha=0.8,
                )
                base_plotted = True

            ax.plot(
                df_p['fecha_objetivo'],
                df_p[cfg['col_pred_precio']],
                label=cfg['nombre'],
                linewidth=1.4,
                alpha=0.9,
            )

        ax.set_title(f'Comparacion de Ensambles USD/PEN — Horizonte h={h} dia(s) habil(es)')
        ax.set_ylabel('Soles por USD')
        ax.set_xlabel('Fecha de valor objetivo')
        ax.legend(loc='best', ncol=2)
        ax.grid(alpha=0.2)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(SALIDA / f'comparacion_ensambles_h{h}.png', dpi=160)
        plt.close(fig)


def graficar_barras_metricas(df_consolidado: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5))
    orden_modelos = ['Random Forest', 'CatBoost', 'XGBoost', 'LightGBM']

    # Panel 1: MAE en niveles (Soles/USD)
    piv_mae = df_consolidado.pivot(
        index='horizonte_dias_habiles',
        columns='modelo',
        values='mae_precio_modelo_pen',
    )[orden_modelos]
    rw_mae = df_consolidado.groupby('horizonte_dias_habiles')['mae_precio_baseline_pen'].first()
    piv_mae['Caminata Aleatoria'] = rw_mae

    piv_mae.plot(kind='bar', ax=axes[0], rot=0, width=0.82)
    axes[0].set_title('Panel A: Error Absoluto Medio (MAE en Soles / USD)')
    axes[0].set_xlabel('Horizonte de pronostico (dias habiles)')
    axes[0].set_ylabel('MAE (Soles por USD)')
    axes[0].set_ylim(0, piv_mae.to_numpy().max() * 1.18)
    axes[0].grid(axis='y', alpha=0.2)
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt='%.4f', fontsize=6.5, rotation=90, padding=2)

    # Panel 2: Acierto direccional (%)
    piv_da = (
        df_consolidado.pivot(
            index='horizonte_dias_habiles',
            columns='modelo',
            values='acierto_direccional_sin_ceros',
        )[orden_modelos]
        * 100.0
    )
    piv_da.plot(kind='bar', ax=axes[1], rot=0, width=0.78)
    axes[1].axhline(50.0, color='gray', linestyle='--', linewidth=1.1, label='Referencia Azar (50.0%)')
    axes[1].set_title('Panel B: Acierto Direccional de Subida/Bajada (%)')
    axes[1].set_xlabel('Horizonte de pronostico (dias habiles)')
    axes[1].set_ylabel('Acierto direccional (%)')
    axes[1].set_ylim(0, 62)
    axes[1].legend(loc='upper right')
    axes[1].grid(axis='y', alpha=0.2)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt='%.1f%%', fontsize=7, padding=2)

    fig.tight_layout()
    fig.savefig(SALIDA / 'resumen_mae_y_acierto_direccional.png', dpi=160)
    plt.close(fig)


def consolidar_proyecciones() -> pd.DataFrame:
    filas = []
    for cfg in MODELOS_CONFIG:
        ruta_p = cfg['carpeta'] / 'proyeccion_ilustrativa_ultima_fecha.csv'
        df_p = pd.read_csv(ruta_p)
        df_p.insert(0, 'modelo', cfg['nombre'])
        filas.append(df_p)
    df_proy = pd.concat(filas, ignore_index=True).sort_values(
        ['horizonte_dias_habiles', 'modelo']
    ).reset_index(drop=True)
    df_proy.to_csv(SALIDA / 'proyecciones_ilustrativas_consolidadas.csv', index=False)
    return df_proy


def ejecutar():
    df_consolidado = consolidar_metricas()
    df_barras = generar_tabla_valores_exactos_barras(df_consolidado)
    df_dic = generar_diccionario_variables_y_metricas()
    graficar_comparativa_por_horizonte()
    graficar_barras_metricas(df_consolidado)
    consolidar_proyecciones()

    print('=== TABLA DE VALORES EXACTOS DEL GRAFICO DE BARRAS ===')
    print(df_barras.to_string(index=False))
    print('\nArchivos comparativos guardados en:', SALIDA)


if __name__ == '__main__':
    ejecutar()
