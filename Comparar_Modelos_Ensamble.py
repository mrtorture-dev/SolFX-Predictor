"""
SolFX-Predictor | Comparacion Consolidada de Modelos de Ensamble (USD/PEN)
Compara Random Forest, XGBoost, LightGBM y CatBoost frente a Caminata Aleatoria
en los horizontes de 1, 5 y 20 dias habiles.

Uso desde la carpeta raiz del repositorio:
    python Comparar_Modelos_Ensamble.py
"""
from __future__ import annotations

import json
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
            filas.append({
                'modelo': cfg['nombre'],
                'horizonte_dias_habiles': h,
                'n_pronosticos': int(row['n_pronosticos']),
                'mae_precio_modelo_pen': float(row[cfg['col_mae_precio']]),
                'mae_precio_baseline_pen': float(row['mae_precio_baseline_pen']),
                'rmse_precio_modelo_pen': float(row[cfg['col_rmse_precio']]),
                'rmse_precio_baseline_pen': float(row['rmse_precio_baseline_pen']),
                'mae_ret_modelo_bp': float(row[cfg['col_mae_ret']]),
                'mae_ret_baseline_bp': float(row['mae_ret_baseline_bp']),
                'rmse_ret_modelo_bp': float(row[cfg['col_rmse_ret']]),
                'rmse_ret_baseline_bp': float(row['rmse_ret_baseline_bp']),
                'theil_u_rmse': float(row['theil_u_rmse']),
                'r2_ret': float(row[cfg['col_r2_ret']]),
                'acierto_direccional_sin_ceros': float(row['acierto_direccional_sin_ceros']),
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
    return df_consolidado


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
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # Panel 1: MAE en niveles (Soles/USD)
    piv_mae = df_consolidado.pivot(
        index='horizonte_dias_habiles',
        columns='modelo',
        values='mae_precio_modelo_pen',
    )
    rw_mae = df_consolidado.groupby('horizonte_dias_habiles')['mae_precio_baseline_pen'].first()
    piv_mae['Caminata Aleatoria'] = rw_mae
    piv_mae.plot(kind='bar', ax=axes[0], rot=0)
    axes[0].set_title('MAE fuera de muestra (Soles / USD)')
    axes[0].set_xlabel('Horizonte (dias habiles)')
    axes[0].set_ylabel('MAE (PEN por USD)')
    axes[0].grid(axis='y', alpha=0.2)

    # Panel 2: Acierto direccional (%)
    piv_da = df_consolidado.pivot(
        index='horizonte_dias_habiles',
        columns='modelo',
        values='acierto_direccional_sin_ceros',
    ) * 100.0
    piv_da.plot(kind='bar', ax=axes[1], rot=0)
    axes[1].axhline(50.0, color='gray', linestyle='--', linewidth=1.0, label='Azar (50%)')
    axes[1].set_title('Acierto Direccional fuera de muestra (%)')
    axes[1].set_xlabel('Horizonte (dias habiles)')
    axes[1].set_ylabel('Acierto direccional (%)')
    axes[1].legend(loc='best')
    axes[1].grid(axis='y', alpha=0.2)

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
    graficar_comparativa_por_horizonte()
    graficar_barras_metricas(df_consolidado)
    df_proy = consolidar_proyecciones()

    print('=== COMPARACION CONSOLIDADA DE ENSAMBLES (HOLDOUT 60 SESIONES) ===')
    print(df_consolidado[[
        'horizonte_dias_habiles', 'modelo',
        'mae_precio_modelo_pen', 'mae_precio_baseline_pen',
        'theil_u_rmse', 'acierto_direccional_sin_ceros', 'r2_ret'
    ]].to_string(index=False))
    print('\nArchivos comparativos guardados en:', SALIDA)
    print('Modelos sincronizados en:', MODELOS_DIR)


if __name__ == '__main__':
    ejecutar()
