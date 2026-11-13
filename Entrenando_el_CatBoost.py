"""
SolFX-Predictor | Experimento reproducible de CatBoost (USD/PEN)

Uso desde la carpeta raiz del repositorio:
    pip install -r requirements.txt
    python Entrenando_el_CatBoost.py

NOTAS PREVIAS:
  - Regresion DIRECTA de los retornos futuros a 1, 5 y 20 sesiones.
  - Solo variables tecnicas rezagadas y de calendario: las series mensuales de IPC del archivo no incluyen fechas reales de publicacion.
    Para evitar fuga de informacion, no se incluyen hasta que se audite su disponibilidad en un trabajo posterior.
  - La validacion temporal tiene 3 cortes de 252 sesiones con purga gap=h, y el holdout de 60 sesiones se conserva intacto durante la seleccion.
  - La evaluacion es de origen movil: cada origen del holdout puede usar informacion observada hasta ese mismo dia;
    NO es un pronostico fijo a 60 sesiones efectuado en la primera fecha del holdout.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

BASE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
DATOS = BASE / 'data' / 'processed'
SALIDA = BASE / 'resultados_catboost'
SALIDA.mkdir(exist_ok=True)

HORIZONTES = [1, 5, 20]

# Se mantienen exactamente las 18 variables tecnicas y de calendario.
COLUMNAS_X = [
    'PEN_log_ret_lag_1', 'PEN_log_ret_lag_2', 'PEN_log_ret_lag_3',
    'PEN_log_ret_lag_5', 'PEN_log_ret_lag_10', 'PEN_log_ret_lag_21',
    'price_vs_SMA_5', 'price_vs_SMA_10', 'price_vs_SMA_20',
    'RSI', 'GARCH_vol', 'MACD_line', 'MACD_signal',
    'MACD_histogram', 'month_sin', 'month_cos',
    'day_of_week_sin', 'day_of_week_cos',
]

# Candidatos conservadores con arboles simetricos (oblivious trees) de baja
# profundidad y regularizacion L2 alta para controlar la varianza.
CANDIDATOS = [
    {
        'iterations': 220,
        'learning_rate': 0.03,
        'depth': 3,
        'l2_leaf_reg': 12.0,
        'min_data_in_leaf': 25,
    },
    {
        'iterations': 320,
        'learning_rate': 0.03,
        'depth': 4,
        'l2_leaf_reg': 15.0,
        'min_data_in_leaf': 35,
    },
    {
        'iterations': 380,
        'learning_rate': 0.02,
        'depth': 3,
        'l2_leaf_reg': 20.0,
        'min_data_in_leaf': 40,
    },
]


def crear_modelo(hiperparametros):
    return CatBoostRegressor(
        loss_function='RMSE',
        subsample=0.85,
        random_seed=42,
        thread_count=2,
        verbose=False,
        allow_writing_files=False,
        **hiperparametros,
    )


def cargar_datos():
    entrenamiento = pd.read_csv(DATOS / 'usdpen_train.csv', parse_dates=['Fecha'])
    prueba = pd.read_csv(DATOS / 'usdpen_holdout.csv', parse_dates=['Fecha'])

    entrenamiento = entrenamiento.sort_values('Fecha').reset_index(drop=True)
    prueba = prueba.sort_values('Fecha').reset_index(drop=True)

    if entrenamiento['Fecha'].max() >= prueba['Fecha'].min():
        raise ValueError('Los bloques de entrenamiento y prueba se superponen.')

    for nombre, df in [('entrenamiento', entrenamiento), ('holdout', prueba)]:
        if df[COLUMNAS_X + ['USD_PEN']].isna().any().any():
            raise ValueError(f'Hay datos faltantes en {nombre}.')
        if not np.isfinite(df[COLUMNAS_X + ['USD_PEN']].to_numpy()).all():
            raise ValueError(f'Hay datos no finitos en {nombre}.')

    return entrenamiento, prueba


def convertir_a_supervisado(df, horizonte):
    """Origen t -> ln(P[t+h] / P[t]) * 10000 (puntos basicos)."""
    df_x = df.iloc[:-horizonte].copy().reset_index(drop=True)
    p_ini = df['USD_PEN'].iloc[:-horizonte].to_numpy(dtype=float)
    p_final = df['USD_PEN'].iloc[horizonte:].to_numpy(dtype=float)
    retorno_bp = np.log(p_final / p_ini) * 10000.0
    fecha_final = df['Fecha'].iloc[horizonte:].reset_index(drop=True)

    return df_x, retorno_bp, fecha_final, p_final


def seleccionar_modelo(X, y, horizonte):
    """Solo entrenamiento: validacion expansiva en 3 ventanas de 252 sesiones."""
    cv = TimeSeriesSplit(n_splits=3, test_size=252, gap=horizonte)
    resumen = []

    for i, params in enumerate(CANDIDATOS, start=1):
        errores = []
        errores_naive = []

        for indices_entrena, indices_val in cv.split(X):
            modelo = crear_modelo(params)
            modelo.fit(X.iloc[indices_entrena], y[indices_entrena])

            pred_val = modelo.predict(X.iloc[indices_val])
            errores.append(mean_absolute_error(y[indices_val], pred_val))
            errores_naive.append(
                mean_absolute_error(y[indices_val], np.zeros(len(indices_val)))
            )

        resumen.append({
            'horizonte': horizonte,
            'candidato': i,
            'mae_cv_bp': float(np.mean(errores)),
            'mae_cv_rw_bp': float(np.mean(errores_naive)),
            'hiperparametros': json.dumps(params, ensure_ascii=False),
        })

    mejor = min(resumen, key=lambda r: r['mae_cv_bp'])
    return CANDIDATOS[mejor['candidato'] - 1], resumen, mejor


def graficar_predicciones(horizonte, resultados):
    fig, ax = plt.subplots(figsize=(11, 4.5))

    ax.plot(
        resultados['fecha_objetivo'],
        resultados['tipo_cambio_real'],
        label='USD/PEN observado',
        linewidth=2.0,
    )
    ax.plot(
        resultados['fecha_objetivo'],
        resultados['tipo_cambio_catboost'],
        label='Prediccion CatBoost',
        linewidth=1.5,
    )
    ax.plot(
        resultados['fecha_objetivo'],
        resultados['tipo_cambio_caminata_aleatoria'],
        label='Referencia: sin cambio',
        linewidth=1.0,
        alpha=0.7,
    )

    ax.set_title(f'CatBoost USD/PEN — {horizonte} dia(s) habil(es)')
    ax.set_ylabel('Soles por USD')
    ax.set_xlabel('Fecha de valor objetivo')
    ax.legend(loc='best')
    ax.grid(alpha=0.2)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(SALIDA / f'grafico_h{horizonte}.png', dpi=160)
    plt.close(fig)


def graficar_importancias(horizonte, modelo):
    raw_imp = np.asarray(modelo.get_feature_importance(), dtype=float)
    total = np.sum(raw_imp)
    valores = raw_imp / total if total > 0 else raw_imp
    importancias = pd.Series(valores, index=COLUMNAS_X)
    importancias = importancias.sort_values(ascending=False).head(12).sort_values()

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.barh(importancias.index, importancias.values)
    ax.set_title(f'Importancia de variables (PredictionValuesChange) — h={horizonte}')
    ax.set_xlabel('Importancia relativa del modelo')

    fig.tight_layout()
    fig.savefig(SALIDA / f'importancias_h{horizonte}.png', dpi=160)
    plt.close(fig)

    return importancias.sort_values(ascending=False)


def ejecutar():
    train, hold = cargar_datos()

    print(
        'Entrenamiento:', len(train),
        train['Fecha'].min().date(), 'a', train['Fecha'].max().date()
    )
    print(
        'Holdout:', len(hold),
        hold['Fecha'].min().date(), 'a', hold['Fecha'].max().date()
    )
    print('Variables de entrada:', len(COLUMNAS_X))

    metricas, validaciones, proyecciones = [], [], []

    for h in HORIZONTES:
        tr, y_tr, _, _ = convertir_a_supervisado(train, h)
        te, y_te, fechas_destino, precios_reales = convertir_a_supervisado(hold, h)

        X_tr, X_te = tr[COLUMNAS_X], te[COLUMNAS_X]

        params, cv_results, best = seleccionar_modelo(X_tr, y_tr, h)
        validaciones.extend(cv_results)

        modelo = crear_modelo(params)
        modelo.fit(X_tr, y_tr)

        estimado_bp = np.asarray(modelo.predict(X_te), dtype=float)
        precios_actuales = te['USD_PEN'].to_numpy(float)
        precios_estimados = precios_actuales * np.exp(estimado_bp / 10000.0)
        precios_naive = precios_actuales
        retorno_naive = np.zeros(len(y_te))

        mask = y_te != 0
        hit = (
            float(np.mean(np.sign(y_te[mask]) == np.sign(estimado_bp[mask])))
            if mask.any() else float('nan')
        )

        rmse_cb_bp = np.sqrt(mean_squared_error(y_te, estimado_bp))
        rmse_naive_bp = np.sqrt(mean_squared_error(y_te, retorno_naive))

        m = {
            'horizonte_dias_habiles': h,
            'n_pronosticos': len(y_te),
            'mae_ret_catboost_bp': float(mean_absolute_error(y_te, estimado_bp)),
            'mae_ret_baseline_bp': float(mean_absolute_error(y_te, retorno_naive)),
            'rmse_ret_catboost_bp': float(rmse_cb_bp),
            'rmse_ret_baseline_bp': float(rmse_naive_bp),
            'theil_u_rmse': (
                float(rmse_cb_bp / rmse_naive_bp)
                if rmse_naive_bp > 0 else None
            ),
            'r2_ret_catboost': float(r2_score(y_te, estimado_bp)),
            'acierto_direccional_sin_ceros': hit,
            'n_cambios_no_nulos': int(mask.sum()),
            'mae_precio_catboost_pen': float(
                mean_absolute_error(precios_reales, precios_estimados)
            ),
            'mae_precio_baseline_pen': float(
                mean_absolute_error(precios_reales, precios_naive)
            ),
            'rmse_precio_catboost_pen': float(
                np.sqrt(mean_squared_error(precios_reales, precios_estimados))
            ),
            'rmse_precio_baseline_pen': float(
                np.sqrt(mean_squared_error(precios_reales, precios_naive))
            ),
            'mae_cv_catboost_bp': float(best['mae_cv_bp']),
            'mae_cv_baseline_bp': float(best['mae_cv_rw_bp']),
            'mejor_candidato': best['candidato'],
        }
        metricas.append(m)

        predicciones = pd.DataFrame({
            'fecha_origen': te['Fecha'],
            'fecha_objetivo': fechas_destino,
            'tipo_cambio_origen': precios_actuales,
            'tipo_cambio_real': precios_reales,
            'tipo_cambio_catboost': precios_estimados,
            'tipo_cambio_caminata_aleatoria': precios_naive,
            'retorno_real_bp': y_te,
            'retorno_catboost_bp': estimado_bp,
        })

        predicciones.to_csv(SALIDA / f'predicciones_h{h}.csv', index=False)
        graficar_predicciones(h, predicciones)

        importancias = graficar_importancias(h, modelo)
        importancias.rename('importancia_relativa').to_csv(
            SALIDA / f'importancias_h{h}.csv'
        )

        modelo.save_model(str(SALIDA / f'modelo_catboost_h{h}.cbm'))

        p_ultimo = float(hold['USD_PEN'].iloc[-1])
        estimado_ultimo_bp = float(
            np.asarray(modelo.predict(hold[COLUMNAS_X].iloc[[-1]]), dtype=float)[0]
        )

        proyecciones.append({
            'fecha_origen': hold['Fecha'].iloc[-1].date().isoformat(),
            'horizonte_dias_habiles': h,
            'fecha_objetivo_aproximada_sin_feriados': (
                hold['Fecha'].iloc[-1] + pd.offsets.BDay(h)
            ).date().isoformat(),
            'precio_origen': p_ultimo,
            'retorno_esperado_bp': estimado_ultimo_bp,
            'prediccion_usdpen': (
                p_ultimo * np.exp(estimado_ultimo_bp / 10000.0)
            ),
        })

        print(
            f'\nHorizonte {h}: {len(y_te)} pronosticos, '
            f'CV MAE={best["mae_cv_bp"]:.2f} pb'
        )
        print(
            f' MAE tipo de cambio: CatBoost={m["mae_precio_catboost_pen"]:.5f} '
            f'| referencia={m["mae_precio_baseline_pen"]:.5f} soles/USD'
        )
        print(
            f' Theil U={m["theil_u_rmse"]:.4f} '
            f'| Acierto direccional={hit:.1%} '
            f'| R² retornos={m["r2_ret_catboost"]:.4f}'
        )
        print(f' Principales variables: {list(importancias.index[:4])}')

    pd.DataFrame(metricas).to_csv(
        SALIDA / 'metricas_comparativas.csv', index=False
    )
    pd.DataFrame(validaciones).to_csv(
        SALIDA / 'validacion_temporal.csv', index=False
    )
    pd.DataFrame(proyecciones).to_csv(
        SALIDA / 'proyeccion_ilustrativa_ultima_fecha.csv', index=False
    )

    with open(
        SALIDA / 'configuracion_experimento.json',
        'w',
        encoding='utf-8',
    ) as f:
        json.dump({
            'descripcion': 'CatBoost directo sobre retornos futuros en pb',
            'fecha_entrenamiento_final': train['Fecha'].max().date().isoformat(),
            'periodo_holdout': [
                hold['Fecha'].min().date().isoformat(),
                hold['Fecha'].max().date().isoformat(),
            ],
            'horizontes': HORIZONTES,
            'columnas_x': COLUMNAS_X,
            'candidatos': CANDIDATOS,
            'cv': 'TimeSeriesSplit(n_splits=3,test_size=252,gap=h)',
            'origen_movil': True,
            'no_se_usaron': [
                'IPC/Inflacion sin fecha de publicacion',
                'USD_PEN actual como predictor',
                'PEN_log_ret contemporaneo',
            ],
        }, f, ensure_ascii=False, indent=2)

    print('\nArchivos de salida:', SALIDA)


if __name__ == '__main__':
    ejecutar()
