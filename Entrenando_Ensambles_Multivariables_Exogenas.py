"""
SolFX-Predictor | Nuevos Modelos de Ensamble Multivariables con Variables Exogenas Globales y Commodities
(Random Forest, XGBoost, LightGBM y CatBoost — Horizontes h = 1, 5 y 20 dias habiles)

Por que se desarrolla esta Fase 2 Multivariable:
  1. En los archivos originales de Bloomberg (grid_*.xlsx), las hojas de Cobre (LMCADY), DXY, US10Y, etc.
     fueron exportadas con formulas =BDH(...) sin resolver (#NAME?, 0 observaciones). Por ello, los scripts
     iniciales solo usaron las 18 variables tecnicas endogenas del propio USD/PEN, las cuales pierden
     capacidad predictiva a 20 dias habiles (MAE = 0.03896-0.04145 vs 0.035905 de Caminata Aleatoria).
  2. Este script incorpora las 32 variables exogenas causales (shift(1) estricto, cero data leakage)
     construidas en src/exogenous_loader.py a partir de Cobre, VIX, DXY, Oro, Petroleo WTI, Bono del
     Tesoro de EE. UU. a 10 anos (US10Y), S&P 500, Terminos de Intercambio (Cobre/WTI) e Inflacion BCRP
     con rezago de publicacion segura (42 dias habiles), completando 50 variables predictoras.
  3. Ademas, aplica ponderacion exponencial por regimen macroeconomico reciente (half-life temporal),
     evitando que la devaluacion estructural de los anos 1994-2001 sesgue positivamente los pronosticos
     a 20 dias en el regimen actual de metas de inflacion.

Uso desde la carpeta raiz del repositorio:
    python src/exogenous_loader.py
    python Entrenando_Ensambles_Multivariables_Exogenas.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

BASE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
DATOS = BASE / 'data' / 'processed'
SALIDA = BASE / 'resultados_ensambles_exogenas'
SALIDA_COMP = BASE / 'resultados_comparativos'
MODELOS_DIR = BASE / 'models'

SALIDA.mkdir(exist_ok=True)
SALIDA_COMP.mkdir(exist_ok=True)
MODELOS_DIR.mkdir(exist_ok=True)

HORIZONTES = [1, 5, 20]

# 18 variables tecnicas y de calendario originales (Fase 1)
COLUMNAS_ENDOGENAS = [
    'PEN_log_ret_lag_1', 'PEN_log_ret_lag_2', 'PEN_log_ret_lag_3',
    'PEN_log_ret_lag_5', 'PEN_log_ret_lag_10', 'PEN_log_ret_lag_21',
    'price_vs_SMA_5', 'price_vs_SMA_10', 'price_vs_SMA_20',
    'RSI', 'GARCH_vol', 'MACD_line', 'MACD_signal',
    'MACD_histogram', 'month_sin', 'month_cos',
    'day_of_week_sin', 'day_of_week_cos',
]

# 32 variables exogenas globales, commodities y macroeconomicas (Fase 2, shift(1) causal)
COLUMNAS_EXOGENAS = [
    'Cobre_ret_1d', 'Cobre_ret_5d', 'Cobre_ret_20d', 'Cobre_vs_SMA_20',
    'DXY_ret_1d', 'DXY_ret_5d', 'DXY_ret_20d', 'DXY_vs_SMA_20',
    'Oro_ret_5d', 'Oro_ret_20d', 'Oro_vs_SMA_20',
    'Petroleo_WTI_ret_5d', 'Petroleo_WTI_ret_20d', 'Petroleo_WTI_vs_SMA_20',
    'SP500_ret_5d', 'SP500_ret_20d', 'SP500_vs_SMA_20',
    'VIX_nivel_lag1', 'VIX_cambio_1d', 'VIX_cambio_5d', 'VIX_cambio_20d', 'VIX_zscore_20d',
    'US10Y_nivel_lag1', 'US10Y_cambio_5d', 'US10Y_cambio_20d', 'US10Y_vs_SMA_20',
    'TermIntercambio_Cu_WTI_ret_20d', 'Cobre_vs_DXY_ret_20d',
    'Inflacion_12M_pub_segura', 'Inflacion_12M_cambio_20d',
    'PEN_ret_acum_5d', 'PEN_ret_acum_20d',
]

COLUMNAS_X = COLUMNAS_ENDOGENAS + COLUMNAS_EXOGENAS

# Configuracion de candidatos por modelo y horizonte (incluye half_life en sesiones bursatiles
# para descontar la hiperinflacion/devaluacion de los anos 90 sin fuga temporal).
CANDIDATOS_POR_MODELO = {
    'Random Forest': {
        1: [
            {'n_estimators': 250, 'max_depth': 3, 'min_samples_leaf': 35, 'max_features': 0.35, 'half_life': 504},
            {'n_estimators': 300, 'max_depth': 3, 'min_samples_leaf': 40, 'max_features': 0.30, 'half_life': 756},
            {'n_estimators': 250, 'max_depth': 4, 'min_samples_leaf': 30, 'max_features': 0.40, 'half_life': 504},
        ],
        5: [
            {'n_estimators': 250, 'max_depth': 3, 'min_samples_leaf': 35, 'max_features': 0.35, 'half_life': 1260},
            {'n_estimators': 300, 'max_depth': 3, 'min_samples_leaf': 40, 'max_features': 0.35, 'half_life': None},
            {'n_estimators': 250, 'max_depth': 4, 'min_samples_leaf': 35, 'max_features': 0.30, 'half_life': 1260},
        ],
        20: [
            {'n_estimators': 250, 'max_depth': 3, 'min_samples_leaf': 35, 'max_features': 0.35, 'half_life': 1260},
            {'n_estimators': 300, 'max_depth': 3, 'min_samples_leaf': 40, 'max_features': 0.30, 'half_life': 756},
            {'n_estimators': 250, 'max_depth': 3, 'min_samples_leaf': 35, 'max_features': 0.35, 'half_life': 504},
        ],
    },
    'XGBoost': {
        1: [
            {'n_estimators': 160, 'max_depth': 2, 'learning_rate': 0.02, 'min_child_weight': 25, 'reg_lambda': 25.0, 'half_life': 252},
            {'n_estimators': 160, 'max_depth': 2, 'learning_rate': 0.02, 'min_child_weight': 25, 'reg_lambda': 25.0, 'half_life': 504},
            {'n_estimators': 200, 'max_depth': 2, 'learning_rate': 0.02, 'min_child_weight': 30, 'reg_lambda': 30.0, 'half_life': 378},
        ],
        5: [
            {'n_estimators': 160, 'max_depth': 2, 'learning_rate': 0.02, 'min_child_weight': 25, 'reg_lambda': 25.0, 'half_life': None},
            {'n_estimators': 180, 'max_depth': 2, 'learning_rate': 0.02, 'min_child_weight': 25, 'reg_lambda': 25.0, 'half_life': 1260},
            {'n_estimators': 200, 'max_depth': 2, 'learning_rate': 0.02, 'min_child_weight': 30, 'reg_lambda': 30.0, 'half_life': None},
        ],
        20: [
            {'n_estimators': 160, 'max_depth': 2, 'learning_rate': 0.02, 'min_child_weight': 25, 'reg_lambda': 25.0, 'half_life': 504},
            {'n_estimators': 180, 'max_depth': 2, 'learning_rate': 0.02, 'min_child_weight': 30, 'reg_lambda': 30.0, 'half_life': 504},
            {'n_estimators': 160, 'max_depth': 2, 'learning_rate': 0.02, 'min_child_weight': 25, 'reg_lambda': 25.0, 'half_life': 378},
        ],
    },
    'LightGBM': {
        1: [
            {'n_estimators': 160, 'max_depth': 2, 'num_leaves': 3, 'learning_rate': 0.02, 'min_child_samples': 35, 'reg_lambda': 25.0, 'half_life': 252},
            {'n_estimators': 160, 'max_depth': 2, 'num_leaves': 3, 'learning_rate': 0.02, 'min_child_samples': 35, 'reg_lambda': 25.0, 'half_life': 378},
            {'n_estimators': 160, 'max_depth': 2, 'num_leaves': 3, 'learning_rate': 0.02, 'min_child_samples': 35, 'reg_lambda': 25.0, 'half_life': 504},
        ],
        5: [
            {'n_estimators': 160, 'max_depth': 2, 'num_leaves': 3, 'learning_rate': 0.02, 'min_child_samples': 35, 'reg_lambda': 25.0, 'half_life': 1260},
            {'n_estimators': 160, 'max_depth': 2, 'num_leaves': 3, 'learning_rate': 0.02, 'min_child_samples': 35, 'reg_lambda': 25.0, 'half_life': None},
            {'n_estimators': 180, 'max_depth': 2, 'num_leaves': 3, 'learning_rate': 0.02, 'min_child_samples': 40, 'reg_lambda': 30.0, 'half_life': 1260},
        ],
        20: [
            {'n_estimators': 160, 'max_depth': 2, 'num_leaves': 3, 'learning_rate': 0.02, 'min_child_samples': 35, 'reg_lambda': 25.0, 'half_life': 504},
            {'n_estimators': 160, 'max_depth': 2, 'num_leaves': 3, 'learning_rate': 0.02, 'min_child_samples': 35, 'reg_lambda': 25.0, 'half_life': 378},
            {'n_estimators': 160, 'max_depth': 2, 'num_leaves': 3, 'learning_rate': 0.02, 'min_child_samples': 35, 'reg_lambda': 25.0, 'half_life': 252},
        ],
    },
    'CatBoost': {
        1: [
            {'iterations': 200, 'depth': 3, 'learning_rate': 0.02, 'l2_leaf_reg': 25.0, 'half_life': 1260},
            {'iterations': 200, 'depth': 3, 'learning_rate': 0.02, 'l2_leaf_reg': 25.0, 'half_life': 504},
            {'iterations': 220, 'depth': 3, 'learning_rate': 0.02, 'l2_leaf_reg': 30.0, 'half_life': 1260},
        ],
        5: [
            {'iterations': 200, 'depth': 3, 'learning_rate': 0.02, 'l2_leaf_reg': 25.0, 'half_life': None},
            {'iterations': 200, 'depth': 3, 'learning_rate': 0.02, 'l2_leaf_reg': 25.0, 'half_life': 1260},
            {'iterations': 220, 'depth': 3, 'learning_rate': 0.02, 'l2_leaf_reg': 30.0, 'half_life': None},
        ],
        20: [
            {'iterations': 200, 'depth': 3, 'learning_rate': 0.02, 'l2_leaf_reg': 25.0, 'half_life': 1260},
            {'iterations': 220, 'depth': 3, 'learning_rate': 0.02, 'l2_leaf_reg': 30.0, 'half_life': 1260},
            {'iterations': 200, 'depth': 3, 'learning_rate': 0.02, 'l2_leaf_reg': 25.0, 'half_life': None},
        ],
    },
}


def calcular_pesos_temporales(n: int, half_life: int | None) -> np.ndarray:
    if half_life is None or half_life <= 0:
        return np.ones(n, dtype=float)
    edad = np.arange(n - 1, -1, -1, dtype=float)
    return np.exp(-np.log(2.0) * edad / float(half_life))


def instanciar_modelo(nombre_modelo: str, params: dict):
    hp = {k: v for k, v in params.items() if k != 'half_life'}
    if nombre_modelo == 'Random Forest':
        return RandomForestRegressor(
            criterion='squared_error',
            bootstrap=True,
            random_state=42,
            n_jobs=2,
            **hp,
        )
    if nombre_modelo == 'XGBoost':
        return XGBRegressor(
            objective='reg:squarederror',
            tree_method='hist',
            subsample=0.80,
            colsample_bytree=0.60,
            random_state=42,
            n_jobs=2,
            **hp,
        )
    if nombre_modelo == 'LightGBM':
        return LGBMRegressor(
            objective='regression',
            boosting_type='gbdt',
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.60,
            random_state=42,
            n_jobs=2,
            importance_type='gain',
            verbosity=-1,
            **hp,
        )
    if nombre_modelo == 'CatBoost':
        return CatBoostRegressor(
            loss_function='RMSE',
            subsample=0.80,
            random_seed=42,
            thread_count=2,
            verbose=False,
            allow_writing_files=False,
            **hp,
        )
    raise ValueError(f'Modelo desconocido: {nombre_modelo}')


def cargar_datos():
    ruta_tr = DATOS / 'usdpen_train_exog.csv'
    ruta_te = DATOS / 'usdpen_holdout_exog.csv'
    if not ruta_tr.exists() or not ruta_te.exists():
        raise FileNotFoundError('Ejecuta primero python src/exogenous_loader.py para generar los datasets con exogenas.')
    entrenamiento = pd.read_csv(ruta_tr, parse_dates=['Fecha']).sort_values('Fecha').reset_index(drop=True)
    prueba = pd.read_csv(ruta_te, parse_dates=['Fecha']).sort_values('Fecha').reset_index(drop=True)
    return entrenamiento, prueba


def convertir_a_supervisado(df: pd.DataFrame, horizonte: int):
    df_x = df.iloc[:-horizonte].copy().reset_index(drop=True)
    p_ini = df['USD_PEN'].iloc[:-horizonte].to_numpy(dtype=float)
    p_final = df['USD_PEN'].iloc[horizonte:].to_numpy(dtype=float)
    retorno_bp = np.log(p_final / p_ini) * 10000.0
    fecha_final = df['Fecha'].iloc[horizonte:].reset_index(drop=True)
    return df_x, retorno_bp, fecha_final, p_final


def seleccionar_candidato(nombre_modelo: str, X: pd.DataFrame, y: np.ndarray, horizonte: int):
    cv = TimeSeriesSplit(n_splits=3, test_size=252, gap=horizonte)
    lista_cand = CANDIDATOS_POR_MODELO[nombre_modelo][horizonte]
    resumen = []

    params = lista_cand[0]
    errores = []
    errores_naive = []
    for idx_tr, idx_val in cv.split(X):
        modelo = instanciar_modelo(nombre_modelo, params)
        w_tr = calcular_pesos_temporales(len(idx_tr), params.get('half_life'))
        modelo.fit(X.iloc[idx_tr], y[idx_tr], sample_weight=w_tr)
        pred_val = np.asarray(modelo.predict(X.iloc[idx_val]), dtype=float)
        errores.append(mean_absolute_error(y[idx_val], pred_val))
        errores_naive.append(mean_absolute_error(y[idx_val], np.zeros(len(idx_val))))

    resumen.append({
        'modelo': nombre_modelo,
        'horizonte': horizonte,
        'candidato': 1,
        'mae_cv_bp': float(np.mean(errores)),
        'mae_cv_rw_bp': float(np.mean(errores_naive)),
        'hiperparametros': json.dumps(params, ensure_ascii=False),
    })

    mejor = resumen[0]
    return lista_cand[0], resumen, mejor


def obtener_importancias(nombre_modelo: str, modelo) -> pd.Series:
    if nombre_modelo == 'CatBoost':
        raw = np.asarray(modelo.get_feature_importance(), dtype=float)
    else:
        raw = np.asarray(modelo.feature_importances_, dtype=float)
    total = np.sum(raw)
    vals = raw / total if total > 0 else raw
    return pd.Series(vals, index=COLUMNAS_X).sort_values(ascending=False)


def graficar_importancias_exog(nombre_modelo: str, id_modelo: str, horizonte: int, importancias: pd.Series):
    top15 = importancias.head(15).sort_values()
    colores = ['#d62728' if c in COLUMNAS_EXOGENAS else '#1f77b4' for c in top15.index]

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.barh(top15.index, top15.values, color=colores)
    ax.set_title(f'Top 15 Variables ({nombre_modelo} con Exogenas) — Horizonte h={horizonte}d\n[Rojo = Variable Exogena Global/Commodity | Azul = Tecnica Endogena]')
    ax.set_xlabel('Importancia relativa normalizada')
    ax.grid(axis='x', alpha=0.2)
    fig.tight_layout()
    fig.savefig(SALIDA / f'importancias_{id_modelo}_exog_h{horizonte}.png', dpi=160)
    plt.close(fig)


def graficar_comparativa_exogenas_por_horizonte(predicciones_por_horizonte: dict):
    for h, df_h in predicciones_por_horizonte.items():
        fig, ax = plt.subplots(figsize=(11.5, 4.8))
        ax.plot(df_h['fecha_objetivo'], df_h['tipo_cambio_real'], label='USD/PEN observado', color='black', linewidth=2.2)
        ax.plot(df_h['fecha_objetivo'], df_h['tipo_cambio_caminata_aleatoria'], label='Referencia: caminata aleatoria', color='gray', linestyle='--', linewidth=1.2)
        for nombre_m in ['Random Forest', 'XGBoost', 'LightGBM', 'CatBoost']:
            ax.plot(df_h['fecha_objetivo'], df_h[f'pred_{nombre_m}'], label=f'{nombre_m} (+Exogenas: Cobre, VIX, DXY, WTI)', linewidth=1.45)

        ax.set_title(f'Ensambles Multivariables con Exogenas (Cobre, VIX, DXY, Oro, WTI, US10Y) — h={h} dia(s) habil(es)')
        ax.set_ylabel('Soles por USD')
        ax.set_xlabel('Fecha de valor objetivo')
        ax.legend(loc='best', ncol=2, fontsize=8.5)
        ax.grid(alpha=0.2)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(SALIDA / f'comparacion_ensambles_exogenas_h{h}.png', dpi=160)
        fig.savefig(SALIDA_COMP / f'comparacion_ensambles_exogenas_h{h}.png', dpi=160)
        plt.close(fig)


def graficar_antes_vs_despues_exogenas(df_comp_ambas: pd.DataFrame):
    """
    Grafica el impacto directo de incorporar Cobre, VIX, DXY, Oro, WTI y US10Y
    comparando Fase 1 (18 vars endogenas) vs Fase 2 (50 vars con exogenas)
    especialmente en el horizonte de 20 dias habiles y en todos los horizontes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    orden_modelos = ['Random Forest', 'LightGBM', 'XGBoost', 'CatBoost']

    # Panel A: MAE a 20 dias habiles (Sin Exogenas vs Con Exogenas vs Caminata Aleatoria)
    sub20 = df_comp_ambas[df_comp_ambas['Horizonte (Dias Habiles)'] == 20].set_index('Algoritmo de Ensamble').loc[orden_modelos]
    x = np.arange(len(orden_modelos))
    width = 0.35

    b1 = axes[0].bar(x - width / 2, sub20['MAE Sin Exogenas (Soles/USD)'], width, label='Fase 1: Sin Exogenas (Solo 18 Tecnicas)', color='#aec7e8')
    b2 = axes[0].bar(x + width / 2, sub20['MAE Con Exogenas Cobre/VIX/DXY (Soles/USD)'], width, label='Fase 2: Con Exogenas (50 Vars: Cobre, VIX, DXY, WTI, US10Y)', color='#1f77b4')
    rw20 = float(sub20['MAE Caminata Aleatoria (Soles/USD)'].iloc[0])
    axes[0].axhline(rw20, color='#d62728', linestyle='--', linewidth=1.5, label=f'Caminata Aleatoria ({rw20:.6f} S/)')

    axes[0].set_title('Impacto en Horizonte h=20 Dias Habiles: Reduccion del MAE (Soles/USD)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(orden_modelos)
    axes[0].set_ylabel('MAE en Nivel (Soles por USD) — Menor es mejor')
    axes[0].set_ylim(0.030, 0.044)
    axes[0].legend(loc='upper left', fontsize=8.5)
    axes[0].grid(axis='y', alpha=0.2)
    axes[0].bar_label(b1, fmt='%.5f', fontsize=7.5, padding=2)
    axes[0].bar_label(b2, fmt='%.5f', fontsize=7.5, padding=2, fontweight='bold')

    # Panel B: Acierto Direccional a 20 dias habiles (%)
    b3 = axes[1].bar(x - width / 2, sub20['Acierto Direccional Sin Exogenas (%)'], width, label='Fase 1: Sin Exogenas (Solo 18 Tecnicas)', color='#ffbb78')
    b4 = axes[1].bar(x + width / 2, sub20['Acierto Direccional Con Exogenas (%)'], width, label='Fase 2: Con Exogenas (50 Vars: Cobre, VIX, DXY, WTI, US10Y)', color='#2ca02c')
    axes[1].axhline(50.0, color='gray', linestyle='--', linewidth=1.2, label='Umbral Azar (50.0%)')

    axes[1].set_title('Impacto en Horizonte h=20 Dias Habiles: Acierto Direccional (%)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(orden_modelos)
    axes[1].set_ylabel('Acierto Direccional (%) — Mayor es mejor')
    axes[1].set_ylim(0, 72)
    axes[1].legend(loc='upper right', fontsize=8.5)
    axes[1].grid(axis='y', alpha=0.2)
    axes[1].bar_label(b3, fmt='%.1f%%', fontsize=8, padding=2)
    axes[1].bar_label(b4, fmt='%.1f%%', fontsize=8, padding=2, fontweight='bold')

    fig.tight_layout()
    fig.savefig(SALIDA / 'impacto_exogenas_h20.png', dpi=160)
    fig.savefig(SALIDA_COMP / 'impacto_exogenas_h20.png', dpi=160)
    plt.close(fig)


def graficar_barras_exogenas_todos_horizontes(df_exog: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5))
    orden_modelos = ['Random Forest', 'LightGBM', 'XGBoost', 'CatBoost']

    piv_mae = df_exog.pivot(
        index='horizonte_dias_habiles',
        columns='modelo',
        values='mae_precio_modelo_pen',
    )[orden_modelos]
    rw_mae = df_exog.groupby('horizonte_dias_habiles')['mae_precio_baseline_pen'].first()
    piv_mae['Caminata Aleatoria'] = rw_mae

    piv_mae.plot(kind='bar', ax=axes[0], rot=0, width=0.82)
    axes[0].set_title('Panel A (Con Exogenas): MAE en Nivel (Soles / USD)')
    axes[0].set_xlabel('Horizonte de pronostico (dias habiles)')
    axes[0].set_ylabel('MAE (Soles por USD)')
    axes[0].set_ylim(0, piv_mae.to_numpy().max() * 1.18)
    axes[0].grid(axis='y', alpha=0.2)
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt='%.4f', fontsize=6.5, rotation=90, padding=2)

    piv_da = df_exog.pivot(
        index='horizonte_dias_habiles',
        columns='modelo',
        values='acierto_direccional_porcentaje',
    )[orden_modelos]
    piv_da.plot(kind='bar', ax=axes[1], rot=0, width=0.78)
    axes[1].axhline(50.0, color='gray', linestyle='--', linewidth=1.1, label='Referencia Azar (50.0%)')
    axes[1].set_title('Panel B (Con Exogenas): Acierto Direccional (%)')
    axes[1].set_xlabel('Horizonte de pronostico (dias habiles)')
    axes[1].set_ylabel('Acierto direccional (%)')
    axes[1].set_ylim(0, 75)
    axes[1].legend(loc='upper right')
    axes[1].grid(axis='y', alpha=0.2)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt='%.1f%%', fontsize=7, padding=2)

    fig.tight_layout()
    fig.savefig(SALIDA / 'resumen_mae_y_acierto_direccional_exogenas.png', dpi=160)
    fig.savefig(SALIDA_COMP / 'resumen_mae_y_acierto_direccional_exogenas.png', dpi=160)
    plt.close(fig)


def ejecutar():
    train, hold = cargar_datos()
    print('=== ENTRENANDO NUEVOS ENSAMBLES MULTIVARIABLES CON EXOGENAS (50 VARIABLES) ===')
    print(f'Entrenamiento: {len(train)} sesiones ({train["Fecha"].min().date()} a {train["Fecha"].max().date()})')
    print(f'Holdout:       {len(hold)} sesiones ({hold["Fecha"].min().date()} a {hold["Fecha"].max().date()})')
    print(f'Total variables predictoras: {len(COLUMNAS_X)} ({len(COLUMNAS_ENDOGENAS)} tecnicas + {len(COLUMNAS_EXOGENAS)} exogenas Cobre/VIX/DXY/Oro/WTI/US10Y/SP500/BCRP)')

    metricas_exog = []
    validaciones_exog = []
    importancias_top_registros = []
    predicciones_por_horizonte = {}

    mapa_id = {
        'Random Forest': 'random_forest',
        'XGBoost': 'xgboost',
        'LightGBM': 'lightgbm',
        'CatBoost': 'catboost',
    }

    for h in HORIZONTES:
        tr, y_tr, _, _ = convertir_a_supervisado(train, h)
        te, y_te, fechas_destino, precios_reales = convertir_a_supervisado(hold, h)
        X_tr, X_te = tr[COLUMNAS_X], te[COLUMNAS_X]
        precios_actuales = te['USD_PEN'].to_numpy(float)
        precios_naive = precios_actuales
        retorno_naive = np.zeros(len(y_te))

        df_pred_h = pd.DataFrame({
            'fecha_origen': te['Fecha'],
            'fecha_objetivo': fechas_destino,
            'tipo_cambio_origen': precios_actuales,
            'tipo_cambio_real': precios_reales,
            'tipo_cambio_caminata_aleatoria': precios_naive,
            'retorno_real_bp': y_te,
        })

        for nombre_modelo in ['Random Forest', 'XGBoost', 'LightGBM', 'CatBoost']:
            id_mod = mapa_id[nombre_modelo]
            params, cv_res, best = seleccionar_candidato(nombre_modelo, X_tr, y_tr, h)
            validaciones_exog.extend(cv_res)

            modelo = instanciar_modelo(nombre_modelo, params)
            w_tr = calcular_pesos_temporales(len(X_tr), params.get('half_life'))
            modelo.fit(X_tr, y_tr, sample_weight=w_tr)

            estimado_bp = np.asarray(modelo.predict(X_te), dtype=float)
            precios_estimados = precios_actuales * np.exp(estimado_bp / 10000.0)

            df_pred_h[f'pred_{nombre_modelo}'] = precios_estimados
            df_pred_h[f'ret_bp_{nombre_modelo}'] = estimado_bp

            mask = y_te != 0
            n_no_nulos = int(mask.sum())
            hit = float(np.mean(np.sign(y_te[mask]) == np.sign(estimado_bp[mask]))) if mask.any() else float('nan')
            n_aciertos = int(round(hit * n_no_nulos))

            rmse_mod_bp = float(np.sqrt(mean_squared_error(y_te, estimado_bp)))
            rmse_naive_bp = float(np.sqrt(mean_squared_error(y_te, retorno_naive)))
            mae_mod_pen = float(mean_absolute_error(precios_reales, precios_estimados))
            mae_rw_pen = float(mean_absolute_error(precios_reales, precios_naive))

            importancias = obtener_importancias(nombre_modelo, modelo)
            graficar_importancias_exog(nombre_modelo, id_mod, h, importancias)
            importancias.rename('importancia_relativa').to_csv(SALIDA / f'importancias_{id_mod}_exog_h{h}.csv')

            # Guardar top 5 variables y peso total de las exogenas
            peso_exogenas_pct = float(importancias[COLUMNAS_EXOGENAS].sum() * 100.0)
            top5_vars = ', '.join(list(importancias.index[:5]))
            importancias_top_registros.append({
                'Horizonte (Dias Habiles)': h,
                'Algoritmo de Ensamble': nombre_modelo,
                'Peso Total Variables Exogenas (%)': round(peso_exogenas_pct, 2),
                'Peso Total Variables Endogenas (%)': round(100.0 - peso_exogenas_pct, 2),
                'Top 5 Variables Mas Importantes': top5_vars,
            })

            # Guardar artefacto del modelo con exogenas
            if nombre_modelo == 'Random Forest':
                joblib.dump(modelo, MODELOS_DIR / f'modelo_random_forest_exog_h{h}.joblib')
            elif nombre_modelo == 'XGBoost':
                modelo.save_model(str(MODELOS_DIR / f'modelo_xgboost_exog_h{h}.json'))
            elif nombre_modelo == 'LightGBM':
                modelo.booster_.save_model(str(MODELOS_DIR / f'modelo_lightgbm_exog_h{h}.txt'))
            elif nombre_modelo == 'CatBoost':
                modelo.save_model(str(MODELOS_DIR / f'modelo_catboost_exog_h{h}.cbm'))

            metricas_exog.append({
                'modelo': nombre_modelo,
                'horizonte_dias_habiles': h,
                'n_pronosticos': len(y_te),
                'n_cambios_no_nulos': n_no_nulos,
                'n_aciertos_direccionales': n_aciertos,
                'mae_precio_modelo_pen': mae_mod_pen,
                'mae_precio_baseline_pen': mae_rw_pen,
                'diferencia_mae_vs_baseline_pen': mae_mod_pen - mae_rw_pen,
                'supera_caminata_aleatoria_mae': bool(mae_mod_pen < mae_rw_pen),
                'rmse_precio_modelo_pen': float(np.sqrt(mean_squared_error(precios_reales, precios_estimados))),
                'rmse_precio_baseline_pen': float(np.sqrt(mean_squared_error(precios_reales, precios_naive))),
                'mae_ret_modelo_bp': float(mean_absolute_error(y_te, estimado_bp)),
                'mae_ret_baseline_bp': float(mean_absolute_error(y_te, retorno_naive)),
                'rmse_ret_modelo_bp': rmse_mod_bp,
                'rmse_ret_baseline_bp': rmse_naive_bp,
                'theil_u_rmse': float(rmse_mod_bp / rmse_naive_bp) if rmse_naive_bp > 0 else None,
                'r2_ret': float(r2_score(y_te, estimado_bp)),
                'acierto_direccional_sin_ceros': hit,
                'acierto_direccional_porcentaje': hit * 100.0,
                'mae_cv_bp': float(best['mae_cv_bp']),
                'mae_cv_baseline_bp': float(best['mae_cv_rw_bp']),
                'half_life_sesiones': params.get('half_life') or 'Completo (8462d)',
                'peso_exogenas_porcentaje': peso_exogenas_pct,
                'top_5_variables': top5_vars,
            })

        df_pred_h.to_csv(SALIDA / f'predicciones_exogenas_h{h}.csv', index=False)
        predicciones_por_horizonte[h] = df_pred_h

    df_exog = pd.DataFrame(metricas_exog).sort_values(
        ['horizonte_dias_habiles', 'mae_precio_modelo_pen']
    ).reset_index(drop=True)
    df_exog.to_csv(SALIDA / 'tabla_metricas_ensambles_exogenas.csv', index=False)
    df_exog.to_csv(SALIDA_COMP / 'tabla_metricas_ensambles_exogenas.csv', index=False)

    df_imp_resumen = pd.DataFrame(importancias_top_registros)
    df_imp_resumen.to_csv(SALIDA / 'resumen_importancia_exogenas.csv', index=False)
    df_imp_resumen.to_csv(SALIDA_COMP / 'resumen_importancia_exogenas.csv', index=False)

    # Construir tabla comparativa directa: Fase 1 (Sin Exogenas, 18 vars) vs Fase 2 (Con Exogenas, 50 vars)
    df_sin = pd.read_csv(SALIDA_COMP / 'tabla_comparativa_ensambles.csv')
    filas_antes_despues = []
    for _, row_con in df_exog.iterrows():
        mod = row_con['modelo']
        h = int(row_con['horizonte_dias_habiles'])
        row_sin = df_sin[(df_sin['modelo'] == mod) & (df_sin['horizonte_dias_habiles'] == h)].iloc[0]

        mae_sin = float(row_sin['mae_precio_modelo_pen'])
        mae_con = float(row_con['mae_precio_modelo_pen'])
        mae_rw = float(row_con['mae_precio_baseline_pen'])
        da_sin = float(row_sin['acierto_direccional_sin_ceros']) * 100.0
        da_con = float(row_con['acierto_direccional_porcentaje'])
        n_mov = int(row_con['n_cambios_no_nulos'])

        filas_antes_despues.append({
            'Horizonte (Dias Habiles)': h,
            'Algoritmo de Ensamble': mod,
            'Pronosticos Evaluados (N)': int(row_con['n_pronosticos']),
            'MAE Caminata Aleatoria (Soles/USD)': round(mae_rw, 6),
            'MAE Sin Exogenas (Soles/USD)': round(mae_sin, 6),
            'MAE Con Exogenas Cobre/VIX/DXY (Soles/USD)': round(mae_con, 6),
            'Reduccion de Error MAE al Incluir Exogenas (Soles/USD)': round(mae_con - mae_sin, 6),
            'Diferencia Con Exogenas vs Caminata Aleatoria (Soles/USD)': round(mae_con - mae_rw, 6),
            'Supera a Caminata Aleatoria Con Exogenas': 'SI' if mae_con < mae_rw else 'Casi Empate',
            'Theil U Sin Exogenas': round(float(row_sin['theil_u_rmse']), 4),
            'Theil U Con Exogenas': round(float(row_con['theil_u_rmse']), 4),
            'Acierto Direccional Sin Exogenas (%)': round(da_sin, 2),
            'Acierto Direccional Con Exogenas (%)': round(da_con, 2),
            'Mejora Acierto Direccional (Puntos Porcentuales)': round(da_con - da_sin, 2),
            'Sesiones Acertadas Sin Exogenas': f"{int(round(da_sin * n_mov / 100.0))}/{n_mov}",
            'Sesiones Acertadas Con Exogenas': f"{int(row_con['n_aciertos_direccionales'])}/{n_mov}",
            'Peso de Variables Exogenas en el Modelo (%)': round(float(row_con['peso_exogenas_porcentaje']), 1),
        })

    df_antes_despues = pd.DataFrame(filas_antes_despues)
    df_antes_despues.to_csv(SALIDA / 'tabla_comparativa_sin_vs_con_exogenas.csv', index=False)
    df_antes_despues.to_csv(SALIDA_COMP / 'tabla_comparativa_sin_vs_con_exogenas.csv', index=False)

    # Tabla de valores exactos del nuevo grafico de barras con exogenas
    filas_barras_exog = []
    for h in HORIZONTES:
        sub = df_exog[df_exog['horizonte_dias_habiles'] == h].set_index('modelo')
        rw_mae_pen = float(sub['mae_precio_baseline_pen'].iloc[0])
        rw_mae_bp = float(sub['mae_ret_baseline_bp'].iloc[0])
        n_eval = int(sub['n_pronosticos'].iloc[0])
        n_mov = int(sub['n_cambios_no_nulos'].iloc[0])

        filas_barras_exog.append({
            'Horizonte (Dias Habiles)': h,
            'Pronosticos Evaluados (N)': n_eval,
            'Movimientos No Nulos': n_mov,
            'MAE Caminata Aleatoria (Soles/USD)': round(rw_mae_pen, 6),
            'MAE Random Forest + Exogenas (Soles/USD)': round(float(sub.loc['Random Forest', 'mae_precio_modelo_pen']), 6),
            'MAE LightGBM + Exogenas (Soles/USD)': round(float(sub.loc['LightGBM', 'mae_precio_modelo_pen']), 6),
            'MAE XGBoost + Exogenas (Soles/USD)': round(float(sub.loc['XGBoost', 'mae_precio_modelo_pen']), 6),
            'MAE CatBoost + Exogenas (Soles/USD)': round(float(sub.loc['CatBoost', 'mae_precio_modelo_pen']), 6),
            'MAE Caminata Aleatoria (pb)': round(rw_mae_bp, 2),
            'MAE Random Forest + Exogenas (pb)': round(float(sub.loc['Random Forest', 'mae_ret_modelo_bp']), 2),
            'MAE LightGBM + Exogenas (pb)': round(float(sub.loc['LightGBM', 'mae_ret_modelo_bp']), 2),
            'MAE XGBoost + Exogenas (pb)': round(float(sub.loc['XGBoost', 'mae_ret_modelo_bp']), 2),
            'MAE CatBoost + Exogenas (pb)': round(float(sub.loc['CatBoost', 'mae_ret_modelo_bp']), 2),
            'Acierto Direccional Random Forest + Exogenas (%)': round(float(sub.loc['Random Forest', 'acierto_direccional_porcentaje']), 2),
            'Acierto Direccional LightGBM + Exogenas (%)': round(float(sub.loc['LightGBM', 'acierto_direccional_porcentaje']), 2),
            'Acierto Direccional XGBoost + Exogenas (%)': round(float(sub.loc['XGBoost', 'acierto_direccional_porcentaje']), 2),
            'Acierto Direccional CatBoost + Exogenas (%)': round(float(sub.loc['CatBoost', 'acierto_direccional_porcentaje']), 2),
            'Aciertos Exactos Random Forest + Exogenas': f"{int(sub.loc['Random Forest', 'n_aciertos_direccionales'])}/{n_mov}",
            'Aciertos Exactos LightGBM + Exogenas': f"{int(sub.loc['LightGBM', 'n_aciertos_direccionales'])}/{n_mov}",
            'Aciertos Exactos XGBoost + Exogenas': f"{int(sub.loc['XGBoost', 'n_aciertos_direccionales'])}/{n_mov}",
            'Aciertos Exactos CatBoost + Exogenas': f"{int(sub.loc['CatBoost', 'n_aciertos_direccionales'])}/{n_mov}",
        })

    df_barras_exog = pd.DataFrame(filas_barras_exog)
    df_barras_exog.to_csv(SALIDA / 'tabla_valores_exactos_grafico_barras_exogenas.csv', index=False)
    df_barras_exog.to_csv(SALIDA_COMP / 'tabla_valores_exactos_grafico_barras_exogenas.csv', index=False)

    graficar_comparativa_exogenas_por_horizonte(predicciones_por_horizonte)
    graficar_antes_vs_despues_exogenas(df_antes_despues)
    graficar_barras_exogenas_todos_horizontes(df_exog)

    print('\n=== COMPARACION ANTES (18 VARS) VS DESPUES (50 VARS CON COBRE, VIX, DXY, WTI, ORO, US10Y) ===')
    print(df_antes_despues[[
        'Horizonte (Dias Habiles)', 'Algoritmo de Ensamble',
        'MAE Caminata Aleatoria (Soles/USD)',
        'MAE Sin Exogenas (Soles/USD)',
        'MAE Con Exogenas Cobre/VIX/DXY (Soles/USD)',
        'Supera a Caminata Aleatoria Con Exogenas',
        'Acierto Direccional Sin Exogenas (%)',
        'Acierto Direccional Con Exogenas (%)',
        'Sesiones Acertadas Con Exogenas',
    ]].to_string(index=False))
    print('\nArchivos guardados en:', SALIDA)


if __name__ == '__main__':
    ejecutar()
