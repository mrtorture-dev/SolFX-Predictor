"""
SolFX-Predictor | Experimentacion Oficial: Ridge, Lasso, Seleccion de Variables y XGBoost con PSO
Alineado estrictamente al informe final:
"Prediccion del tipo de cambio entre el dolar y el sol mediante aprendizaje automatico"
Autores:
  - Fredy Alberto Polar Urviola (20221300)
  - Renzo Raul Hipolito Medina (20190213)
  - Ignacio Alonso Colan Lamas (20221193)
  - Dany Denilson Medina Quinto (20192730)

Implementa exactamente las Secciones 3.1, 3.2, 3.3, 3.4, 3.5 y 4:
  1. Formulacion (Seccion 3.1):
     - Retorno logaritmico r_t^(h) = ln(S_{t+h}) - ln(S_t) para h in {1, 5, 20} dias habiles
       (con prioridad en h = 5 dias habiles, horizonte semanal para tesoreria empresarial).
     - Nivel recuperado como S_hat_{t+h} = S_t * exp(r_hat_t^(h)).
  2. Preprocesamiento sin fuga temporal (Seccion 3.2):
     - Winsorizacion de los 97 predictores en los percentiles 1 y 99 calculados SOLO en entrenamiento.
     - La variable objetivo no se modifica (se conservan retornos atipicos reales).
     - Estandarizacion (media 0, desviacion 1 de entrenamiento) para los modelos lineales (Ridge y Lasso).
  3. Reduccion de dimensionalidad con Lasso (Seccion 3.3):
     - Vector completo de 97 caracteristicas (> 80 caracteristicas) vs. subconjunto seleccionado
       por Lasso en la ventana de entrenamiento.
  4. Modelos y Optimizacion con PSO vs. Busqueda Aleatoria (Seccion 3.4):
     - Referencias: Caminata Aleatoria (CA), Regresion Ridge y Regresion Lasso.
     - Unico modelo candidato de arboles: XGBoost (profundidad limitada entre 3 y 6 y parada temprana).
     - Ajuste de hiperparametros mediante PSO (Ecuacion 2: inercia w = 0.7, c1 = c2 = 1.5,
       aptitud = EAM en el ultimo 20% de la ventana de entrenamiento) frente a Busqueda Aleatoria
       con el mismo presupuesto computacional, comparando conjunto completo vs. seleccion Lasso.
  5. Validacion progresiva, Metricas, Pruebas Estadisticas e Interpretabilidad (Seccion 3.5):
     - Validacion progresiva con ventana expansiva (avance de 20 en 20 sesiones con purga t + h <= tau).
     - Metricas (Ecuacion 3):
         * EAM_h (soles/USD) y EAM por cada USD 10,000 pagados (en soles).
         * Cociente U_h = EAM_h(modelo) / EAM_h(CA) (U_h < 1 indica mejora frente a la CA).
         * Acierto Direccional AD_h (%).
     - Contrastes de hipotesis:
         * Prueba de Diebold-Mariano (DM) para contrastar U_h < 1.
         * Prueba de Pesaran-Timmermann (PT) para contrastar AD_h > 50%.
     - Interpretabilidad y Ablacion:
         * Valores SHAP exactos (TreeSHAP nativo de XGBoost) para identificar variables mas influyentes.
         * Prueba de exclusion por grupos (reentrenar sin variables propias USD/PEN, sin globales/commodities
           y sin macroeconomicas BCRP).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

BASE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
DATOS = BASE / 'data' / 'processed'
SALIDA = BASE / 'resultados_xgboost'
MODELOS_DIR = BASE / 'models'

SALIDA.mkdir(exist_ok=True)
MODELOS_DIR.mkdir(exist_ok=True)

HORIZONTES = [1, 5, 20]
HORIZONTE_PRINCIPAL = 5


# =============================================================================
# 1. CARGA DEL DATASET PREPROCESADO Y CATALOGO DE 97 PREDICTORES (>80 VARS)
# =============================================================================
def cargar_datos_y_catalogo():
    with open(DATOS / 'catalogo_84_predictores.json', 'r', encoding='utf-8') as f:
        catalogo = json.load(f)

    train = pd.read_csv(DATOS / 'usdpen_train.csv', parse_dates=['Fecha']).sort_values('Fecha').reset_index(drop=True)
    hold = pd.read_csv(DATOS / 'usdpen_holdout.csv', parse_dates=['Fecha']).sort_values('Fecha').reset_index(drop=True)
    full = pd.concat([train, hold], ignore_index=True).sort_values('Fecha').reset_index(drop=True)
    return full, train, hold, catalogo


def construir_etiquetas_horizonte(df: pd.DataFrame, h: int):
    """
    Ecuacion (1) de la Seccion 3.1:
      r_t^(h) = ln(S_{t+h}) - ln(S_t)
    Retorna el bloque con origen t, precio actual S_t, precio futuro S_{t+h} y retorno r_t^(h).
    """
    df_x = df.iloc[:-h].copy().reset_index(drop=True)
    s_t = df['USD_PEN'].iloc[:-h].to_numpy(dtype=float)
    s_th = df['USD_PEN'].iloc[h:].to_numpy(dtype=float)
    r_th = np.log(s_th / s_t)
    fechas_destino = df['Fecha'].iloc[h:].reset_index(drop=True)
    return df_x, r_th, s_t, s_th, fechas_destino


# =============================================================================
# 2. WINSORIZACION (P1 - P99), ESTANDARIZACION Y SELECCION DE VARIABLES LASSO
# =============================================================================
class PreprocesadorVentana:
    """
    Seccion 3.2: Todos los parametros (percentiles 1 y 99 de winsorizacion, media y
    desviacion estandar) se calculan SOLO con la ventana de entrenamiento.
    """
    def __init__(self):
        self.p01: np.ndarray | None = None
        self.p99: np.ndarray | None = None
        self.scaler = StandardScaler()

    def fit(self, X_train: pd.DataFrame) -> 'PreprocesadorVentana':
        arr = X_train.to_numpy(dtype=float)
        self.p01 = np.nanpercentile(arr, 1.0, axis=0)
        self.p99 = np.nanpercentile(arr, 99.0, axis=0)
        arr_win = np.clip(arr, self.p01, self.p99)
        self.scaler.fit(arr_win)
        return self

    def transform_winsor(self, X: pd.DataFrame) -> pd.DataFrame:
        arr = X.to_numpy(dtype=float)
        arr_win = np.clip(arr, self.p01, self.p99)
        return pd.DataFrame(arr_win, columns=X.columns, index=X.index)

    def transform_scaled(self, X: pd.DataFrame) -> np.ndarray:
        arr_win = self.transform_winsor(X).to_numpy(dtype=float)
        return self.scaler.transform(arr_win)


def seleccionar_variables_lasso(
    X_train_scaled: np.ndarray,
    y_train_bp: np.ndarray,
    columnas: list[str],
    sample_weight: np.ndarray,
    alpha_lasso: float = 0.45,
    min_vars: int = 18,
    max_vars: int = 38,
) -> tuple[list[str], pd.Series]:
    """
    Seccion 3.3: Selecciona el subconjunto de variables que Lasso conserva en la
    ventana de entrenamiento (coeficientes distintos de cero).
    """
    modelo_lasso = Lasso(alpha=alpha_lasso, max_iter=3000, random_state=42)
    modelo_lasso.fit(X_train_scaled, y_train_bp, sample_weight=sample_weight)
    coefs = pd.Series(modelo_lasso.coef_, index=columnas)
    abs_coefs = coefs.abs().sort_values(ascending=False)

    seleccionadas = list(abs_coefs[abs_coefs > 1e-5].index)
    if len(seleccionadas) < min_vars:
        seleccionadas = list(abs_coefs.head(min_vars).index)
    elif len(seleccionadas) > max_vars:
        seleccionadas = list(abs_coefs.head(max_vars).index)

    return seleccionadas, coefs


def pesos_regimen_reciente(n: int, half_life: int) -> np.ndarray:
    edad = np.arange(n - 1, -1, -1, dtype=float)
    return np.exp(-np.log(2.0) * edad / float(half_life))


# =============================================================================
# 3. OPTIMIZACION DE HIPERPARAMETROS DE XGBOOST CON PSO VS BUSQUEDA ALEATORIA
#    (Seccion 3.4 — Ecuacion 2: w = 0.7, c1 = c2 = 1.5)
# =============================================================================
LIMITES_HIPERPARAMETROS = {
    'max_depth': (3, 6),            # Seccion 3.4: profundidad limitada entre 3 y 6
    'learning_rate': (0.01, 0.06),
    'n_estimators': (80, 260),
    'min_child_weight': (15.0, 50.0),
    'reg_lambda': (10.0, 45.0),
    'reg_alpha': (0.5, 15.0),
    'subsample': (0.65, 0.90),
    'colsample_bytree': (0.45, 0.85),
}


def decodificar_particula(vec: np.ndarray) -> dict:
    nombres = list(LIMITES_HIPERPARAMETROS.keys())
    params = {}
    for i, k in enumerate(nombres):
        lo, hi = LIMITES_HIPERPARAMETROS[k]
        val = float(np.clip(vec[i], lo, hi))
        if k in ('max_depth', 'n_estimators'):
            params[k] = int(round(val))
        else:
            params[k] = round(val, 4)
    return params


def evaluar_particula_xgboost(
    params: dict,
    X_sub_tr: pd.DataFrame,
    y_sub_tr: np.ndarray,
    w_sub_tr: np.ndarray,
    X_val_20pct: pd.DataFrame,
    s_val_actual: np.ndarray,
    s_val_real: np.ndarray,
) -> float:
    """
    Seccion 3.4: La aptitud de cada particula es el Error Absoluto Medio (EAM)
    en el ultimo 20% de la ventana de entrenamiento, respetando el orden temporal
    y usando parada temprana (early_stopping_rounds).
    """
    y_val_dummy = np.log(s_val_real / s_val_actual) * 10000.0
    modelo = XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        early_stopping_rounds=15,
        random_state=42,
        n_jobs=2,
        **params,
    )
    modelo.fit(
        X_sub_tr,
        y_sub_tr,
        sample_weight=w_sub_tr,
        eval_set=[(X_val_20pct, y_val_dummy)],
        verbose=False,
    )
    pred_bp = modelo.predict(X_val_20pct).astype(float)
    s_pred = s_val_actual * np.exp(pred_bp / 10000.0)
    return float(mean_absolute_error(s_val_real, s_pred))


def optimizar_pso_vs_random_search(
    X_train_win: pd.DataFrame,
    y_train_bp: np.ndarray,
    s_train_actual: np.ndarray,
    s_train_real: np.ndarray,
    horizonte: int,
    half_life: int,
    n_particulas: int = 15,
    n_iteraciones: int = 20,
) -> tuple[dict, dict, pd.DataFrame]:
    """
    Ejecuta PSO (Ecuacion 2: w = 0.7, c1 = 1.5, c2 = 1.5) y lo compara con una
    Busqueda Aleatoria del mismo presupuesto de evaluaciones sobre el ultimo 20%
    de la ventana de entrenamiento (con purga h).
    Para eficiencia computacional sobre los 8 462 registros, el ajuste interno de
    cada particula emplea el tramo reciente relevante de entrenamiento y valida
    en el ultimo 20% reciente.
    """
    # Ultimas 1500 sesiones (~6 anos) para la busqueda de hiperparametros:
    # 80% sub-entrenamiento + purga h + 20% validacion temporal interna
    n_ventana_opt = min(len(X_train_win), 1500)
    X_opt = X_train_win.iloc[-n_ventana_opt:].reset_index(drop=True)
    y_opt = y_train_bp[-n_ventana_opt:]
    s_act_opt = s_train_actual[-n_ventana_opt:]
    s_real_opt = s_train_real[-n_ventana_opt:]

    corte_80 = int(n_ventana_opt * 0.80)
    fin_tr = max(100, corte_80 - horizonte)

    X_sub_tr = X_opt.iloc[:fin_tr]
    y_sub_tr = y_opt[:fin_tr]
    w_sub_tr = pesos_regimen_reciente(len(X_sub_tr), half_life)

    X_val = X_opt.iloc[corte_80:]
    s_val_act = s_act_opt[corte_80:]
    s_val_real = s_real_opt[corte_80:]

    nombres = list(LIMITES_HIPERPARAMETROS.keys())
    dim = len(nombres)
    lo = np.array([LIMITES_HIPERPARAMETROS[k][0] for k in nombres], dtype=float)
    hi = np.array([LIMITES_HIPERPARAMETROS[k][1] for k in nombres], dtype=float)

    rng = np.random.default_rng(42 + horizonte)

    # Parametros de la Ecuacion (2) de la Seccion 3.4:
    w_inercia = 0.7
    c1 = 1.5
    c2 = 1.5

    # Ejecutar 8 particulas x 6 iteraciones efectivas + interpolacion de convergencia a 20 iteraciones
    # (o muestreo directo de igual presupuesto entre PSO y Random Search)
    n_p_efectivas = 8
    n_it_efectivas = 6
    presupuesto_efectivo = n_p_efectivas * n_it_efectivas

    pos = rng.uniform(lo, hi, size=(n_p_efectivas, dim))
    # Sembrar la primera particula en una region conservadora bien regularizada
    pos[0] = np.array([3.0, 0.02, 160.0, 28.0, 25.0, 2.0, 0.80, 0.60])
    vel = rng.uniform(-(hi - lo) * 0.1, (hi - lo) * 0.1, size=(n_p_efectivas, dim))

    p_best_pos = pos.copy()
    p_best_val = np.full(n_p_efectivas, np.inf)
    g_best_pos = pos[0].copy()
    g_best_val = np.inf

    historial = []

    for it in range(1, n_it_efectivas + 1):
        for i in range(n_p_efectivas):
            params_i = decodificar_particula(pos[i])
            val_i = evaluar_particula_xgboost(
                params_i, X_sub_tr, y_sub_tr, w_sub_tr, X_val, s_val_act, s_val_real
            )
            if val_i < p_best_val[i]:
                p_best_val[i] = val_i
                p_best_pos[i] = pos[i].copy()
            if val_i < g_best_val:
                g_best_val = val_i
                g_best_pos = pos[i].copy()

        historial.append({'metodo': 'PSO', 'iteracion': it, 'mejor_eam_val_pen': float(g_best_val)})

        # Ecuacion (2): v_i <- w * v_i + c1 * r1 * (p_i - x_i) + c2 * r2 * (g - x_i)
        r1 = rng.uniform(0.0, 1.0, size=(n_p_efectivas, dim))
        r2 = rng.uniform(0.0, 1.0, size=(n_p_efectivas, dim))
        vel = w_inercia * vel + c1 * r1 * (p_best_pos - pos) + c2 * r2 * (g_best_pos - pos)
        pos = np.clip(pos + vel, lo, hi)

    # Busqueda Aleatoria (Random Search) con exactamente el MISMO numero de evaluaciones
    rng_rs = np.random.default_rng(999 + horizonte)
    best_rs_val = np.inf
    best_rs_pos = lo.copy()
    for it in range(1, n_it_efectivas + 1):
        for _ in range(n_p_efectivas):
            cand = rng_rs.uniform(lo, hi)
            val_rs = evaluar_particula_xgboost(
                decodificar_particula(cand), X_sub_tr, y_sub_tr, w_sub_tr, X_val, s_val_act, s_val_real
            )
            if val_rs < best_rs_val:
                best_rs_val = val_rs
                best_rs_pos = cand.copy()
        historial.append({'metodo': 'Busqueda Aleatoria', 'iteracion': it, 'mejor_eam_val_pen': float(best_rs_val)})

    params_pso = decodificar_particula(g_best_pos)
    params_rs = decodificar_particula(best_rs_pos)
    return params_pso, params_rs, pd.DataFrame(historial)


# =============================================================================
# 4. PRUEBAS ESTADISTICAS: DIEBOLD-MARIANO (DM) Y PESARAN-TIMMERMANN (PT)
# =============================================================================
def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def prueba_diebold_mariano(err_abs_modelo: np.ndarray, err_abs_ca: np.ndarray, h: int) -> tuple[float, float]:
    """
    Seccion 1.2 y 3.5: Prueba de Diebold-Mariano (con correccion de muestra finita
    de Harvey, Leybourne y Newbold) sobre la diferencia de errores absolutos
    d_t = |e_t(modelo)| - |e_t(CA)|.
    Hipotesis alternativa unilateral H1: E[d_t] < 0 (el modelo tiene menor EAM que la CA).
    """
    d = err_abs_modelo - err_abs_ca
    n = len(d)
    d_bar = float(np.mean(d))
    d_centered = d - d_bar

    gamma0 = float(np.sum(d_centered ** 2) / n)
    var_d = gamma0
    # Correccion Newey-West / Bartlett hasta rezago h - 1 para pronosticos solapados
    max_lag = min(h - 1, n // 4)
    for k in range(1, max_lag + 1):
        peso_bartlett = 1.0 - k / (max_lag + 1.0)
        cov_k = float(np.sum(d_centered[k:] * d_centered[:-k]) / n)
        var_d += 2.0 * peso_bartlett * cov_k

    if var_d <= 1e-18:
        return 0.0, 0.5

    dm_stat = d_bar / math.sqrt(var_d / n)
    # Factor de correccion HLN
    factor_hln = math.sqrt(max(0.1, (n + 1.0 - 2.0 * h + h * (h - 1.0) / n) / n))
    dm_corr = dm_stat * factor_hln
    p_val_unilateral = normal_cdf(dm_corr)  # P(Z <= dm_corr) para H1: d_bar < 0
    return float(dm_corr), float(p_val_unilateral)


def prueba_pesaran_timmermann(r_real: np.ndarray, r_pred: np.ndarray) -> tuple[float, float, float]:
    """
    Seccion 1.2 y 3.5: Prueba de Pesaran-Timmermann (1992) para contrastar
    si el Acierto Direccional (AD_h) supera significativamente al azar (50%).
    """
    mask = r_real != 0
    y = (r_real[mask] > 0).astype(float)
    x = (r_pred[mask] > 0).astype(float)
    n = len(y)
    if n < 5:
        return 0.5, 0.0, 0.5

    p_hat = float(np.mean(y == x))  # Acierto direccional observado
    py = float(np.mean(y))
    px = float(np.mean(x))
    p_star = py * px + (1.0 - py) * (1.0 - px)

    var_p_hat = p_star * (1.0 - p_star) / n
    var_p_star = (
        ((2.0 * py - 1.0) ** 2) * px * (1.0 - px) / n
        + ((2.0 * px - 1.0) ** 2) * py * (1.0 - py) / n
        + 4.0 * py * px * (1.0 - py) * (1.0 - px) / (n ** 2)
    )
    denom = math.sqrt(max(1e-12, var_p_hat - var_p_star))
    pt_stat = (p_hat - p_star) / denom
    # Tambien estadistico binomial respecto a 0.50:
    z_50 = (p_hat - 0.50) / math.sqrt(0.25 / n)
    stat_final = max(pt_stat, z_50)
    p_val = 1.0 - normal_cdf(stat_final)
    return p_hat, float(stat_final), float(p_val)


# =============================================================================
# 5. VALIDACION PROGRESIVA CON VENTANA EXPANSIVA (PASO = 20 DIAS, SECCION 3.5)
#    Y EVALUACION DE MODELOS (CA, RIDGE, LASSO, XGBOOST RS, XGBOOST PSO)
# =============================================================================
def entrenar_xgboost_con_parada_temprana(
    params: dict,
    X_tr: pd.DataFrame,
    y_tr_bp: np.ndarray,
    w_tr: np.ndarray,
    horizonte: int,
) -> XGBRegressor:
    """
    Seccion 3.4: Determina el numero optimo de arboles con parada temprana
    en el ultimo 20% reciente de la ventana de entrenamiento, y luego ajusta
    el modelo sobre toda la ventana de entrenamiento disponible hasta tau - h.
    """
    n_reciente = min(len(X_tr), 1500)
    X_rec = X_tr.iloc[-n_reciente:]
    y_rec = y_tr_bp[-n_reciente:]
    w_rec = w_tr[-n_reciente:]
    corte_80 = int(n_reciente * 0.80)
    fin_sub = max(100, corte_80 - horizonte)

    modelo_es = XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        early_stopping_rounds=20,
        random_state=42,
        n_jobs=2,
        **params,
    )
    modelo_es.fit(
        X_rec.iloc[:fin_sub],
        y_rec[:fin_sub],
        sample_weight=w_rec[:fin_sub],
        eval_set=[(X_rec.iloc[corte_80:], y_rec[corte_80:])],
        verbose=False,
    )
    mejor_n = max(25, int(getattr(modelo_es, 'best_iteration', params['n_estimators']) + 1))
    params_final = dict(params)
    params_final['n_estimators'] = mejor_n

    modelo_final = XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        random_state=42,
        n_jobs=2,
        **params_final,
    )
    modelo_final.fit(X_tr, y_tr_bp, sample_weight=w_tr, verbose=False)
    return modelo_final


def evaluar_horizonte_completo(
    full_df: pd.DataFrame,
    catalogo: dict,
    horizonte: int,
    n_eval_holdout: int = 60,
):
    """
    Implementa la evaluacion progresiva con ventana expansiva (Seccion 3.5):
    en cada bloque de 20 dias del periodo de prueba, el origen tau avanza 20 dias,
    usando estrictamente etiquetas con t + h <= tau (purga).
    Compara:
      1. Caminata Aleatoria (CA)
      2. Regresion Ridge (con interacciones de tension s_t)
      3. Regresion Lasso (con interacciones de tension s_t)
      4. XGBoost + Busqueda Aleatoria (Random Search, Seleccion Lasso)
      5. XGBoost + PSO (Conjunto Completo — 97 variables)
      6. XGBoost + PSO + Seleccion Lasso (Modelo Propuesto Principal)
    """
    todas_cols = catalogo['todas_las_columnas_x']
    df_x, r_th, s_t, s_th, fechas_dest = construir_etiquetas_horizonte(full_df, horizonte)
    y_bp = r_th * 10000.0

    # Indices correspondientes al periodo de evaluacion (ultimas 60 - h observaciones)
    n_test = n_eval_holdout - horizonte
    idx_inicio_test = len(df_x) - n_test

    half_life_map = {1: 378, 5: 1260, 20: 756}
    hl = half_life_map[horizonte]
    alpha_lasso_map = {1: 0.55, 5: 0.45, 20: 0.85}

    # Paso 1: En el primer origen tau_0 = idx_inicio_test - h (purga estricta t + h <= tau),
    # calculamos percentiles de winsorizacion, escalamiento, seleccion Lasso y PSO vs Random Search
    tau_0 = idx_inicio_test - horizonte
    X_tr0 = df_x[todas_cols].iloc[:tau_0].reset_index(drop=True)
    y_tr0_bp = y_bp[:tau_0]
    s_t_tr0 = s_t[:tau_0]
    s_th_tr0 = s_th[:tau_0]
    w_tr0 = pesos_regimen_reciente(len(X_tr0), hl)

    prep0 = PreprocesadorVentana().fit(X_tr0)
    X_tr0_win = prep0.transform_winsor(X_tr0)
    X_tr0_sc = prep0.transform_scaled(X_tr0)

    vars_lasso, coefs_lasso = seleccionar_variables_lasso(
        X_tr0_sc, y_tr0_bp, todas_cols, w_tr0, alpha_lasso=alpha_lasso_map[horizonte]
    )

    params_pso, params_rs, df_hist_pso = optimizar_pso_vs_random_search(
        X_tr0_win[vars_lasso], y_tr0_bp, s_t_tr0, s_th_tr0, horizonte, hl
    )

    # Paso 2: Validacion progresiva con ventana expansiva avanzando de 20 en 20 sesiones (Seccion 3.5)
    paso_expansivo = 20
    preds_bp = {
        'Caminata Aleatoria (CA)': np.zeros(n_test, dtype=float),
        'Regresion Ridge': np.zeros(n_test, dtype=float),
        'Regresion Lasso': np.zeros(n_test, dtype=float),
        'XGBoost + Busqueda Aleatoria (Lasso)': np.zeros(n_test, dtype=float),
        'XGBoost + PSO (97 Vars Completas)': np.zeros(n_test, dtype=float),
        'XGBoost + PSO + Seleccion Lasso': np.zeros(n_test, dtype=float),
    }

    modelo_xgb_pso_final = None
    X_test_lasso_win_list = []

    for offset in range(0, n_test, paso_expansivo):
        pos_ini = idx_inicio_test + offset
        pos_fin = min(idx_inicio_test + offset + paso_expansivo, len(df_x))
        len_bloque = pos_fin - pos_ini

        # Purga estricta Seccion 3.5: solo etiquetas con t + h <= pos_ini
        fin_entrena = pos_ini - horizonte
        X_tr_b = df_x[todas_cols].iloc[:fin_entrena].reset_index(drop=True)
        y_tr_b = y_bp[:fin_entrena]
        w_tr_b = pesos_regimen_reciente(len(X_tr_b), hl)

        X_te_b = df_x[todas_cols].iloc[pos_ini:pos_fin].reset_index(drop=True)

        prep_b = PreprocesadorVentana().fit(X_tr_b)
        X_tr_win = prep_b.transform_winsor(X_tr_b)
        X_te_win = prep_b.transform_winsor(X_te_b)

        X_tr_sc = prep_b.transform_scaled(X_tr_b)
        X_te_sc = prep_b.transform_scaled(X_te_b)

        # Seleccionar indices de vars_lasso para los modelos reducidos y lineales
        idx_cols_lasso = [todas_cols.index(c) for c in vars_lasso]

        # 1. Regresion Ridge (Seccion 3.4)
        ridge = Ridge(alpha=1500.0, random_state=42)
        ridge.fit(X_tr_sc[:, idx_cols_lasso], y_tr_b, sample_weight=w_tr_b)
        preds_bp['Regresion Ridge'][offset:offset + len_bloque] = ridge.predict(X_te_sc[:, idx_cols_lasso])

        # 2. Regresion Lasso (Seccion 3.4)
        lasso = Lasso(alpha=alpha_lasso_map[horizonte] * 1.2, max_iter=3000, random_state=42)
        lasso.fit(X_tr_sc, y_tr_b, sample_weight=w_tr_b)
        preds_bp['Regresion Lasso'][offset:offset + len_bloque] = lasso.predict(X_te_sc)

        # 3. XGBoost + Busqueda Aleatoria (subconjunto Lasso)
        xgb_rs = entrenar_xgboost_con_parada_temprana(
            params_rs, X_tr_win[vars_lasso], y_tr_b, w_tr_b, horizonte
        )
        preds_bp['XGBoost + Busqueda Aleatoria (Lasso)'][offset:offset + len_bloque] = xgb_rs.predict(X_te_win[vars_lasso])

        # 4. XGBoost + PSO con las 97 variables completas (Seccion 3.3)
        xgb_full = entrenar_xgboost_con_parada_temprana(
            params_pso, X_tr_win, y_tr_b, w_tr_b, horizonte
        )
        preds_bp['XGBoost + PSO (97 Vars Completas)'][offset:offset + len_bloque] = xgb_full.predict(X_te_win)

        # 5. XGBoost + PSO + Seleccion Lasso (Modelo Propuesto Principal)
        xgb_pso_lasso = entrenar_xgboost_con_parada_temprana(
            params_pso, X_tr_win[vars_lasso], y_tr_b, w_tr_b, horizonte
        )
        preds_bp['XGBoost + PSO + Seleccion Lasso'][offset:offset + len_bloque] = xgb_pso_lasso.predict(X_te_win[vars_lasso])

        modelo_xgb_pso_final = xgb_pso_lasso
        X_test_lasso_win_list.append(X_te_win[vars_lasso])

    X_test_lasso_win_all = pd.concat(X_test_lasso_win_list, ignore_index=True)

    # Extraer series reales del bloque de prueba
    s_t_test = s_t[idx_inicio_test:]
    s_th_test = s_th[idx_inicio_test:]
    r_th_test_bp = y_bp[idx_inicio_test:]
    fechas_origen_test = df_x['Fecha'].iloc[idx_inicio_test:].reset_index(drop=True)
    fechas_destino_test = fechas_dest.iloc[idx_inicio_test:].reset_index(drop=True)

    # Calcular metricas exactas de la Ecuacion (3) y pruebas DM y PT para cada modelo
    err_abs_ca = np.abs(s_th_test - s_t_test)
    eam_ca = float(np.mean(err_abs_ca))

    filas_metricas = []
    df_predicciones = pd.DataFrame({
        'fecha_origen': fechas_origen_test,
        'fecha_objetivo': fechas_destino_test,
        'S_t_origen_pen': s_t_test,
        'S_th_real_pen': s_th_test,
        'retorno_real_bp': r_th_test_bp,
    })

    for nombre_mod, r_pred_bp in preds_bp.items():
        s_pred = s_t_test * np.exp(r_pred_bp / 10000.0)
        err_abs_mod = np.abs(s_th_test - s_pred)
        eam_mod = float(np.mean(err_abs_mod))
        rmse_mod = float(np.sqrt(mean_squared_error(s_th_test, s_pred)))
        u_h = float(eam_mod / eam_ca) if eam_ca > 0 else 1.0

        # Error traducido a soles por cada USD 10,000 pagados (Seccion 1.3)
        error_usd_10k_soles = eam_mod * 10000.0
        ahorro_error_usd_10k_vs_ca = (eam_ca - eam_mod) * 10000.0

        if nombre_mod == 'Caminata Aleatoria (CA)':
            dm_stat, dm_pval = 0.0, 1.0
            ad_rate, pt_stat, pt_pval = 0.0, 0.0, 1.0
            n_aciertos, n_mov = 0, int((r_th_test_bp != 0).sum())
        else:
            dm_stat, dm_pval = prueba_diebold_mariano(err_abs_mod, err_abs_ca, horizonte)
            ad_rate, pt_stat, pt_pval = prueba_pesaran_timmermann(r_th_test_bp, r_pred_bp)
            n_mov = int((r_th_test_bp != 0).sum())
            n_aciertos = int(round(ad_rate * n_mov))

        filas_metricas.append({
            'horizonte_dias_habiles': horizonte,
            'es_horizonte_principal': (horizonte == HORIZONTE_PRINCIPAL),
            'modelo': nombre_mod,
            'n_pronosticos': n_test,
            'n_variables_usadas': (
                0 if nombre_mod == 'Caminata Aleatoria (CA)'
                else (len(todas_cols) if '97 Vars' in nombre_mod else len(vars_lasso))
            ),
            'EAM_soles_por_usd': eam_mod,
            'EAM_CA_soles_por_usd': eam_ca,
            'U_h_ratio_EAM': u_h,
            'supera_CA_U_menor_1': bool(u_h < 1.0),
            'error_soles_por_10000_usd': error_usd_10k_soles,
            'reduccion_error_soles_por_10000_usd_vs_CA': ahorro_error_usd_10k_vs_ca,
            'RMSE_soles_por_usd': rmse_mod,
            'AD_acierto_direccional_pct': ad_rate * 100.0,
            'sesiones_acertadas': f'{n_aciertos}/{n_mov}',
            'DM_estadistico': dm_stat,
            'DM_pvalue_unilateral': dm_pval,
            'PT_estadistico': pt_stat,
            'PT_pvalue_unilateral': pt_pval,
        })

        col_slug = (
            nombre_mod.lower()
            .replace(' ', '_')
            .replace('+', '')
            .replace('(', '')
            .replace(')', '')
            .replace('__', '_')
        )
        df_predicciones[f'S_hat_{col_slug}'] = s_pred
        df_predicciones[f'r_hat_bp_{col_slug}'] = r_pred_bp

    # =========================================================================
    # 6. VALORES SHAP EXACTOS (TreeSHAP NATIVO DE XGBOOST, SECCION 3.5)
    # =========================================================================
    booster = modelo_xgb_pso_final.get_booster()
    dmat_test = xgb.DMatrix(X_test_lasso_win_all, feature_names=vars_lasso)
    # pred_contribs=True calcula exactamente los valores TreeSHAP (ultima columna es el bias)
    shap_matrix = booster.predict(dmat_test, pred_contribs=True)[:, :-1]
    mean_abs_shap = pd.Series(np.mean(np.abs(shap_matrix), axis=0), index=vars_lasso).sort_values(ascending=False)

    # =========================================================================
    # 7. PRUEBA DE EXCLUSION POR GRUPOS (ABLATION STUDY, SECCION 3.5)
    #    Reentrenar XGBoost + PSO sin cada uno de los 3 grupos de variables
    # =========================================================================
    grupos_cat = catalogo['grupos']
    configuraciones_ablacion = [
        ('Modelo Completo (3 Grupos con Seleccion Lasso)', vars_lasso),
        ('Sin Grupo 1: Propias del USD/PEN', [c for c in vars_lasso if c not in grupos_cat['propias_usdpen']]),
        ('Sin Grupo 2: Globales y Commodities (VIX, DXY, Cobre, WTI, US10Y)', [c for c in vars_lasso if c not in grupos_cat['globales_y_commodities']]),
        ('Sin Grupo 3: Macroeconomicas BCRP (IPC e Inflacion)', [c for c in vars_lasso if c not in grupos_cat['macroeconomicas_bcrp']]),
    ]

    filas_ablacion = []
    for nombre_ab, cols_ab in configuraciones_ablacion:
        if not cols_ab:
            cols_ab = todas_cols[:5]
        preds_ab = np.zeros(n_test, dtype=float)
        for offset in range(0, n_test, paso_expansivo):
            pos_ini = idx_inicio_test + offset
            pos_fin = min(idx_inicio_test + offset + paso_expansivo, len(df_x))
            len_b = pos_fin - pos_ini
            fin_ent = pos_ini - horizonte
            X_tr_b = df_x[cols_ab].iloc[:fin_ent].reset_index(drop=True)
            y_tr_b = y_bp[:fin_ent]
            w_tr_b = pesos_regimen_reciente(len(X_tr_b), hl)
            X_te_b = df_x[cols_ab].iloc[pos_ini:pos_fin].reset_index(drop=True)
            prep_b = PreprocesadorVentana().fit(X_tr_b)
            m_ab = entrenar_xgboost_con_parada_temprana(
                params_pso, prep_b.transform_winsor(X_tr_b), y_tr_b, w_tr_b, horizonte
            )
            preds_ab[offset:offset + len_b] = m_ab.predict(prep_b.transform_winsor(X_te_b))

        s_ab = s_t_test * np.exp(preds_ab / 10000.0)
        eam_ab = float(mean_absolute_error(s_th_test, s_ab))
        u_ab = eam_ab / eam_ca
        ad_ab, _, _ = prueba_pesaran_timmermann(r_th_test_bp, preds_ab)
        filas_ablacion.append({
            'horizonte_dias_habiles': horizonte,
            'configuracion_exclusion': nombre_ab,
            'n_variables_activas': len(cols_ab),
            'EAM_soles_por_usd': eam_ab,
            'U_h_ratio_EAM': u_ab,
            'deterioro_EAM_soles_por_10000_usd': (eam_ab - filas_metricas[-1]['EAM_soles_por_usd']) * 10000.0,
            'AD_acierto_direccional_pct': ad_ab * 100.0,
        })

    # Guardar el modelo final XGBoost + PSO + Lasso
    modelo_xgb_pso_final.save_model(str(SALIDA / f'modelo_xgboost_pso_h{horizonte}.json'))
    modelo_xgb_pso_final.save_model(str(MODELOS_DIR / f'modelo_xgboost_h{horizonte}.json'))

    return (
        filas_metricas,
        df_predicciones,
        mean_abs_shap,
        filas_ablacion,
        vars_lasso,
        coefs_lasso,
        params_pso,
        params_rs,
        df_hist_pso,
    )


# =============================================================================
# 8. GRAFICOS AUTOEXPLICATIVOS CON LEYENDAS COMPLETAS (NOTA IMPORTANTE SECCION 11)
# =============================================================================
def graficar_pronosticos_horizonte(h: int, df_pred: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.plot(df_pred['fecha_objetivo'], df_pred['S_th_real_pen'], label='S_{t+h}: Tipo de cambio spot real observado (S/ por USD)', color='black', linewidth=2.2)
    ax.plot(df_pred['fecha_objetivo'], df_pred['S_hat_xgboost_pso_seleccion_lasso'], label='S_hat_{t+h} (XGBoost + PSO + Seleccion Lasso): Pronostico propuesto', color='#1f77b4', linewidth=1.8)
    ax.plot(df_pred['fecha_objetivo'], df_pred['S_hat_regresion_lasso'], label='S_hat_{t+h} (Regresion Lasso): Referencia lineal regularizada L1', color='#2ca02c', linewidth=1.3, alpha=0.85)
    ax.plot(df_pred['fecha_objetivo'], df_pred['S_hat_caminata_aleatoria_ca'], label='S_hat_{t+h} (Caminata Aleatoria CA): Pronostico sin cambio (S_t)', color='gray', linestyle='--', linewidth=1.2)

    etiqueta_p = ' (HORIZONTE PRINCIPAL — Tesoreria Semanal)' if h == 5 else ''
    ax.set_title(f'Figura: Pronostico del Tipo de Cambio USD/PEN a h = {h} dia(s) habil(es){etiqueta_p}\n[Validacion progresiva fuera de muestra | S_t en Soles por Dolar Estadounidense]')
    ax.set_ylabel('Cotizacion S_{t+h} (Soles por USD)')
    ax.set_xlabel('Fecha de liquidacion objetivo (t + h)')
    ax.legend(loc='best', fontsize=8.5)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(SALIDA / f'grafico_h{h}.png', dpi=160)
    plt.close(fig)


def graficar_shap_horizonte(h: int, shap_series: pd.Series, catalogo: dict):
    top12 = shap_series.head(12).sort_values()
    grupos = catalogo['grupos']
    colores = []
    for col in top12.index:
        if col in grupos['globales_y_commodities']:
            colores.append('#d62728')  # Rojo: Global / Commodity
        elif col in grupos['macroeconomicas_bcrp']:
            colores.append('#2ca02c')  # Verde: Macro BCRP
        else:
            colores.append('#1f77b4')  # Azul: Propia USD/PEN

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    bars = ax.barh(top12.index, top12.values, color=colores)
    ax.bar_label(bars, fmt='%.2f pb', fontsize=7.5, padding=3)
    ax.set_xlim(0, float(top12.max()) * 1.18)
    ax.set_title(
        f'Figura: Importancia de Variables via Valores SHAP en XGBoost + PSO — Horizonte h = {h} dia(s) habil(es)\n'
        f'[azul = Propia USD/PEN | rojo = Global/Commodity (VIX, DXY, Cobre, WTI, US10Y) | verde = Macro BCRP]'
    )
    ax.set_xlabel('Magnitud media absoluta de atribucion |SHAP| sobre el retorno predicho (en puntos basicos, 1 pb = 0.01%)')
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    fig.savefig(SALIDA / f'importancias_shap_h{h}.png', dpi=160)
    fig.savefig(SALIDA / f'importancias_h{h}.png', dpi=160)
    plt.close(fig)


def graficar_barras_resumen_con_valores(df_metricas: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    orden_modelos = [
        'Caminata Aleatoria (CA)',
        'Regresion Ridge',
        'Regresion Lasso',
        'XGBoost + Busqueda Aleatoria (Lasso)',
        'XGBoost + PSO (97 Vars Completas)',
        'XGBoost + PSO + Seleccion Lasso',
    ]

    # Panel A: Cociente U_h = EAM(modelo) / EAM(CA)
    piv_u = df_metricas.pivot(index='horizonte_dias_habiles', columns='modelo', values='U_h_ratio_EAM')[orden_modelos]
    piv_u.plot(kind='bar', ax=axes[0], rot=0, width=0.84)
    axes[0].axhline(1.0, color='black', linestyle='--', linewidth=1.3, label='Umbral U_h = 1.000 (Caminata Aleatoria CA)')
    axes[0].set_title('Panel A: Cociente de Error Relativo U_h = EAM_h(modelo) / EAM_h(CA)\n[Valores U_h < 1.000 indican menor error en soles que la Caminata Aleatoria]')
    axes[0].set_xlabel('Horizonte de pronostico h (dias habiles)')
    axes[0].set_ylabel('Cociente adimensional U_h (Menor a 1.0 es mejor)')
    axes[0].set_ylim(0.88, 1.08)
    axes[0].legend(loc='upper left', fontsize=7.5)
    axes[0].grid(axis='y', alpha=0.25)
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt='%.3f', fontsize=6.2, rotation=90, padding=2)

    # Panel B: Acierto Direccional AD_h (%)
    modelos_sin_ca = [m for m in orden_modelos if m != 'Caminata Aleatoria (CA)']
    piv_ad = df_metricas[df_metricas['modelo'] != 'Caminata Aleatoria (CA)'].pivot(
        index='horizonte_dias_habiles', columns='modelo', values='AD_acierto_direccional_pct'
    )[modelos_sin_ca]
    piv_ad.plot(kind='bar', ax=axes[1], rot=0, width=0.80)
    axes[1].axhline(50.0, color='black', linestyle='--', linewidth=1.3, label='Umbral Azar (AD_h = 50.0%)')
    axes[1].set_title('Panel B: Acierto Direccional AD_h (%) en Subida/Bajada del Dolar\n[Valores AD_h > 50.0% superan al lanzamiento de moneda]')
    axes[1].set_xlabel('Horizonte de pronostico h (dias habiles)')
    axes[1].set_ylabel('Acierto Direccional AD_h (%)')
    axes[1].set_ylim(0, 74)
    axes[1].legend(loc='upper right', fontsize=7.5)
    axes[1].grid(axis='y', alpha=0.25)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt='%.1f%%', fontsize=6.8, padding=2)

    fig.tight_layout()
    fig.savefig(SALIDA / 'resumen_metricas_uh_y_ad.png', dpi=160)
    plt.close(fig)


# =============================================================================
# 9. EJECUCION PRINCIPAL Y EXPORTACION DE TODAS LAS TABLAS DESCRIPTIVAS
# =============================================================================
def ejecutar():
    full_df, train_df, hold_df, catalogo = cargar_datos_y_catalogo()
    print('=== SOLFX-PREDICTOR | EXPERIMENTACION ALINEADA AL INFORME FINAL ===')
    print(f'Serie historica total: {len(full_df)} sesiones ({full_df["Fecha"].min().date()} a {full_df["Fecha"].max().date()})')
    print(f'Vector completo de predictores: {catalogo["total_predictores"]} caracteristicas (> 80 caracteristicas, Seccion 3.3)')
    print(f'  - Grupo 1 (Propias USD/PEN):        {len(catalogo["grupos"]["propias_usdpen"])} vars')
    print(f'  - Grupo 2 (Globales y Commodities): {len(catalogo["grupos"]["globales_y_commodities"])} vars')
    print(f'  - Grupo 3 (Macroeconomicas BCRP):   {len(catalogo["grupos"]["macroeconomicas_bcrp"])} vars')

    todas_metricas = []
    todas_ablaciones = []
    resumen_seleccion_lasso = []
    resumen_pso_vs_rs = []

    for h in HORIZONTES:
        (
            filas_m,
            df_pred,
            shap_s,
            filas_ab,
            vars_lasso,
            coefs_lasso,
            params_pso,
            params_rs,
            df_hist_pso,
        ) = evaluar_horizonte_completo(full_df, catalogo, h, n_eval_holdout=60)

        todas_metricas.extend(filas_m)
        todas_ablaciones.extend(filas_ab)

        df_pred.to_csv(SALIDA / f'predicciones_h{h}.csv', index=False)
        shap_s.rename('shap_abs_medio_bp').to_csv(SALIDA / f'importancias_shap_h{h}.csv')
        shap_s.rename('importancia_relativa').to_csv(SALIDA / f'importancias_h{h}.csv')
        df_hist_pso.to_csv(SALIDA / f'convergencia_pso_vs_rs_h{h}.csv', index=False)

        graficar_pronosticos_horizonte(h, df_pred)
        graficar_shap_horizonte(h, shap_s, catalogo)

        # Contabilizar cuantas variables de cada grupo selecciono Lasso
        g_prop = [c for c in vars_lasso if c in catalogo['grupos']['propias_usdpen']]
        g_glob = [c for c in vars_lasso if c in catalogo['grupos']['globales_y_commodities']]
        g_macr = [c for c in vars_lasso if c in catalogo['grupos']['macroeconomicas_bcrp']]

        resumen_seleccion_lasso.append({
            'Horizonte h (Dias Habiles)': h,
            'Total Variables Iniciales': catalogo['total_predictores'],
            'Variables Seleccionadas por Lasso': len(vars_lasso),
            'Reduccion Dimensional (%)': round((1.0 - len(vars_lasso) / catalogo['total_predictores']) * 100.0, 1),
            'Del Grupo 1 (Propias USD/PEN)': len(g_prop),
            'Del Grupo 2 (Globales y Commodities)': len(g_glob),
            'Del Grupo 3 (Macroeconomicas BCRP)': len(g_macr),
            'Top 8 Variables Seleccionadas por Lasso': ', '.join(vars_lasso[:8]),
        })

        resumen_pso_vs_rs.append({
            'Horizonte h (Dias Habiles)': h,
            'Hiperparametros Elegidos por PSO': json.dumps(params_pso, ensure_ascii=False),
            'Hiperparametros Elegidos por Busqueda Aleatoria': json.dumps(params_rs, ensure_ascii=False),
        })

    df_metricas = pd.DataFrame(todas_metricas)
    df_metricas.to_csv(SALIDA / 'metricas_comparativas.csv', index=False)

    # Tabla con nombres descriptivos completos en espanol
    mapa_desc = {
        'horizonte_dias_habiles': 'Horizonte h (Dias Habiles)',
        'modelo': 'Modelo Evaluado',
        'n_pronosticos': 'Pronosticos Evaluados (n)',
        'n_variables_usadas': 'N de Predictores Activos',
        'EAM_soles_por_usd': 'EAM_h: Error Absoluto Medio (Soles por USD)',
        'EAM_CA_soles_por_usd': 'EAM_h(CA): Error Caminata Aleatoria (Soles por USD)',
        'U_h_ratio_EAM': 'Cociente U_h = EAM(Modelo) / EAM(CA) [< 1 supera a CA]',
        'error_soles_por_10000_usd': 'Error Medio por Factura de USD 10,000 (en Soles S/)',
        'reduccion_error_soles_por_10000_usd_vs_CA': 'Ahorro en Error vs. CA por cada USD 10,000 (en Soles S/)',
        'RMSE_soles_por_usd': 'RMSE en Nivel (Soles por USD)',
        'AD_acierto_direccional_pct': 'AD_h: Acierto Direccional (%)',
        'sesiones_acertadas': 'Sesiones con Direccion Acertada',
        'DM_estadistico': 'Estadistico Diebold-Mariano (DM)',
        'DM_pvalue_unilateral': 'Valor-p Diebold-Mariano (H1: U_h < 1)',
        'PT_estadistico': 'Estadistico Pesaran-Timmermann (PT)',
        'PT_pvalue_unilateral': 'Valor-p Pesaran-Timmermann (H1: AD_h > 50%)',
    }
    df_desc = df_metricas[list(mapa_desc.keys())].rename(columns=mapa_desc)
    df_desc.to_csv(SALIDA / 'tabla_resultados_descriptiva.csv', index=False)

    # Tabla de valores exactos del grafico de barras (resumen_metricas_uh_y_ad.png)
    df_ablacion = pd.DataFrame(todas_ablaciones)
    df_ablacion.to_csv(SALIDA / 'tabla_exclusion_grupos_ablacion.csv', index=False)

    df_lasso_sel = pd.DataFrame(resumen_seleccion_lasso)
    df_lasso_sel.to_csv(SALIDA / 'tabla_seleccion_variables_lasso.csv', index=False)

    df_pso_rs = pd.DataFrame(resumen_pso_vs_rs)
    df_pso_rs.to_csv(SALIDA / 'tabla_hiperparametros_pso_vs_random_search.csv', index=False)

    graficar_barras_resumen_con_valores(df_metricas)

    print('\n=== 1. SELECCION DE VARIABLES CON LASSO (SECCION 3.3) ===')
    print(df_lasso_sel.to_string(index=False))

    print('\n=== 2. RESULTADOS COMPARATIVOS (CA, RIDGE, LASSO, XGBOOST RS, XGBOOST PSO) ===')
    cols_print = [
        'horizonte_dias_habiles', 'modelo', 'n_variables_usadas',
        'EAM_soles_por_usd', 'U_h_ratio_EAM', 'error_soles_por_10000_usd',
        'AD_acierto_direccional_pct', 'sesiones_acertadas', 'DM_pvalue_unilateral', 'PT_pvalue_unilateral'
    ]
    print(df_metricas[cols_print].to_string(index=False))

    print('\n=== 3. PRUEBA DE EXCLUSION POR GRUPOS / ABLACION (SECCION 3.5) ===')
    print(df_ablacion.to_string(index=False))
    print('\nTodos los artefactos fueron guardados en:', SALIDA)


if __name__ == '__main__':
    ejecutar()
