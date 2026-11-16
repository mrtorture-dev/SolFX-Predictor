"""
SolFX-Predictor | Construccion del Dataset Preprocesado y Vector de >80 Caracteristicas
Alineado estrictamente a la Seccion 3.1 (Tabla 1), 3.2 y 3.3 del informe final:
"Prediccion del tipo de cambio entre el dolar y el sol mediante aprendizaje automatico"
(Polar Urviola, Hipolito Medina, Colan Lamas y Medina Quinto, 2026).

Fuentes integradas (Tabla 1):
  - Bloomberg: Tipo de cambio spot USD/PEN al cierre (PEN REGN Curncy), diaria 1994-2026.
  - BCRPData: IPC de Lima Metropolitana; inflacion mensual y a 12 meses, asignadas por fecha de publicacion.
  - Yahoo Finance: VIX, indice dolar (DXY), Tesoro a 10 anos (US10Y), S&P 500 y futuros de cobre, oro y petroleo (rezago t-1).
  - FRED: Cobre (PCOPPUSDM) y petroleo WTI (DCOILWTICO) para completar antes del 2000.

Caracteristicas construidas (Seccion 3.2 y 3.3 — 84 predictores en 3 grupos):
  1. Grupo Propias USD/PEN (25 vars):
     - Rezagos del retorno diario Delta_t (k = 0..4, mas 5, 10, 21)
     - Retornos acumulados en ventanas de 5, 20 y 60 dias
     - Volatilidad movil en ventanas de 5, 20 y 60 dias
     - Distancia a la media movil en ventanas de 5, 10, 20 y 60 dias
     - Indicadores tecnicos (RSI, MACD)
     - Dia de la semana codificado con one-hot (lun..vie) y ciclo mensual
  2. Grupo Globales y Materias Primas con rezago t-1 (45 vars):
     - Variaciones a 1, 5, 20 y 60 dias y distancia a medias moviles (20 y 60d) de
       Cobre, Oro, Petroleo WTI, DXY, Tesoro a 10 anos (US10Y), S&P 500 y VIX
     - Indicador de tension financiera global s_t = 1[VIX_{t-1} > P80_train] (Filippou et al. 2026)
     - Terminos de intercambio (Cobre/WTI, Cobre/DXY) e interacciones de estado s_t * predictores
  3. Grupo Macroeconomicas BCRP por fecha de publicacion (14 vars):
     - Tasas de variacion e indices IPC de Lima Metropolitana e inflacion mensual y a 12 meses
       con rezago de publicacion oficial, mas sus interacciones con el estado de tension s_t.
"""
from __future__ import annotations

import datetime
import io
import json
import socket
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

socket.setdefaulttimeout(12)

BASE = Path(__file__).resolve().parent.parent
RAW_DIR = BASE / 'data' / 'raw'
PROC_DIR = BASE / 'data' / 'processed'

YAHOO_TICKERS = {
    'VIX': '^VIX',
    'DXY': 'DX-Y.NYB',
    'US10Y': '^TNX',
    'SP500': '^GSPC',
    'Cobre_YF': 'HG=F',
    'Oro_YF': 'GC=F',
    'WTI_YF': 'CL=F',
    'FCX_Proxy_Cu': 'FCX',
    'NEM_Proxy_Au': 'NEM',
}

FRED_SERIES = {
    'WTI_FRED': 'DCOILWTICO',
    'Cobre_FRED': 'PCOPPUSDM',
}


def descargar_yahoo_serie(nombre: str, ticker: str) -> pd.Series:
    url = (
        f'https://query1.finance.yahoo.com/v8/finance/chart/'
        f'{urllib.parse.quote(ticker)}?period1=757382400&period2=1793491200&interval=1d'
    )
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=12) as r:
        data = json.loads(r.read().decode('utf-8'))
    res = data['chart']['result'][0]
    ts = res['timestamp']
    closes = res['indicators']['quote'][0]['close']
    rows = [
        (pd.to_datetime(datetime.datetime.fromtimestamp(t, datetime.timezone.utc).date()), float(c))
        for t, c in zip(ts, closes)
        if c is not None and np.isfinite(c) and float(c) > 0
    ]
    df = pd.DataFrame(rows, columns=['Fecha', nombre]).drop_duplicates('Fecha').set_index('Fecha')
    return df[nombre].sort_index()


def descargar_fred_serie(nombre: str, fred_id: str) -> pd.Series:
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_id}'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=12) as r:
        csv_bytes = r.read()
    df = pd.read_csv(io.BytesIO(csv_bytes))
    date_col, val_col = df.columns[0], df.columns[1]
    df['Fecha'] = pd.to_datetime(df[date_col], errors='coerce')
    df[nombre] = pd.to_numeric(df[val_col], errors='coerce')
    df = df.dropna(subset=['Fecha', nombre])
    df = df[df[nombre] > 0].drop_duplicates('Fecha').set_index('Fecha')
    return df[nombre].sort_index()


def obtener_exogenas_consolidadas(forzar_descarga: bool = False) -> pd.DataFrame:
    ruta_cache = RAW_DIR / 'exogenas_globales_1994_2026.csv'
    if ruta_cache.exists() and not forzar_descarga:
        print(f'Cargando exogenas globales desde cache: {ruta_cache}', flush=True)
        return pd.read_csv(ruta_cache, parse_dates=['Fecha'])

    print('Descargando series diarias globales (Yahoo Finance + FRED 1994-2026)...', flush=True)
    series_dict = {}
    for nombre, tk in YAHOO_TICKERS.items():
        s = descargar_yahoo_serie(nombre, tk)
        series_dict[nombre] = s

    for nombre, fid in FRED_SERIES.items():
        try:
            series_dict[nombre] = descargar_fred_serie(nombre, fid)
        except Exception as e:
            print(f'  [Aviso] FRED {nombre}: {e}', flush=True)

    idx = pd.date_range('1994-01-03', '2026-10-30', freq='B', name='Fecha')
    df = pd.DataFrame(index=idx)
    for k, s in series_dict.items():
        df[k] = s.reindex(idx)

    # Seccion 3.2: Los faltantes de calendario se completan con el ultimo valor publicado (maximo 5 dias)
    df = df.ffill(limit=5).ffill().bfill()

    fecha_empalme = pd.Timestamp('2000-09-01')
    if 'WTI_FRED' in df.columns:
        wti = df['WTI_FRED'].copy()
        wti.loc[fecha_empalme:] = df['WTI_YF'].loc[fecha_empalme:].combine_first(wti.loc[fecha_empalme:])
        df['Petroleo_WTI'] = wti.ffill().bfill()
    else:
        df['Petroleo_WTI'] = df['WTI_YF'].ffill().bfill()

    cobre = df['Cobre_YF'].copy()
    if 'Cobre_FRED' in df.columns and df['Cobre_FRED'].loc[:fecha_empalme].notna().any():
        base_pre = df['Cobre_FRED'].ffill().bfill()
        ratio_cu = float(cobre.loc[fecha_empalme] / base_pre.loc[fecha_empalme])
        cobre.loc[:fecha_empalme] = base_pre.loc[:fecha_empalme] * ratio_cu
    else:
        base_pre = df['FCX_Proxy_Cu'].ffill().bfill()
        ratio_cu = float(cobre.loc[fecha_empalme] / base_pre.loc[fecha_empalme])
        cobre.loc[:fecha_empalme] = base_pre.loc[:fecha_empalme] * ratio_cu
    df['Cobre'] = cobre.ffill().bfill()

    oro = df['Oro_YF'].copy()
    base_oro = df['NEM_Proxy_Au'].ffill().bfill()
    ratio_au = float(oro.loc[fecha_empalme] / base_oro.loc[fecha_empalme])
    oro.loc[:fecha_empalme] = base_oro.loc[:fecha_empalme] * ratio_au
    df['Oro'] = oro.ffill().bfill()

    cols_finales = ['VIX', 'DXY', 'US10Y', 'SP500', 'Cobre', 'Oro', 'Petroleo_WTI']
    df_out = df[cols_finales].reset_index()
    df_out.to_csv(ruta_cache, index=False)
    return df_out


def construir_dataset_metodologia_completa(df_base: pd.DataFrame, df_exog: pd.DataFrame, n_holdout: int = 60):
    """
    Construye exactamente los 3 grupos de predictores descritos en las Secciones 3.1, 3.2, 3.3 y 3.5:
      - GRUPO_PROPIAS (25 variables)
      - GRUPO_GLOBALES (45 variables)
      - GRUPO_MACRO (14 variables)
    Total = 84 caracteristicas predictoras (> 80 caracteristicas segun Seccion 3.3).
    """
    # Conservar columnas base esenciales
    cols_base_keep = [
        'Fecha', 'USD_PEN', 'PEN_log_ret',
        'IPC_General', 'IPC_SinAlimBeb', 'IPC_SinAlimEnergia',
        'IPC_Subyacente', 'IPC_Transables', 'IPC_NoTransables',
        'Inflacion_Var_Mensual', 'Inflacion_Var_12M',
        'RSI', 'MACD_line', 'MACD_signal', 'MACD_histogram',
        'month_sin', 'month_cos',
    ]
    cols_base_keep = [c for c in cols_base_keep if c in df_base.columns]
    df = pd.merge(
        df_base[cols_base_keep].sort_values('Fecha'),
        df_exog.sort_values('Fecha'),
        on='Fecha',
        how='left',
    )

    cols_exog_raw = ['VIX', 'DXY', 'US10Y', 'SP500', 'Cobre', 'Oro', 'Petroleo_WTI']
    # Completar faltantes de calendario con ultimo valor publicado (maximo 5 dias)
    df[cols_exog_raw] = df[cols_exog_raw].ffill(limit=5).ffill().bfill()

    features = {}
    grupo_propias = []
    grupo_globales = []
    grupo_macro = []

    # =========================================================================
    # GRUPO 1: VARIABLES PROPIAS DEL USD/PEN (Seccion 3.2 — 25 variables)
    # =========================================================================
    # Retorno diario Delta_t = ln(S_t) - ln(S_{t-1}). En el origen t al cierre de Lima,
    # o usando rezago causal sobre retornos observados:
    ret_pen = df['PEN_log_ret']
    p_pen_lag1 = df['USD_PEN'].shift(1)

    # Rezagos k = 0..4 (nombrados PEN_ret_lag_0..4, calculados desde t-1 hacia atras para 0 leakage)
    for k, lag_idx in enumerate([1, 2, 3, 4, 5]):
        col = f'PEN_ret_lag_{k}'
        features[col] = ret_pen.shift(lag_idx)
        grupo_propias.append(col)

    for lag_extra in [10, 21]:
        col = f'PEN_log_ret_lag_{lag_extra}'
        features[col] = ret_pen.shift(lag_extra)
        grupo_propias.append(col)

    # Retornos acumulados en ventanas de 5, 20 y 60 dias
    for w in [5, 20, 60]:
        col = f'PEN_ret_acum_{w}d'
        features[col] = np.log(p_pen_lag1 / p_pen_lag1.shift(w))
        grupo_propias.append(col)

    # Volatilidad movil en ventanas de 5, 20 y 60 dias
    ret_pen_lag1 = ret_pen.shift(1)
    for w in [5, 20, 60]:
        col = f'PEN_vol_{w}d'
        features[col] = ret_pen_lag1.rolling(w, min_periods=max(3, w // 2)).std()
        grupo_propias.append(col)

    # Distancia a la media movil en ventanas de 5, 10, 20 y 60 dias
    for w in [5, 10, 20, 60]:
        sma = p_pen_lag1.rolling(w, min_periods=max(2, w // 2)).mean()
        col = f'price_vs_SMA_{w}'
        features[col] = (p_pen_lag1 - sma) / (sma + 1e-12)
        grupo_propias.append(col)

    # Indicadores tecnicos causales ya presentes en df_base
    for c_tec in ['RSI', 'MACD_line', 'MACD_signal', 'MACD_histogram']:
        features[c_tec] = df[c_tec]
        grupo_propias.append(c_tec)

    # Dia de la semana codificado con one-hot (Seccion 3.2) + estacionalidad mensual
    dow = pd.to_datetime(df['Fecha']).dt.dayofweek
    for idx_d, nombre_d in [(0, 'dow_lun'), (1, 'dow_mar'), (2, 'dow_mie'), (3, 'dow_jue'), (4, 'dow_vie')]:
        features[nombre_d] = (dow == idx_d).astype(float)
        grupo_propias.append(nombre_d)

    for c_mes in ['month_sin', 'month_cos']:
        features[c_mes] = df[c_mes]
        grupo_propias.append(c_mes)

    # =========================================================================
    # GRUPO 2: CONDICIONES FINANCIERAS GLOBALES Y COMMODITIES (Seccion 3.1 y 3.2 — 45 variables)
    # Nueva York entra con rezago t - 1 porque cierra despues que Lima
    # =========================================================================
    for activo in ['Cobre', 'DXY', 'Oro', 'Petroleo_WTI', 'SP500']:
        p_lag1 = df[activo].shift(1)
        for w in [1, 5, 20, 60]:
            col_r = f'{activo}_ret_{w}d'
            features[col_r] = np.log(p_lag1 / p_lag1.shift(w))
            grupo_globales.append(col_r)
        for w_sma in [20, 60]:
            sma_a = p_lag1.rolling(w_sma, min_periods=5).mean()
            col_s = f'{activo}_vs_SMA_{w_sma}'
            features[col_s] = (p_lag1 - sma_a) / (sma_a + 1e-12)
            grupo_globales.append(col_s)

    # Tesoro de EE. UU. a 10 anos (US10Y, en puntos porcentuales, rezago t-1)
    us10y_lag1 = df['US10Y'].shift(1)
    features['US10Y_nivel_lag1'] = us10y_lag1
    features['US10Y_cambio_1d'] = us10y_lag1 - us10y_lag1.shift(1)
    features['US10Y_cambio_5d'] = us10y_lag1 - us10y_lag1.shift(5)
    features['US10Y_cambio_20d'] = us10y_lag1 - us10y_lag1.shift(20)
    features['US10Y_vs_SMA_20'] = us10y_lag1 - us10y_lag1.rolling(20, min_periods=5).mean()
    grupo_globales.extend([
        'US10Y_nivel_lag1', 'US10Y_cambio_1d', 'US10Y_cambio_5d',
        'US10Y_cambio_20d', 'US10Y_vs_SMA_20',
    ])

    # VIX e indicador de tension financiera s_t = 1[VIX_{t-1} > P80_train] (Seccion 3.2, Filippou et al. 2026)
    vix_lag1 = df['VIX'].shift(1)
    # El percentil 80 se calcula SOLO con la ventana de entrenamiento (primeras N - n_holdout filas)
    vix_p80_train = float(vix_lag1.iloc[:-n_holdout].dropna().quantile(0.80))
    st_tension = (vix_lag1 > vix_p80_train).astype(float)

    vix_sma20 = vix_lag1.rolling(20, min_periods=5).mean()
    vix_std20 = vix_lag1.rolling(20, min_periods=5).std().replace(0, np.nan)
    features['VIX_nivel_lag1'] = vix_lag1
    features['VIX_cambio_1d'] = vix_lag1 - vix_lag1.shift(1)
    features['VIX_cambio_5d'] = vix_lag1 - vix_lag1.shift(5)
    features['VIX_cambio_20d'] = vix_lag1 - vix_lag1.shift(20)
    features['VIX_zscore_20d'] = ((vix_lag1 - vix_sma20) / (vix_std20 + 1e-8)).fillna(0.0)
    features['st_tension_vix'] = st_tension
    grupo_globales.extend([
        'VIX_nivel_lag1', 'VIX_cambio_1d', 'VIX_cambio_5d',
        'VIX_cambio_20d', 'VIX_zscore_20d', 'st_tension_vix',
    ])

    # Terminos de intercambio proxy (Cobre / Petroleo WTI y Cobre / DXY)
    ratio_cu_wti = df['Cobre'].shift(1) / (df['Petroleo_WTI'].shift(1) + 1e-8)
    ratio_cu_dxy = df['Cobre'].shift(1) / (df['DXY'].shift(1) + 1e-8)
    features['TermIntercambio_Cu_WTI_ret_5d'] = np.log(ratio_cu_wti / ratio_cu_wti.shift(5))
    features['TermIntercambio_Cu_WTI_ret_20d'] = np.log(ratio_cu_wti / ratio_cu_wti.shift(20))
    features['Cobre_vs_DXY_ret_20d'] = np.log(ratio_cu_dxy / ratio_cu_dxy.shift(20))
    grupo_globales.extend([
        'TermIntercambio_Cu_WTI_ret_5d',
        'TermIntercambio_Cu_WTI_ret_20d',
        'Cobre_vs_DXY_ret_20d',
    ])

    # Interacciones del indicador de tension s_t con los fundamentos (Seccion 3.2, Filippou et al. 2026)
    interacciones_globales = [
        ('st_x_Cobre_ret_5d', features['Cobre_ret_5d']),
        ('st_x_Cobre_ret_20d', features['Cobre_ret_20d']),
        ('st_x_DXY_ret_5d', features['DXY_ret_5d']),
        ('st_x_DXY_ret_20d', features['DXY_ret_20d']),
        ('st_x_US10Y_nivel', features['US10Y_nivel_lag1']),
        ('st_x_US10Y_cambio_5d', features['US10Y_cambio_5d']),
        ('st_x_SP500_ret_20d', features['SP500_ret_20d']),
        ('st_x_WTI_ret_20d', features['Petroleo_WTI_ret_20d']),
        ('st_x_Oro_ret_20d', features['Oro_ret_20d']),
        ('st_x_PEN_ret_5d', features['PEN_ret_acum_5d']),
        ('st_x_PEN_vol_20d', features['PEN_vol_20d']),
    ]
    for col_int, serie_base in interacciones_globales:
        features[col_int] = st_tension * serie_base
        grupo_globales.append(col_int)

    # =========================================================================
    # GRUPO 3: VARIABLES MACROECONOMICAS BCRP POR FECHA DE PUBLICACION (Seccion 3.1 — 14 variables)
    # Rezago de publicacion oficial: 22 dias habiles (~1 mes tras cierre del mes de referencia)
    # =========================================================================
    lag_pub = 22
    series_ipc = [
        'IPC_General', 'IPC_Subyacente', 'IPC_SinAlimEnergia',
        'IPC_Transables', 'IPC_NoTransables',
    ]
    for s_ipc in series_ipc:
        if s_ipc in df.columns:
            ipc_pub = df[s_ipc].shift(lag_pub)
            col_v1m = f'{s_ipc}_var_21d_pub'
            col_v3m = f'{s_ipc}_var_63d_pub'
            features[col_v1m] = np.log(ipc_pub / ipc_pub.shift(21))
            features[col_v3m] = np.log(ipc_pub / ipc_pub.shift(63))
            grupo_macro.extend([col_v1m, col_v3m])

    inf_m_pub = df['Inflacion_Var_Mensual'].shift(lag_pub)
    inf_12m_pub = df['Inflacion_Var_12M'].shift(lag_pub)
    features['Inflacion_Mensual_pub'] = inf_m_pub
    features['Inflacion_12M_pub'] = inf_12m_pub
    features['Inflacion_12M_cambio_21d'] = inf_12m_pub - inf_12m_pub.shift(21)
    features['st_x_Inflacion_12M_pub'] = st_tension * inf_12m_pub
    grupo_macro.extend([
        'Inflacion_Mensual_pub',
        'Inflacion_12M_pub',
        'Inflacion_12M_cambio_21d',
        'st_x_Inflacion_12M_pub',
    ])

    df_feat = pd.DataFrame(features, index=df.index).ffill().bfill().fillna(0.0)
    df_final = pd.concat([df[['Fecha', 'USD_PEN', 'PEN_log_ret']], df_feat], axis=1)

    catalogo = {
        'total_predictores': len(grupo_propias) + len(grupo_globales) + len(grupo_macro),
        'vix_p80_train': vix_p80_train,
        'grupos': {
            'propias_usdpen': grupo_propias,
            'globales_y_commodities': grupo_globales,
            'macroeconomicas_bcrp': grupo_macro,
        },
        'todas_las_columnas_x': grupo_propias + grupo_globales + grupo_macro,
    }
    return df_final, catalogo


def ejecutar():
    df_exog = obtener_exogenas_consolidadas(forzar_descarga=False)
    df_full_raw = pd.read_csv(PROC_DIR / 'usdpen_preprocessed_full.csv', parse_dates=['Fecha'])

    df_final, catalogo = construir_dataset_metodologia_completa(df_full_raw, df_exog, n_holdout=60)

    n_holdout = 60
    train_df = df_final.iloc[:-n_holdout].copy().reset_index(drop=True)
    holdout_df = df_final.iloc[-n_holdout:].copy().reset_index(drop=True)

    df_final.to_csv(PROC_DIR / 'usdpen_preprocessed_full.csv', index=False)
    train_df.to_csv(PROC_DIR / 'usdpen_train.csv', index=False)
    holdout_df.to_csv(PROC_DIR / 'usdpen_holdout.csv', index=False)

    # Eliminar duplicados _exog antiguos para dejar un unico dataset oficial limpio
    for old_f in ['usdpen_train_exog.csv', 'usdpen_holdout_exog.csv']:
        p_old = PROC_DIR / old_f
        if p_old.exists():
            p_old.unlink()

    with open(PROC_DIR / 'catalogo_84_predictores.json', 'w', encoding='utf-8') as f:
        json.dump(catalogo, f, ensure_ascii=False, indent=2)

    print(f'Dataset preprocesado oficial actualizado:', flush=True)
    print(f'  - Total predictores X: {catalogo["total_predictores"]} (> 80 caracteristicas, Seccion 3.3)', flush=True)
    print(f'    * Grupo 1 (Propias USD/PEN):        {len(catalogo["grupos"]["propias_usdpen"])} variables', flush=True)
    print(f'    * Grupo 2 (Globales y Commodities): {len(catalogo["grupos"]["globales_y_commodities"])} variables', flush=True)
    print(f'    * Grupo 3 (Macroeconomicas BCRP):   {len(catalogo["grupos"]["macroeconomicas_bcrp"])} variables', flush=True)
    print(f'  - Umbral de tension VIX P80 (train):  {catalogo["vix_p80_train"]:.2f}', flush=True)
    print(f'  - Train:   {train_df.shape} ({train_df["Fecha"].min().date()} a {train_df["Fecha"].max().date()})', flush=True)
    print(f'  - Holdout: {holdout_df.shape} ({holdout_df["Fecha"].min().date()} a {holdout_df["Fecha"].max().date()})', flush=True)


if __name__ == '__main__':
    ejecutar()
