"""
SolFX-Predictor | Descargador e Integrador de Variables Exogenas Globales y Commodities (1994 - 2026)

Integra sin data leakage (shift(1) causal estricto):
  1. Indice de Volatilidad CBOE VIX (^VIX, 1994-2026)
  2. Indice del Dolar Estadounidense DXY (DX-Y.NYB, 1994-2026)
  3. Rendimiento del Bono del Tesoro de EE. UU. a 10 anos US10Y (^TNX / USGG10YR, 1994-2026)
  4. Indice Bursatil S&P 500 (^GSPC, 1994-2026)
  5. Precio Internacional del Cobre (FRED PCOPPUSDM 1994-2000 + futuros COMEX HG=F 2000-2026)
  6. Precio Internacional del Oro (Futuros COMEX GC=F 2000-2026 + serie historica 1994-2000)
  7. Precio Internacional del Petroleo WTI (FRED DCOILWTICO diario 1994-2026 + futuros CL=F)
  8. Inflacion Interanual BCRP con rezago de publicacion segura (42 dias habiles ~ 2 meses)
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
    'WTI_FRED': 'DCOILWTICO',     # Petroleo WTI diario 1986-2026 (USD/barril)
    'Cobre_FRED': 'PCOPPUSDM',    # Cobre LME FMI 1990-2026 (USD/tonelada metrica)
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
    date_col = df.columns[0]
    val_col = df.columns[1]
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
        print(f'  Yahoo {nombre} ({tk}): {len(s)} obs ({s.index.min().date()} a {s.index.max().date()})', flush=True)
        series_dict[nombre] = s

    for nombre, fid in FRED_SERIES.items():
        try:
            s = descargar_fred_serie(nombre, fid)
            print(f'  FRED {nombre} ({fid}): {len(s)} obs ({s.index.min().date()} a {s.index.max().date()})', flush=True)
            series_dict[nombre] = s
        except Exception as e:
            print(f'  [Aviso] FRED {nombre}: {e}', flush=True)

    # Calendario habil completo 1994-01-03 a 2026-10-30
    idx = pd.date_range('1994-01-03', '2026-10-30', freq='B', name='Fecha')
    df = pd.DataFrame(index=idx)
    for k, s in series_dict.items():
        df[k] = s.reindex(idx)

    # Imputacion causal LOCF (ffill)
    df = df.ffill()

    fecha_empalme = pd.Timestamp('2000-09-01')

    # 1. Petroleo WTI: FRED diario cubre 1994-2026 completo; complementamos con WTI_YF donde aplique
    if 'WTI_FRED' in df.columns:
        wti = df['WTI_FRED'].copy()
        wti.loc[fecha_empalme:] = df['WTI_YF'].loc[fecha_empalme:].combine_first(wti.loc[fecha_empalme:])
        df['Petroleo_WTI'] = wti.ffill().bfill()
    else:
        df['Petroleo_WTI'] = df['WTI_YF'].ffill().bfill()

    # 2. Cobre: Futuro COMEX HG=F (2000-08-30 a 2026-10-30) + empalme causal 1994-2000 con Cobre_FRED / FCX
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

    # 3. Oro: Futuro COMEX GC=F (2000-08-30 a 2026-10-30) + empalme causal 1994-2000 con NEM_Proxy_Au
    oro = df['Oro_YF'].copy()
    base_oro = df['NEM_Proxy_Au'].ffill().bfill()
    ratio_au = float(oro.loc[fecha_empalme] / base_oro.loc[fecha_empalme])
    oro.loc[:fecha_empalme] = base_oro.loc[:fecha_empalme] * ratio_au
    df['Oro'] = oro.ffill().bfill()

    cols_finales = ['VIX', 'DXY', 'US10Y', 'SP500', 'Cobre', 'Oro', 'Petroleo_WTI']
    df[cols_finales] = df[cols_finales].ffill().bfill()
    df_out = df[cols_finales].reset_index()
    df_out.to_csv(ruta_cache, index=False)
    print(f'Archivo crudo de exogenas guardado en: {ruta_cache} ({len(df_out)} filas)', flush=True)
    return df_out


def construir_features_exogenas_causales(df_base: pd.DataFrame, df_exog: pd.DataFrame) -> pd.DataFrame:
    """
    Une las series exogenas diarias al dataset preprocesado y construye indicadores
    multiescala (1, 5 y 20 sesiones) usando ESTRICTAMENTE informacion hasta t-1 (shift(1)).
    """
    df = pd.merge(df_base.sort_values('Fecha'), df_exog.sort_values('Fecha'), on='Fecha', how='left')
    cols_exog = ['VIX', 'DXY', 'US10Y', 'SP500', 'Cobre', 'Oro', 'Petroleo_WTI']
    df[cols_exog] = df[cols_exog].ffill().bfill()

    nuevas = {}

    # 1. Activos en precios (Cobre, DXY, Oro, Petroleo_WTI, SP500):
    #    Siempre partimos del precio rezagado P_{t-1} = serie.shift(1) para evitar data leakage!
    for activo in ['Cobre', 'DXY', 'Oro', 'Petroleo_WTI', 'SP500']:
        p_lag1 = df[activo].shift(1)
        nuevas[f'{activo}_ret_1d'] = np.log(p_lag1 / p_lag1.shift(1))
        nuevas[f'{activo}_ret_5d'] = np.log(p_lag1 / p_lag1.shift(5))
        nuevas[f'{activo}_ret_20d'] = np.log(p_lag1 / p_lag1.shift(20))
        nuevas[f'{activo}_ret_60d'] = np.log(p_lag1 / p_lag1.shift(60))
        sma20 = p_lag1.rolling(20, min_periods=5).mean()
        sma60 = p_lag1.rolling(60, min_periods=10).mean()
        nuevas[f'{activo}_vs_SMA_20'] = (p_lag1 - sma20) / (sma20 + 1e-12)
        nuevas[f'{activo}_vs_SMA_60'] = (p_lag1 - sma60) / (sma60 + 1e-12)

    # 2. Indice de Volatilidad VIX (nivel rezagado, cambio a 1, 5 y 20 dias y z-score movil de 20 dias):
    vix_lag1 = df['VIX'].shift(1)
    vix_sma20 = vix_lag1.rolling(20, min_periods=5).mean()
    vix_std20 = vix_lag1.rolling(20, min_periods=5).std().replace(0, np.nan)
    nuevas['VIX_nivel_lag1'] = vix_lag1
    nuevas['VIX_cambio_1d'] = vix_lag1 - vix_lag1.shift(1)
    nuevas['VIX_cambio_5d'] = vix_lag1 - vix_lag1.shift(5)
    nuevas['VIX_cambio_20d'] = vix_lag1 - vix_lag1.shift(20)
    nuevas['VIX_zscore_20d'] = ((vix_lag1 - vix_sma20) / (vix_std20 + 1e-8)).fillna(0.0)

    # 3. Tasa del Bono del Tesoro de EE. UU. a 10 anos (US10Y en %):
    us10y_lag1 = df['US10Y'].shift(1)
    us10y_sma20 = us10y_lag1.rolling(20, min_periods=5).mean()
    nuevas['US10Y_nivel_lag1'] = us10y_lag1
    nuevas['US10Y_cambio_1d'] = us10y_lag1 - us10y_lag1.shift(1)
    nuevas['US10Y_cambio_5d'] = us10y_lag1 - us10y_lag1.shift(5)
    nuevas['US10Y_cambio_20d'] = us10y_lag1 - us10y_lag1.shift(20)
    nuevas['US10Y_vs_SMA_20'] = us10y_lag1 - us10y_sma20

    # 4. Terminos de intercambio proxy (Ratio Cobre / Petroleo WTI en t-1) y Ratio Cobre / DXY:
    ratio_cu_wti = (df['Cobre'].shift(1) / (df['Petroleo_WTI'].shift(1) + 1e-8))
    nuevas['TermIntercambio_Cu_WTI_ret_20d'] = np.log(ratio_cu_wti / ratio_cu_wti.shift(20))
    ratio_cu_dxy = (df['Cobre'].shift(1) / (df['DXY'].shift(1) + 1e-8))
    nuevas['Cobre_vs_DXY_ret_20d'] = np.log(ratio_cu_dxy / ratio_cu_dxy.shift(20))

    # 5. Inflacion BCRP con rezago de publicacion segura (42 dias habiles ~ 2 meses calendario):
    if 'Inflacion_Var_12M' in df.columns:
        inf_safe = df['Inflacion_Var_12M'].shift(42)
        nuevas['Inflacion_12M_pub_segura'] = inf_safe
        nuevas['Inflacion_12M_cambio_20d'] = inf_safe - inf_safe.shift(20)

    # 6. Momentum de mediano plazo del propio USD/PEN (acumulado de 5 y 20 sesiones hasta t-1):
    p_pen_lag1 = df['USD_PEN'].shift(1)
    nuevas['PEN_ret_acum_5d'] = np.log(p_pen_lag1 / p_pen_lag1.shift(5))
    nuevas['PEN_ret_acum_20d'] = np.log(p_pen_lag1 / p_pen_lag1.shift(20))

    df_nuevas = pd.DataFrame(nuevas, index=df.index).bfill().fillna(0.0)
    df_final = pd.concat([df, df_nuevas], axis=1)
    return df_final


def ejecutar():
    df_exog = obtener_exogenas_consolidadas(forzar_descarga=False)
    df_full = pd.read_csv(PROC_DIR / 'usdpen_preprocessed_full.csv', parse_dates=['Fecha'])

    # Eliminar columnas exogenas previas si ya se habia corrido este script
    cols_previas = [c for c in df_full.columns if any(
        c.startswith(pref) for pref in [
            'VIX', 'DXY', 'US10Y', 'SP500', 'Cobre', 'Oro', 'Petroleo_WTI',
            'TermIntercambio_', 'Inflacion_12M_', 'PEN_ret_acum_'
        ]
    )]
    if cols_previas:
        df_full = df_full.drop(columns=cols_previas)

    df_enriquecido = construir_features_exogenas_causales(df_full, df_exog)

    # Particion estricta: 8462 entrenamiento y 60 holdout (mismas fechas exactas)
    n_holdout = 60
    train_df = df_enriquecido.iloc[:-n_holdout].copy().reset_index(drop=True)
    holdout_df = df_enriquecido.iloc[-n_holdout:].copy().reset_index(drop=True)

    df_enriquecido.to_csv(PROC_DIR / 'usdpen_preprocessed_full.csv', index=False)
    train_df.to_csv(PROC_DIR / 'usdpen_train.csv', index=False)
    holdout_df.to_csv(PROC_DIR / 'usdpen_holdout.csv', index=False)

    train_df.to_csv(PROC_DIR / 'usdpen_train_exog.csv', index=False)
    holdout_df.to_csv(PROC_DIR / 'usdpen_holdout_exog.csv', index=False)

    print(f'Datasets enriquecidos guardados exitosamente:', flush=True)
    print(f'  - Train:   {train_df.shape} ({train_df["Fecha"].min().date()} a {train_df["Fecha"].max().date()})', flush=True)
    print(f'  - Holdout: {holdout_df.shape} ({holdout_df["Fecha"].min().date()} a {holdout_df["Fecha"].max().date()})', flush=True)


if __name__ == '__main__':
    ejecutar()
