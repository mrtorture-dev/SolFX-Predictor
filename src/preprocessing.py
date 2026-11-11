"""
SolFX-Predictor: Pipeline de Preprocesamiento e Imputación de Series Temporales
Basado en la metodología de:
    Palma Arzubialde, C. y Salazar Sedano, J. G. (2025).
    "Predicción multivariable del tipo de cambio USD/PEN mediante modelos de machine learning,
    deep learning y modelos fundacionales de series de tiempo". Tesis de Grado.

Principios Metodológicos Clave:
1. Target: Log-retorno diario r_t = ln(P_t / P_{t-1}).
2. Imputación temporal estricta:
   - Alineación sobre días hábiles de cotización (business days).
   - Días feriados / sin cotización no coincidentes entre mercados (Lima, NY, Londres):
     Forward Fill / Last Observation Carried Forward (LOCF) para evitar lookahead bias.
   - Frecuencias mixtas (ej. IPC mensual): Propagación continua hacia adelante (ffill).
3. Detección y Tratamiento de Valores Atípicos:
   - Diagnóstico mediante Rango Intercuartílico (IQR / Tukey fences) y Winsorización opcional.
   - Escalamiento robusto con RobustScaler (mediana y rango intercuartílico).
4. Pruebas de Estacionariedad:
   - Test Augmented Dickey-Fuller (ADF) y KPSS.
5. Ingeniería de Características con Control Anti-Leakage (Algoritmo 2):
   - Desplazamiento shift(1) estricto antes de calcular indicadores derivados sobre exógenas.
   - Rezagos autorregresivos (AR): r_{t-1}, r_{t-2}, r_{t-3}, r_{t-5}, r_{t-10}, r_{t-21}.
   - Volatilidad histórica / GARCH proxy: rolling std de 21 días sobre r_{t-1}.
   - RSI(14) causal normalizado a [-1, 1].
   - MACD(12, 26, 9) causal normalizado por el precio rezagado.
   - Distancia a Medias Móviles Simples (SMA 5, 10, 20): (P_{t-1} - SMA) / SMA.
   - Transformaciones periódicas estacionales: sin/cos de mes (12) y día de la semana (7).
6. Partición Temporal:
   - Holdout estricto de las últimas N_HOLDOUT = 60 observaciones para evaluación final fuera de muestra.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuración estándar de preprocesamiento según Palma y Salazar (2025)."""
    date_col: str = "Fecha"
    price_col: str = "USD_PEN"
    target_col: str = "PEN_log_ret"
    n_holdout: int = 60
    min_train_size: int = 252
    step_size: int = 21
    ar_lags: Tuple[int, ...] = (1, 2, 3, 5, 10, 21)
    sma_windows: Tuple[int, ...] = (5, 10, 20)
    rsi_period: int = 14
    vol_window: int = 21
    fill_method: str = "ffill"  # LOCF (Last Observation Carried Forward)
    outlier_iqr_factor: float = 3.0  # Fences para detección de anomalías severas


class FXDataCleaner:
    """Maneja la carga inicial, limpieza de inconsistencias y depuración de celdas corruptas."""

    @staticmethod
    def clean_bloomberg_raw(
        df: pd.DataFrame,
        date_col_name: str = "Date",
        price_col_name: str = "PX_LAST",
        bid_col_name: Optional[str] = "PX_BID",
    ) -> pd.DataFrame:
        """
        Limpia un DataFrame crudo exportado desde Bloomberg Terminal.
        Elimina filas con fórmulas no resueltas (#NAME?), textos o fechas inválidas.
        """
        df_clean = df.copy()

        # Si viene con nombres alternativos de columna
        if date_col_name not in df_clean.columns:
            # Buscar columna que contenga fecha
            candidate = [c for c in df_clean.columns if "fecha" in str(c).lower() or "date" in str(c).lower()]
            if candidate:
                date_col_name = candidate[0]
            else:
                date_col_name = df_clean.columns[0]

        # Limpiar errores típicos como #NAME?, cadenas vacías, valores nulos
        df_clean = df_clean[df_clean[date_col_name].notna()]
        df_clean = df_clean[~df_clean[date_col_name].astype(str).str.contains(r"#NAME|\=BDH", na=False)]

        # Conversión a datetime
        df_clean["Fecha"] = pd.to_datetime(df_clean[date_col_name], errors="coerce")
        df_clean = df_clean[df_clean["Fecha"].notna()]

        # Seleccionar precio
        if price_col_name in df_clean.columns:
            df_clean["USD_PEN"] = pd.to_numeric(df_clean[price_col_name], errors="coerce")
        else:
            # Buscar columna numérica que parezca el precio spot (rango 1.5 a 5.0 para USD/PEN)
            for c in df_clean.columns:
                if c != "Fecha":
                    s = pd.to_numeric(df_clean[c], errors="coerce")
                    if s.dropna().median() is not None and 1.5 < s.dropna().median() < 5.0:
                        df_clean["USD_PEN"] = s
                        break

        if bid_col_name and bid_col_name in df_clean.columns:
            df_clean["PX_BID"] = pd.to_numeric(df_clean[bid_col_name], errors="coerce")

        df_clean = df_clean[df_clean["USD_PEN"].notna()]
        df_clean = df_clean.sort_values("Fecha").drop_duplicates(subset=["Fecha"]).reset_index(drop=True)

        return df_clean[["Fecha", "USD_PEN"] + (["PX_BID"] if "PX_BID" in df_clean.columns else [])]


class FXCalendarAligner:
    """
    Alineación de calendario e imputación estricta sin data leakage.
    Garantiza que feriados locales y festivos internacionales se sincronicen usando
    propagación hacia adelante (LOCF).
    """

    @staticmethod
    def align_business_days(
        df: pd.DataFrame,
        date_col: str = "Fecha",
        freq: str = "B",
        forward_fill: bool = True,
    ) -> pd.DataFrame:
        """
        Homologa las fechas sobre el calendario hábil bursátil (freq='B').
        Imputa observaciones intermedias de feriados/días no coincidentes con ffill (LOCF).
        """
        df_sorted = df.sort_values(date_col).copy()
        df_sorted = df_sorted.set_index(date_col)

        start_date = df_sorted.index.min()
        end_date = df_sorted.index.max()

        full_calendar = pd.date_range(start=start_date, end=end_date, freq=freq, name=date_col)
        df_reindexed = df_sorted.reindex(full_calendar)

        if forward_fill:
            # Imputación hacia adelante: el precio del feriado se asume igual al último día hábil cotizado
            df_reindexed = df_reindexed.ffill()
            # En caso excepcional de nulos iniciales (bfill residual solo si el primer día es nulo)
            df_reindexed = df_reindexed.bfill()

        df_aligned = df_reindexed.reset_index()
        return df_aligned

    @staticmethod
    def merge_monthly_macro(
        daily_df: pd.DataFrame,
        monthly_df: pd.DataFrame,
        daily_date_col: str = "Fecha",
        monthly_date_col: Optional[str] = None,
        macro_prefix: str = "IPC_",
    ) -> pd.DataFrame:
        """
        Fusiona variables macroeconómicas de baja frecuencia (ej. IPC mensual de BCRP o Bloomberg) con la serie diaria.
        Aplica forward-fill para mantener la última información conocida disponible hasta el día t.
        Aplica shift(1) para garantizar el protocolo anti-leakage (conocer la información previa al día t).
        """
        daily = daily_df.sort_values(daily_date_col).copy()
        monthly = monthly_df.copy()

        # Detección automática de la columna de fecha en el DataFrame mensual
        if monthly_date_col is None or monthly_date_col not in monthly.columns:
            for cand in ["Mes", "Mes de referencia", "Fecha", "Date", "Dates"]:
                if cand in monthly.columns:
                    monthly_date_col = cand
                    break
            if monthly_date_col is None:
                monthly_date_col = monthly.columns[0]

        monthly["join_month"] = pd.to_datetime(monthly[monthly_date_col]).dt.to_period("M")
        daily["join_month"] = pd.to_datetime(daily[daily_date_col]).dt.to_period("M")

        feature_cols = [c for c in monthly.columns if c not in [monthly_date_col, "join_month"]]
        rename_map = {}
        for c in feature_cols:
            if c.startswith("IPC_") or c.startswith("Inflacion_"):
                rename_map[c] = c
            else:
                rename_map[c] = f"{macro_prefix}{c}"

        monthly_sub = monthly[["join_month"] + feature_cols].rename(columns=rename_map)

        merged = pd.merge(daily, monthly_sub, on="join_month", how="left")
        merged = merged.drop(columns=["join_month"])

        # Forward fill continuo (LOCF)
        macro_cols = list(rename_map.values())
        merged[macro_cols] = merged[macro_cols].ffill()

        # Si hay primeros meses nulos (por ejemplo, serie diaria empieza antes del primer mes de IPC),
        # solo se rellena bfill residual si es estrictamente necesario, o se mantienen
        if merged[macro_cols].iloc[0].isna().any():
            merged[macro_cols] = merged[macro_cols].bfill()

        # Protocolo Anti-Leakage: Las variables macro se rezagan un día hábil (shift(1))
        # para asegurar que al predecir el día t solo se use el dato conocido hasta t-1
        for col in macro_cols:
            merged[f"{col}_lag_1"] = merged[col].shift(1)

        return merged


class AntiLeakageFeatureEngineer:
    """
    Construcción causal de variables explicativas e indicadores técnicos
    según el Algoritmo 2 de Palma y Salazar (2025).
    Principio fundamental: Toda variable exógena e indicador derivado se calcula
    utilizando información disponible estrictamente hasta t-1.
    """

    @staticmethod
    def compute_target(df: pd.DataFrame, price_col: str = "USD_PEN") -> pd.Series:
        """
        Calcula la variable objetivo: Log-retorno diario r_t = ln(P_t / P_{t-1}).
        """
        prices = df[price_col].astype(float)
        return np.log(prices / prices.shift(1))

    @staticmethod
    def compute_ar_lags(
        returns: pd.Series,
        lags: Tuple[int, ...] = (1, 2, 3, 5, 10, 21),
        prefix: str = "PEN_log_ret_lag_",
    ) -> pd.DataFrame:
        """
        Rezagos autorregresivos del log-retorno.
        """
        lags_df = pd.DataFrame(index=returns.index)
        for lag in lags:
            lags_df[f"{prefix}{lag}"] = returns.shift(lag)
        return lags_df

    @staticmethod
    def compute_sma_features(
        prices: pd.Series,
        windows: Tuple[int, ...] = (5, 10, 20),
    ) -> pd.DataFrame:
        """
        Medias Móviles Simples y distancias porcentuales relativas SIN LEAKAGE.
        Aplica shift(1) antes de calcular la media móvil para usar solo precios hasta t-1.
        """
        sma_df = pd.DataFrame(index=prices.index)
        prices_lagged = prices.shift(1)

        for w in windows:
            sma = prices_lagged.rolling(window=w, min_periods=max(2, w // 2)).mean()
            sma_df[f"SMA_{w}"] = sma
            sma_df[f"price_vs_SMA_{w}"] = (prices_lagged - sma) / (sma + 1e-12)

        return sma_df

    @staticmethod
    def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index (RSI) SIN LEAKAGE.
        Calculado sobre prices.shift(1) y normalizado en el rango [-1, 1].
        """
        prices_lagged = prices.shift(1)
        delta = prices_lagged.diff()

        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period, min_periods=period // 2).mean()
        avg_loss = loss.rolling(window=period, min_periods=period // 2).mean()

        rs = avg_gain / (avg_loss + 1e-12)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Normalización a [-1, 1] como describe la tesis
        rsi_normalized = (rsi - 50.0) / 50.0
        return rsi_normalized

    @staticmethod
    def compute_macd(
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """
        MACD causal normalizado por el precio rezagado.
        """
        prices_lagged = prices.shift(1)

        ema_fast = prices_lagged.ewm(span=fast, adjust=False).mean()
        ema_slow = prices_lagged.ewm(span=slow, adjust=False).mean()

        macd_line = (ema_fast - ema_slow) / (prices_lagged + 1e-12)
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return pd.DataFrame(
            {
                "MACD_line": macd_line,
                "MACD_signal": signal_line,
                "MACD_histogram": histogram,
            },
            index=prices.index,
        )

    @staticmethod
    def compute_volatility(
        returns: pd.Series,
        window: int = 21,
    ) -> pd.Series:
        """
        Estimador de volatilidad histórica causal (proxy GARCH):
        Desviación estándar móvil sobre retornos desplazados shift(1) con ventana de 21 días bursátiles.
        """
        returns_lagged = returns.shift(1)
        vol = returns_lagged.rolling(window=window, min_periods=max(5, window // 2)).std()
        return vol

    @staticmethod
    def compute_cyclical_calendar(dates: pd.Series) -> pd.DataFrame:
        """
        Codificación estacional periódica seno-coseno para día de semana y mes.
        """
        dt = pd.to_datetime(dates)
        month = dt.dt.month
        day_of_week = dt.dt.dayofweek

        cal_df = pd.DataFrame(
            {
                "month_sin": np.sin(2.0 * np.pi * month / 12.0),
                "month_cos": np.cos(2.0 * np.pi * month / 12.0),
                "day_of_week_sin": np.sin(2.0 * np.pi * day_of_week / 7.0),
                "day_of_week_cos": np.cos(2.0 * np.pi * day_of_week / 7.0),
            },
            index=dates.index,
        )
        return cal_df

    @staticmethod
    def lag_exogenous_features(
        df: pd.DataFrame,
        exog_cols: List[str],
        lag: int = 1,
    ) -> pd.DataFrame:
        """
        Aplica shift(lag) a covariables exógenas puras antes de cualquier análisis.
        """
        lagged_df = pd.DataFrame(index=df.index)
        for col in exog_cols:
            if col in df.columns:
                lagged_df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        return lagged_df


class OutlierHandler:
    """Diagnóstico y escalamiento robusto de valores atípicos."""

    @staticmethod
    def get_iqr_bounds(series: pd.Series, factor: float = 3.0) -> Tuple[float, float, float]:
        """
        Calcula límites de Tukey: [Q1 - factor*IQR, Q3 + factor*IQR] y mediana.
        """
        clean = series.dropna()
        q1 = clean.quantile(0.25)
        q3 = clean.quantile(0.75)
        iqr = q3 - q1
        median = clean.median()
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        return lower, upper, median

    @staticmethod
    def robust_scale_series(series: pd.Series) -> pd.Series:
        """
        Aplica RobustScaler: z = (x - mediana) / IQR.
        """
        clean = series.dropna()
        q1 = clean.quantile(0.25)
        q3 = clean.quantile(0.75)
        iqr = q3 - q1
        median = clean.median()
        if iqr == 0:
            iqr = clean.std() or 1.0
        return (series - median) / iqr


class StationarityTester:
    """Ejecuta los tests estadísticos de estacionariedad Augmented Dickey-Fuller y KPSS."""

    @staticmethod
    def run_tests(series: pd.Series, series_name: str = "Series") -> Dict[str, Union[float, str, bool]]:
        results: Dict[str, Union[float, str, bool]] = {"name": series_name}
        clean = series.dropna()

        if len(clean) < 30:
            results["error"] = "Insuficientes observaciones para pruebas estadísticas"
            return results

        try:
            from statsmodels.tsa.stattools import adfuller, kpss

            # Augmented Dickey-Fuller
            adf_res = adfuller(clean, autolag="AIC")
            results["adf_stat"] = float(adf_res[0])
            results["adf_pvalue"] = float(adf_res[1])
            results["adf_is_stationary"] = bool(adf_res[1] < 0.05)

            # KPSS
            kpss_res = kpss(clean, regression="c", nlags="auto")
            results["kpss_stat"] = float(kpss_res[0])
            results["kpss_pvalue"] = float(kpss_res[1])
            results["kpss_is_stationary"] = bool(kpss_res[1] >= 0.05)

        except Exception as e:
            # Fallback numérico si statsmodels no está disponible
            mean = clean.mean()
            std = clean.std()
            results["mean"] = float(mean)
            results["std"] = float(std)
            results["note"] = f"statsmodels no disponible ({e})"

        return results


class SolFXPipeline:
    """
    Pipeline principal orquestador del preprocesamiento para SolFX-Predictor.
    Integra limpieza, alineación de calendario, imputación causal,
    ingeniería de características anti-leakage y división train/holdout.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.metadata: Dict = {}

    def run_pipeline(
        self,
        raw_pen_df: pd.DataFrame,
        monthly_macro_df: Optional[pd.DataFrame] = None,
        exog_daily_df: Optional[pd.DataFrame] = None,
        use_bcrp_ipc: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Ejecuta el pipeline completo y retorna:
        (df_full_processed, df_train, df_holdout)

        Args:
            raw_pen_df: DataFrame con las cotizaciones diarias spot del USD/PEN.
            monthly_macro_df: DataFrame opcional con series mensuales (ej. Bloomberg o BCRP).
            exog_daily_df: DataFrame opcional con covariables exógenas diarias adicionales.
            use_bcrp_ipc: Si es True y monthly_macro_df no se suministra o tiene cobertura parcial,
                          descarga e integra automáticamente las series históricas oficiales del BCRP (1994-2026).
        """
        logger.info("Iniciando Pipeline de Preprocesamiento SolFX (Palma & Salazar 2025)...")

        # 1. Limpieza base
        base_df = FXDataCleaner.clean_bloomberg_raw(
            raw_pen_df,
            date_col_name=self.config.date_col,
            price_col_name=self.config.price_col,
        )
        logger.info(f"1. Datos base depurados: {len(base_df)} cotizaciones observadas.")

        # 2. Alineación de calendario e imputación por Forward-Fill (LOCF)
        aligned_df = FXCalendarAligner.align_business_days(
            base_df,
            date_col="Fecha",
            freq="B",
            forward_fill=(self.config.fill_method == "ffill"),
        )
        logger.info(
            f"2. Alineación a calendario hábil completada: {len(aligned_df)} días hábiles "
            f"({aligned_df['Fecha'].min().date()} a {aligned_df['Fecha'].max().date()})."
        )

        # 3. Fusión con variables mensuales (IPC oficial BCRP o suministrado)
        if use_bcrp_ipc:
            try:
                from .bcrp_loader import BCRPLoader
                bcrp_macro = BCRPLoader.fetch_all_ipc_series()
                aligned_df = FXCalendarAligner.merge_monthly_macro(
                    aligned_df,
                    bcrp_macro,
                    daily_date_col="Fecha",
                    monthly_date_col="Mes",
                )
                logger.info("3. Integración e imputación forward de series completas de IPC de BCRP (1994-2026) completada.")
            except Exception as e:
                logger.warning(f"No se pudo cargar API de BCRP ({e}). Intentando con monthly_macro_df...")
                if monthly_macro_df is not None and not monthly_macro_df.empty:
                    aligned_df = FXCalendarAligner.merge_monthly_macro(
                        aligned_df,
                        monthly_macro_df,
                        daily_date_col="Fecha",
                    )
        elif monthly_macro_df is not None and not monthly_macro_df.empty:
            aligned_df = FXCalendarAligner.merge_monthly_macro(
                aligned_df,
                monthly_macro_df,
                daily_date_col="Fecha",
            )
            logger.info("3. Fusión e imputación forward de variables macroeconómicas mensuales completada.")

        # 4. Fusión con covariables diarias adicionales si existen
        if exog_daily_df is not None and not exog_daily_df.empty:
            aligned_df = pd.merge(aligned_df, exog_daily_df, on="Fecha", how="left").ffill()
            logger.info("4. Covariables exógenas diarias fusionadas con imputación LOCF.")

        # 5. Cálculo del target: PEN_log_ret
        aligned_df[self.config.target_col] = AntiLeakageFeatureEngineer.compute_target(
            aligned_df, price_col="USD_PEN"
        )

        # 6. Ingeniería de Características Causal (Anti-Leakage)
        target_series = aligned_df[self.config.target_col]
        price_series = aligned_df["USD_PEN"]

        # 6.1 Rezagos autorregresivos (AR)
        ar_df = AntiLeakageFeatureEngineer.compute_ar_lags(target_series, lags=self.config.ar_lags)

        # 6.2 Indicadores técnicos causales
        sma_df = AntiLeakageFeatureEngineer.compute_sma_features(
            price_series, windows=self.config.sma_windows
        )
        rsi_series = AntiLeakageFeatureEngineer.compute_rsi(
            price_series, period=self.config.rsi_period
        )
        macd_df = AntiLeakageFeatureEngineer.compute_macd(price_series)
        vol_series = AntiLeakageFeatureEngineer.compute_volatility(
            target_series, window=self.config.vol_window
        )

        # 6.3 Características temporales estacionales cíclicas
        cyclical_df = AntiLeakageFeatureEngineer.compute_cyclical_calendar(aligned_df["Fecha"])

        # Consolidar características
        feature_blocks = [
            aligned_df,
            ar_df,
            sma_df,
            pd.DataFrame({"RSI": rsi_series, "GARCH_vol": vol_series}, index=aligned_df.index),
            macd_df,
            cyclical_df,
        ]

        df_processed = pd.concat(feature_blocks, axis=1)

        # 7. Diagnóstico de estacionariedad sobre el target
        stat_results = StationarityTester.run_tests(df_processed[self.config.target_col], series_name="PEN_log_ret")
        logger.info(f"7. Diagnóstico de estacionariedad sobre target: {stat_results}")

        # 8. Diagnóstico de outliers
        low_b, up_b, med = OutlierHandler.get_iqr_bounds(
            df_processed[self.config.target_col], factor=self.config.outlier_iqr_factor
        )
        outliers_count = (
            (df_processed[self.config.target_col] < low_b) | (df_processed[self.config.target_col] > up_b)
        ).sum()
        logger.info(
            f"8. Detección de atípicos severos (IQR*3): {outliers_count} observaciones fuera de [{low_b:.5f}, {up_b:.5f}]"
        )

        # 9. Filtrar filas completas (eliminar NaNs introducidos por rezagos y ventanas)
        lag_cols_to_check = [f"PEN_log_ret_lag_{l}" for l in self.config.ar_lags if f"PEN_log_ret_lag_{l}" in df_processed.columns]
        required_cols = [self.config.target_col, "GARCH_vol", "RSI", "MACD_line"] + lag_cols_to_check
        df_clean_final = df_processed.dropna(subset=required_cols).reset_index(drop=True)
        logger.info(f"9. Dataset estructurado final sin NaNs: {len(df_clean_final)} observaciones.")

        # 10. Partición temporal estricta Train / Holdout (N_HOLDOUT = 60)
        n_holdout = self.config.n_holdout
        if len(df_clean_final) <= n_holdout:
            raise ValueError(f"Observaciones insuficientes ({len(df_clean_final)}) para holdout={n_holdout}")

        train_df = df_clean_final.iloc[:-n_holdout].copy().reset_index(drop=True)
        holdout_df = df_clean_final.iloc[-n_holdout:].copy().reset_index(drop=True)

        logger.info(
            f"10. Partición Train/Holdout realizada:\n"
            f"    - Train:   {len(train_df)} registros ({train_df['Fecha'].min().date()} a {train_df['Fecha'].max().date()})\n"
            f"    - Holdout: {len(holdout_df)} registros ({holdout_df['Fecha'].min().date()} a {holdout_df['Fecha'].max().date()})"
        )

        # Metadata resumen
        self.metadata = {
            "config": asdict(self.config),
            "total_observations_raw": len(base_df),
            "total_observations_aligned": len(aligned_df),
            "total_observations_clean": len(df_clean_final),
            "train_observations": len(train_df),
            "holdout_observations": len(holdout_df),
            "train_period": [str(train_df["Fecha"].min().date()), str(train_df["Fecha"].max().date())],
            "holdout_period": [str(holdout_df["Fecha"].min().date()), str(holdout_df["Fecha"].max().date())],
            "features_generated": [c for c in df_clean_final.columns if c not in ["Fecha", "USD_PEN", self.config.target_col]],
            "stationarity_tests": stat_results,
            "outlier_bounds": {"lower": low_b, "upper": up_b, "median": med, "extreme_count": int(outliers_count)},
        }

        return df_clean_final, train_df, holdout_df

    def save_results(
        self,
        full_df: pd.DataFrame,
        train_df: pd.DataFrame,
        holdout_df: pd.DataFrame,
        output_dir: str = "data/processed",
    ) -> Dict[str, str]:
        """Guarda los datasets generados y el reporte de metadatos."""
        os.makedirs(output_dir, exist_ok=True)

        paths = {
            "full_csv": os.path.join(output_dir, "usdpen_preprocessed_full.csv"),
            "full_parquet": os.path.join(output_dir, "usdpen_preprocessed_full.parquet"),
            "train_csv": os.path.join(output_dir, "usdpen_train.csv"),
            "holdout_csv": os.path.join(output_dir, "usdpen_holdout.csv"),
            "metadata_json": os.path.join(output_dir, "preprocessing_metadata.json"),
        }

        full_df.to_csv(paths["full_csv"], index=False)
        try:
            full_df.to_parquet(paths["full_parquet"], index=False)
        except Exception:
            pass

        train_df.to_csv(paths["train_csv"], index=False)
        holdout_df.to_csv(paths["holdout_csv"], index=False)

        with open(paths["metadata_json"], "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=4, default=str)

        logger.info(f"Artefactos guardados exitosamente en: {output_dir}")
        return paths
