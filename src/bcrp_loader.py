"""
Módulo de Extracción de Datos Macroeconómicos del Banco Central de Reserva del Perú (BCRP).
Descarga e integra series oficiales del portal BCRPData API.

Series descargadas:
- PN38705PM: IPC General Lima Metropolitana (Índice Dic.2021 = 100)
- PN38706PM: IPC Sin Alimentos y Bebidas
- PN38707PM: IPC Sin Alimentos y Energía
- PN38708PM: IPC Subyacente
- PN38709PM: IPC Bienes Transables
- PN38710PM: IPC Bienes No Transables
- PN01271PM: Inflación Variación % Mensual
- PN01273PM: Inflación Variación % 12 Meses
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

BCRP_IPC_SERIES: Dict[str, str] = {
    "PN38705PM": "IPC_General",
    "PN38706PM": "IPC_SinAlimBeb",
    "PN38707PM": "IPC_SinAlimEnergia",
    "PN38708PM": "IPC_Subyacente",
    "PN38709PM": "IPC_Transables",
    "PN38710PM": "IPC_NoTransables",
    "PN01271PM": "Inflacion_Var_Mensual",
    "PN01273PM": "Inflacion_Var_12M",
}

MONTH_MAP: Dict[str, str] = {
    "Ene": "01",
    "Feb": "02",
    "Mar": "03",
    "Abr": "04",
    "May": "05",
    "Jun": "06",
    "Jul": "07",
    "Ago": "08",
    "Set": "09",
    "Sep": "09",
    "Oct": "10",
    "Nov": "11",
    "Dic": "12",
}


class BCRPLoader:
    """Descargador e integrador de series estadísticas desde la API de BCRPData."""

    BASE_URL = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api/{code}/json/{start_date}/{end_date}/"

    @classmethod
    def fetch_series(
        cls,
        code: str,
        name: str,
        start_date: str = "1994-1",
        end_date: str = "2026-10",
    ) -> pd.DataFrame:
        """Descarga una serie individual del BCRP y la estructura como DataFrame mensual."""
        url = cls.BASE_URL.format(code=code, start_date=start_date, end_date=end_date)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

        try:
            with urllib.request.urlopen(req, timeout=20) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as e:
            logger.error(f"Error al descargar serie {code} ({name}): {e}")
            raise

        records = []
        for period in data.get("periods", []):
            period_str = period.get("name", "")
            parts = period_str.replace(".", " ").split()
            if len(parts) >= 2:
                m_str, y_str = parts[0][:3].capitalize(), parts[1]
                m_num = MONTH_MAP.get(m_str, "01")
                dt = pd.to_datetime(f"{y_str}-{m_num}-01")

                vals = period.get("values", [])
                val_num = None
                if vals and vals[0] not in ["n.d.", "", None]:
                    try:
                        val_num = float(vals[0])
                    except ValueError:
                        val_num = None

                records.append({"Mes": dt, name: val_num})

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("Mes").reset_index(drop=True)
        return df

    @classmethod
    def fetch_all_ipc_series(
        cls,
        start_date: str = "1994-1",
        end_date: str = "2026-10",
        cache_path: Optional[str] = "data/raw/bcrp_ipc_1994_2026.csv",
        force_download: bool = False,
    ) -> pd.DataFrame:
        """
        Descarga todas las series oficiales de IPC del BCRP y las consolida.
        Utiliza caché local en cache_path si ya existe.
        """
        if cache_path and os.path.exists(cache_path) and not force_download:
            logger.info(f"Cargando series IPC BCRP desde caché local: {cache_path}")
            df_cached = pd.read_csv(cache_path)
            df_cached["Mes"] = pd.to_datetime(df_cached["Mes"])
            return df_cached

        logger.info("Descargando series oficiales de IPC desde BCRPData API (1994 a 2026)...")
        dfs = []
        for code, name in BCRP_IPC_SERIES.items():
            df_s = cls.fetch_series(code=code, name=name, start_date=start_date, end_date=end_date)
            dfs.append(df_s.set_index("Mes"))

        bcrp_df = pd.concat(dfs, axis=1).reset_index()
        bcrp_df = bcrp_df.sort_values("Mes").reset_index(drop=True)

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            bcrp_df.to_csv(cache_path, index=False)
            logger.info(f"Series BCRP guardadas en caché local: {cache_path}")

        return bcrp_df
