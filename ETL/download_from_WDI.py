# -*- coding: utf-8 -*-

import wbgapi as wbg
import pandas as pd
import argparse
import os

SERIES = [
    # CO2
    "EN.GHG.CO2.LU.MT.CE.AR5", "EN.GHG.CO2.MT.CE.AR5", "EN.GHG.CO2.PC.CE.AR5",
    "EN.GHG.CO2.RT.GDP.PP.KD", "EN.GHG.CO2.ZG.AR5", "EN.GHG.CO2.TR.MT.CE.AR5",
    # PIB
    "NY.GDP.MKTP.CD", "NY.GDP.PCAP.CD",
    # Poblacion
    "SP.POP.TOTL",
    "SP.URB.TOTL.IN.ZS",
    # Educacion
    "SE.ADT.LITR.ZS", "SE.SEC.ENRR", "SE.TER.ENRR",
    # Energia
    "EG.USE.PCAP.KG.OE", "EG.USE.ELEC.KH.PC", "EG.FEC.RNEW.ZS",
    "EG.GDP.PUSE.KO.PP.KD",
    "EG.ELC.COAL.ZS", "EG.ELC.FOSL.ZS", "EG.ELC.HYRO.ZS", "EG.ELC.NGAS.ZS",
    "EG.ELC.NUCL.ZS", "EG.ELC.PETR.ZS", "EG.ELC.RNEW.ZS", "EG.ELC.RNWX.KH",
    "EG.ELC.RNWX.ZS", "EG.FEC.RNEW.ZS", "EG.USE.COMM.FO.ZS", "EG.USE.COMM.GD.PP.KD",
    "EG.USE.CRNW.ZS",
    # Invesment
    "IE.PPI.ENGY.CD", "IE.PPI.ICTI.CD", "IE.PPI.TRAN.CD", "IE.PPI.WATR.CD",
    "IE.PPN.ENGY.CD", "IE.PPN.ICTI.CD", "IE.PPN.TRAN.CD", "IE.PPN.WATR.CD",
    # I+D / Tech / Educación
    "GB.XPD.RSDV.GD.ZS", "IT.NET.USER.ZS", "SE.XPD.TOTL.GD.ZS"
]


def download_wdi(output_file: str, series: list=SERIES, start: int=1970, end: int=2024):
    """Descarga datos de WDI y los guarda en CSV."""
    df = wbg.data.DataFrame(series, time=range(start, end+1), labels=True,
                            params={'per_page': 500})
    df = df.reset_index().rename(columns={"economy": "iso3c", "Time": "year"})

    meta = pd.DataFrame([{
        "iso3c": e["id"],
        "country": e["value"],
        "region": e["region"],
        "income_level": e["incomeLevel"]
    } for e in wbg.economy.list()])

    out = df.merge(meta, on="iso3c", how="left")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    out.to_csv(output_file, index=False)
    print(f"✅ Datos descargados y guardados en {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descarga datos de WDI (CO2 + indicadores).")
    parser.add_argument("--output", type=str, default="data/raw_data/wb_co2_and_indicators.csv",
                        help="Ruta del archivo CSV de salida.")
    parser.add_argument("--series", nargs="+", default=SERIES, 
                        help="Series a descargar (ej: --series NY.GDP.MKTP.CD EN.ATM.CO2E.PC)")
    parser.add_argument("--start", type=int, default=1970, help="Año inicial.")
    parser.add_argument("--end", type=int, default=2022, help="Año final.")
    args = parser.parse_args()

    download_wdi(args.output, args.series, args.start, args.end)

