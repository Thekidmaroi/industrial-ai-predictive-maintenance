import pandas as pd
import numpy as np
from pathlib import Path


def load_skab(data_path: str, folders: list, sensor_cols: list) -> pd.DataFrame:
    """Charge et concatène tous les fichiers SKAB."""
    dfs = []
    for folder in folders:
        for f in sorted((Path(data_path) / folder).glob("*.csv")):
            df = pd.read_csv(f, sep=";", index_col="datetime", parse_dates=True)
            df["source"] = folder
            dfs.append(df)

    df_all = pd.concat(dfs, join="outer").sort_index()
    df_all[sensor_cols] = df_all[sensor_cols].fillna(df_all[sensor_cols].median())
    return df_all


def add_rolling_features(df: pd.DataFrame, sensor_cols: list,
                          discriminant_sensors: list, window: int = 30) -> pd.DataFrame:
    """Ajoute les features de variance glissante et le ratio vibration/débit."""
    df_feat = df[sensor_cols].copy()

    for col in discriminant_sensors:
        df_feat[f'{col}_rollstd']  = df[col].rolling(window).std()
        df_feat[f'{col}_rollmean'] = df[col].rolling(window).mean()

    df_feat['vib_flow_ratio'] = (
        (df['Accelerometer1RMS'] + df['Accelerometer2RMS']) /
        (df['Volume Flow RateRMS'] + 1e-6)
    )

    return df_feat.dropna().reset_index(drop=True)


def make_sequences(data: np.ndarray, window: int) -> np.ndarray:
    """Transforme un array 2D en séquences 3D pour LSTM."""
    return np.array([data[i:i+window] for i in range(len(data) - window + 1)])
