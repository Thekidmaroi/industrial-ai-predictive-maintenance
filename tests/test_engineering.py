import numpy as np
import pandas as pd
import pytest
from src.features.engineering import add_rolling_features, make_sequences


@pytest.fixture
def sample_df():
    """DataFrame minimal simulant des données SCADA."""
    np.random.seed(42)
    n = 200
    sensor_cols = [
        'Accelerometer1RMS', 'Accelerometer2RMS', 'Current',
        'Pressure', 'Temperature', 'Thermocouple',
        'Voltage', 'Volume Flow RateRMS'
    ]
    data = {col: np.random.rand(n) for col in sensor_cols}
    data['anomaly'] = (np.random.rand(n) > 0.7).astype(int)
    return pd.DataFrame(data)


def test_add_rolling_features_shape(sample_df):
    """Les features rolling doivent ajouter des colonnes."""
    sensor_cols = [c for c in sample_df.columns if c != 'anomaly']
    discriminant = ['Accelerometer1RMS', 'Accelerometer2RMS', 'Volume Flow RateRMS']
    result = add_rolling_features(sample_df, sensor_cols, discriminant, window=10)

    # 8 capteurs + 3x2 rolling + 1 ratio = 15 colonnes
    assert result.shape[1] == 15, f"Attendu 15 colonnes, obtenu {result.shape[1]}"


def test_add_rolling_features_no_nan(sample_df):
    """Pas de NaN après dropna."""
    sensor_cols = [c for c in sample_df.columns if c != 'anomaly']
    discriminant = ['Accelerometer1RMS', 'Accelerometer2RMS', 'Volume Flow RateRMS']
    result = add_rolling_features(sample_df, sensor_cols, discriminant, window=10)
    assert result.isnull().sum().sum() == 0


def test_make_sequences_shape():
    """Les séquences doivent avoir la bonne shape 3D."""
    data = np.random.rand(100, 8)
    window = 30
    sequences = make_sequences(data, window)
    assert sequences.shape == (71, 30, 8), f"Shape inattendue : {sequences.shape}"


def test_make_sequences_content():
    """La première séquence doit correspondre aux premières lignes."""
    data = np.arange(50).reshape(25, 2).astype(float)
    sequences = make_sequences(data, window=5)
    np.testing.assert_array_equal(sequences[0], data[:5])
    np.testing.assert_array_equal(sequences[1], data[1:6])
