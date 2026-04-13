import numpy as np
import pytest
from src.models.isolation_forest import IFAnomalyDetector


@pytest.fixture
def detector():
    return IFAnomalyDetector(n_estimators=50, contamination=0.35, random_state=42)


@pytest.fixture
def synthetic_data():
    """Données normales + anomalies synthétiques."""
    np.random.seed(42)
    X_normal   = np.random.randn(500, 8) * 0.5
    X_anomaly  = np.random.randn(100, 8) * 3.0 + 5.0
    X_all      = np.vstack([X_normal, X_anomaly])
    y          = np.array([0]*500 + [1]*100)
    return X_normal, X_all, y


def test_fit_returns_self(detector, synthetic_data):
    X_normal, _, _ = synthetic_data
    result = detector.fit(X_normal)
    assert result is detector


def test_score_shape(detector, synthetic_data):
    X_normal, X_all, _ = synthetic_data
    detector.fit(X_normal)
    scores = detector.score(X_all)
    assert scores.shape == (600,)


def test_scores_anomalies_higher(detector, synthetic_data):
    """Les anomalies doivent avoir un score moyen plus élevé."""
    X_normal, X_all, y = synthetic_data
    detector.fit(X_normal)
    scores = detector.score(X_all)
    assert scores[y==1].mean() > scores[y==0].mean()


def test_predict_before_threshold_raises(detector, synthetic_data):
    X_normal, X_all, _ = synthetic_data
    detector.fit(X_normal)
    with pytest.raises(ValueError, match="optimize_threshold"):
        detector.predict(X_all)


def test_predict_binary_output(detector, synthetic_data):
    X_normal, X_all, y = synthetic_data
    detector.fit(X_normal)
    detector.optimize_threshold(X_all, y)
    preds = detector.predict(X_all)
    assert set(np.unique(preds)).issubset({0, 1})
