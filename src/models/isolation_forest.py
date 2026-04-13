import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve


class IFAnomalyDetector:
    """Isolation Forest avec optimisation automatique du seuil."""

    def __init__(self, n_estimators=200, contamination=0.35, random_state=42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler    = StandardScaler()
        self.threshold = None

    def fit(self, X_normal: np.ndarray):
        """Entraîne sur les données normales uniquement."""
        X_scaled = self.scaler.fit_transform(X_normal)
        self.model.fit(X_scaled)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Retourne les scores d'anomalie (plus haut = plus anormal)."""
        X_scaled = self.scaler.transform(X)
        return -self.model.score_samples(X_scaled)

    def optimize_threshold(self, X: np.ndarray, y_true: np.ndarray):
        """Trouve le seuil qui maximise le F1-score."""
        scores = self.score(X)
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        self.threshold = thresholds[f1.argmax()]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédit les anomalies avec le seuil optimisé."""
        if self.threshold is None:
            raise ValueError("Appelle optimize_threshold() avant predict()")
        return (self.score(X) >= self.threshold).astype(int)
