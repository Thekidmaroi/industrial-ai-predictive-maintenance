import pandas as pd
import numpy as np
from sklearn.metrics import (classification_report, roc_auc_score,
                             recall_score, f1_score, precision_score)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             scores: np.ndarray, model_name: str) -> dict:
    """Calcule et affiche toutes les métriques."""
    metrics = {
        'model'    : model_name,
        'roc_auc'  : round(roc_auc_score(y_true, scores), 4),
        'recall'   : round(recall_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred), 4),
        'f1'       : round(f1_score(y_true, y_pred), 4),
    }

    print(f"\n=== {model_name} ===")
    print(classification_report(y_true, y_pred,
                                 target_names=["Normal", "Anomalie"]))
    for k, v in metrics.items():
        if k != 'model':
            print(f"{k:12} : {v}")

    return metrics


def compare_models(results: list[dict]) -> pd.DataFrame:
    """Tableau comparatif de plusieurs modèles."""
    df = pd.DataFrame(results)
    df = df.sort_values('roc_auc', ascending=False)
    return df
