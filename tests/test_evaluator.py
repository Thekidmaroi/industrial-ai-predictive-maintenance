import numpy as np
import pandas as pd
import pytest
from src.models.evaluator import evaluate, compare_models


@pytest.fixture
def perfect_predictions():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.1, 0.2, 0.8, 0.9, 0.95])
    return y_true, y_pred, scores


@pytest.fixture
def imperfect_predictions():
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    scores = np.array([0.1, 0.6, 0.8, 0.9, 0.3, 0.2])
    return y_true, y_pred, scores


def test_evaluate_returns_dict(perfect_predictions):
    y_true, y_pred, scores = perfect_predictions
    result = evaluate(y_true, y_pred, scores, "Test Model")
    assert isinstance(result, dict)
    assert 'roc_auc' in result
    assert 'recall' in result
    assert 'f1' in result


def test_evaluate_perfect_scores(perfect_predictions):
    y_true, y_pred, scores = perfect_predictions
    result = evaluate(y_true, y_pred, scores, "Perfect")
    assert result['roc_auc'] == 1.0
    assert result['recall']  == 1.0
    assert result['f1']      == 1.0


def test_compare_models_sorted(imperfect_predictions):
    y_true, y_pred, scores = imperfect_predictions
    r1 = evaluate(y_true, y_pred, scores, "Model A")
    r2 = evaluate(y_true, y_pred[::-1], scores[::-1], "Model B")
    df = compare_models([r1, r2])
    assert isinstance(df, pd.DataFrame)
    assert df.iloc[0]['roc_auc'] >= df.iloc[1]['roc_auc']
