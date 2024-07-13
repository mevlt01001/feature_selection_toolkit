import pytest
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from feature_selection_toolkit.feature_selection import FeatureSelection
import numpy as np

@pytest.fixture
def data():
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y

def test_initialization(data):
    X, y = data
    fs = FeatureSelection(X, y)
    assert fs._X.equals(X)
    assert fs._y.equals(y)

def test_forward_selection(data):
    X, y = data
    fs = FeatureSelection(X, y)
    selected_features = fs.forward_selection(significance_level=0.05)
    assert isinstance(selected_features, list)
    assert len(selected_features) > 0

def test_backward_elimination(data):
    X, y = data
    fs = FeatureSelection(X, y)
    selected_features = fs.backward_elimination(significance_level=0.05)
    assert isinstance(selected_features, list)
    assert len(selected_features) > 0

def test_filter_method_chi2(data):
    X, y = data
    fs = FeatureSelection(X, y)
    scores, p_values = fs.filter_method(method='chi2')
    assert len(scores) == X.shape[1]
    assert len(p_values) == X.shape[1]

def test_filter_method_anova(data):
    X, y = data
    fs = FeatureSelection(X, y)
    scores, p_values = fs.filter_method(method='anova')
    assert len(scores) == X.shape[1]
    assert len(p_values) == X.shape[1]

def test_recursive_feature_elimination(data):
    X, y = data
    fs = FeatureSelection(X, y)
    support = fs.recursive_feature_elimination(estimator=RandomForestClassifier(), n_features_to_select=2)
    assert isinstance(support, np.ndarray)
    assert len(support) == X.shape[1]

def test_embedded_method_lasso(data):
    X, y = data
    fs = FeatureSelection(X, y)
    coefficients = fs.embedded_method(method='lasso', alpha=0.01)
    assert len(coefficients) == X.shape[1]

def test_scored_columns(data):
    X, y = data
    fs = FeatureSelection(X, y)
    best_scores = fs.scored_columns(test_size=0.33, random_state=42, r_start_on=1)
    assert isinstance(best_scores, list)
    assert len(best_scores) > 0
