import unittest
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from feature_selection_toolkit.feature_selection import FeatureSelection

class TestFeatureSelectionMethods(unittest.TestCase):

    def setUp(self):
        # Classification dataset
        self.X_classification, self.y_classification = make_classification(n_samples=100, n_features=6, random_state=42)
        self.X_classification = pd.DataFrame(self.X_classification, columns=[f'Feature_{i}' for i in range(6)])
        self.y_classification = pd.Series(self.y_classification, name='Target')

        # Regression dataset
        self.X_regression, self.y_regression = make_regression(n_samples=100, n_features=6, random_state=42)
        self.X_regression = pd.DataFrame(self.X_regression, columns=[f'Feature_{i}' for i in range(6)])
        self.y_regression = pd.Series(self.y_regression, name='Target')

    def test_backward_elimination(self):
        fs = FeatureSelection(self.X_regression, self.y_regression)
        selected_features = fs.backward_elimination()
        self.assertTrue(len(selected_features) > 0, "No features were selected using backward elimination")

    def test_forward_selection(self):
        fs = FeatureSelection(self.X_regression, self.y_regression)
        selected_features = fs.forward_selection()
        self.assertTrue(len(selected_features) > 0, "No features were selected using forward selection")

    def test_scored_columns(self):
        fs = FeatureSelection(self.X_classification, self.y_classification)
        best_features = fs.scored_columns(r_start_on=2)
        self.assertTrue(len(best_features) > 0, "No best features were found using scored_columns method")

    def test_filter_method(self):
        fs = FeatureSelection(self.X_classification, self.y_classification)
        scores, p_values = fs.filter_method(method='chi2')
        self.assertEqual(len(scores), self.X_classification.shape[1])
        self.assertEqual(len(p_values), self.X_classification.shape[1])

    def test_recursive_feature_elimination(self):
        fs = FeatureSelection(self.X_classification, self.y_classification)
        ranked_features = fs.recursive_feature_elimination(estimator=RandomForestClassifier(), n_features_to_select=5)
        self.assertTrue(len(ranked_features) > 0, "No features were ranked using recursive feature elimination")

    def test_embedded_method(self):
        fs = FeatureSelection(self.X_regression, self.y_regression)
        importance_scores = fs.embedded_method(method='lasso', alpha=0.1)
        self.assertTrue(len(importance_scores) > 0, "No importance scores were returned using embedded method")

    def test_rfe_brute_force(self):
        fs = FeatureSelection(self.X_classification, self.y_classification)
        best_features = fs.rfe_brute_force(estimator=RandomForestClassifier(), n_features_to_select=5, force=True)
        self.assertTrue(len(best_features) > 0, "No best features were found using RFE brute force method")

if __name__ == '__main__':
    unittest.main()
