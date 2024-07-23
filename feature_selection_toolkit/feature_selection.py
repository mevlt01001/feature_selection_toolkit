import math
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, r2_score, confusion_matrix
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.feature_selection import chi2, f_classif, RFE
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import statsmodels.api as sm
import warnings
import inspect

# Ignore warnings
warnings.filterwarnings("ignore")

# Numpy and pandas display settings
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


class FeatureSelection:
    """A class used to perform various feature selection techniques.

    **Attributes**
    ----------
    `_classifiers` : list
        List of classification models.
    `_regressors` : list
        List of regression models.
    `_X` : DataFrame
        The input features.
    `_y` : Series
        The target variable.
    `_X_train` : DataFrame
        Training set features.
    `_X_test` : DataFrame
        Test set features.
    `_y_train` : Series
        Training set target variable.
    `_y_test` : Series
        Test set target variable.
    `_problem_type` : str
        Type of problem ('classification' or 'regression').

    **Methods**
    -------
    `backward_elimination(significance_level=0.05)`
        Perform backward elimination for feature selection.
    `forward_selection(significance_level=0.05)`
        Perform forward selection for feature selection.
    `scored_columns(test_size=0.33, random_state=42, r_start_on=1)`
        Score columns using different models and return the best features.
    `filter_method(method='chi2')`
        Perform filter method for feature selection using chi2 or anova.
    `recursive_feature_elimination(estimator, n_features_to_select=None, force=False)`
        Perform recursive feature elimination (RFE).
    `embedded_method(method='lasso', alpha=1.0)`
        Perform embedded method for feature selection.
    """

    _classifiers = [
        ('GaussianNB', GaussianNB),
        ('BernoulliNB', BernoulliNB),
        ('LogisticRegression', LogisticRegression),
        ('RandomForestClassifier', RandomForestClassifier),
        ('GradientBoostingClassifier', GradientBoostingClassifier),
        ('KNeighborsClassifier', KNeighborsClassifier),
        ('DecisionTreeClassifier', DecisionTreeClassifier),
        ('SVC', SVC),
        ('XGBClassifier', XGBClassifier),
    ]

    _regressors = [
        ('LinearRegression', LinearRegression),
        ('RandomForestRegressor', RandomForestRegressor),
        ('GradientBoostingRegressor', GradientBoostingRegressor),
        ('KNeighborsRegressor', KNeighborsRegressor),
        ('DecisionTreeRegressor', DecisionTreeRegressor),
        ('SVR', SVR),
        ('XGBRegressor', XGBRegressor),
    ]

    def __init__(self, X, y, estimator=RandomForestRegressor(), test_size=0.33, random_state=42):
        """
        **Parameters**
        ----------
        `X` : DataFrame
            The input features.
        `y` : Series
            The target variable.
        `estimator` : model object, optional
            The model to use for feature selection (default is RandomForestRegressor).
        """
        self._estimator = estimator
        self._X = X
        self._y = y

        # Split into training and test sets
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._X, self._y,
                                                                                    test_size=test_size,
                                                                                    random_state=random_state)

        # Determine the type of problem
        self._problem_type = self.__determine_problem_type()
        self._estimators = self._regressors if self._problem_type == 'regression' else self._classifiers
        self.__perform_feature_selection()

    def backward_elimination(self, significance_level=0.05, fast=False):
        """
        Perform backward elimination for feature selection.

        Parameters
        ----------
        significance_level : float, optional
            The significance level to use for feature elimination (default is 0.05).
            This parameter helps to determine which features to remove based on their 
            statistical significance. Features with a p-value greater than the 
            significance level are considered not significant and are removed to improve 
            the model's performance and reduce overfitting. 
            Note: Lower significance levels can make the model too strict and 
            eliminate important features, while higher levels can leave too many 
            insignificant features.
            
        fast : bool, optional
            If True, use a faster method for backward elimination by only evaluating p-values 
            from the full model. If False, iteratively refit the model with remaining features 
            (default is False).

        Returns
        -------
        list
            The selected features.

        Notes
        -----
        This method performs backward elimination to remove features based on their statistical 
        significance. It can be faster if `fast` is set to True, but may miss some important 
        feature interactions.

        Examples
        --------
        >>> fs = FeatureSelection(X, y)
        >>> selected_features = fs.backward_elimination(significance_level=0.01, fast=True)
        """
        features = self._X.columns.tolist()
        total_features = len(features)

        if fast:
            with tqdm(total=total_features, desc="Fast-Backward Elimination", unit="feature") as pbar:
                while features:
                    pbar.set_description(f'Evaluating {len(features)}/{total_features} features')
                    model = sm.OLS(self._y, sm.add_constant(self._X[features])).fit()
                    p_values = model.pvalues

                    bad_feature = self.__find_bad_feature(pvalues=p_values, columns=features)
                    pbar.update(1)

                    if bad_feature is not None:
                        if p_values[bad_feature] > significance_level:
                            features.remove(bad_feature)
                            pbar.set_description(f'Removed feature: {bad_feature}')
                        else:
                            break
                    else:
                        break
        else:
            with tqdm(total=total_features, desc="Backward Elimination", unit="feature") as pbar:
                while features:
                    pbar.set_description(f'Evaluating {len(features)}/{total_features} features')
                    self._estimator.fit(self._X_train[features], self._y_train)
                    pred = self._estimator.predict(self._X_test[features])

                    model = sm.OLS(pred, sm.add_constant(self._X_test[features])).fit()
                    p_values = model.pvalues

                    bad_feature = self.__find_bad_feature(pvalues=p_values, columns=features)
                    pbar.update(1)

                    if bad_feature is not None:
                        if p_values[bad_feature] > significance_level:
                            features.remove(bad_feature)
                            pbar.set_description(f'Removed feature: {bad_feature}')
                        else:
                            break
                    else:
                        break
        pbar.set_description('Backward elimination completed.')
        pbar.close()
        return features

    def forward_selection(self, significance_level=0.05, fast=False):
        """
        Perform forward selection for feature selection.

        Parameters
        ----------
        significance_level : float, optional
            The significance level to use for feature selection (default is 0.05).
            This parameter helps to determine which features to add based on their 
            statistical significance. Features with a p-value less than or equal to the 
            significance level are considered significant and are added to the model to 
            improve its performance.
            Note: Similar to backward elimination, the choice of significance level 
            affects the model's inclusiveness of features.

        fast : bool, optional
            If True, use a faster method for forward selection by only evaluating p-values 
            from the full model. If False, iteratively refit the model with added features 
            (default is False).

        Returns
        -------
        list
            The selected features.

        Notes
        -----
        This method performs forward selection to add features based on their statistical 
        significance. It can be faster if `fast` is set to True, but may miss some important 
        feature interactions.

        Examples
        --------
        >>> fs = FeatureSelection(X, y)
        >>> selected_features = fs.forward_selection(significance_level=0.01, fast=True)
        """
        features = self._X.columns.tolist()
        selected_features = []

        if fast:
            with tqdm(total=len(features), desc="Fast-Forward Selection", unit="feature") as pbar:
                for i in range(len(features)):
                    for feature in features:
                        current_features = selected_features + [feature]

                        model = sm.OLS(self._y, sm.add_constant(self._X[current_features])).fit()
                        p_values = model.pvalues

                        last_feature = self.__last_feature_is_okey(p_values=p_values, last_feature=feature,
                                                                significance_level=significance_level)
                        if last_feature:
                            selected_features.append(feature)
                            features.remove(feature)
                            pbar.set_description(f'Selected {len(selected_features)}/{len(features) + len(selected_features)} features')
                            break
                    pbar.update(1)
        else:
            with tqdm(total=len(features), desc="Forward Selection", unit="feature") as pbar:
                for i in range(len(features)):
                    for feature in features:
                        current_features = selected_features + [feature]

                        self._estimator.fit(self._X_train[current_features], self._y_train)
                        pred = self._estimator.predict(self._X_test[current_features])

                        model = sm.OLS(pred, sm.add_constant(self._X_test[current_features])).fit()
                        p_values = model.pvalues

                        last_feature = self.__last_feature_is_okey(p_values=p_values, last_feature=feature,
                                                                significance_level=significance_level)

                        if last_feature:
                            selected_features.append(feature)
                            features.remove(feature)
                            pbar.set_description(f'Selected {len(selected_features)}/{len(features) + len(selected_features)} features')
                            break
                    pbar.update(1)
        pbar.set_description('Forward selection completed.')
        pbar.close()
        return selected_features

    def scored_columns(self, test_size=0.33, random_state=42, r_start_on=1):
        """
        Score columns using different models and return the best features.

        Parameters
        ----------
        test_size : float, optional
            The proportion of the dataset to include in the test split (default is 0.33).

        random_state : int, optional
            Random state for reproducibility (default is 42).

        r_start_on : int, optional
            The starting number of columns for combinations (default is 1).
            Note: Increasing `r_start_on` reduces computation time but may miss out 
            on potential combinations of features.

        Returns
        -------
        DataFrame
            The best feature combinations and their scores.

        Notes
        -----
        This method evaluates different combinations of features using multiple models 
        and scores them based on the specified metric (accuracy for classification, 
        R-squared for regression). It also includes the confusion matrix for classifiers.

        Examples
        --------
        >>> fs = FeatureSelection(X, y)
        >>> best_features = fs.scored_columns(test_size=0.2, random_state=1, r_start_on=2)
        """

        metric = accuracy_score if self._problem_type == 'classification' else r2_score

        scores = []
        total_combinations = self.__get_possible_combinations_count(self._X.columns, r_start_on)
        pbar = tqdm(total=total_combinations, desc='Scoring columns...', unit='combination')

        for r in range(r_start_on, len(self._X.columns) + 1):
            for columns in self.__get_combinations(self._X.columns, r):
                x_train, x_test, y_train, y_test = train_test_split(self._X[list(columns)], self._y,
                                                                    test_size=test_size, random_state=random_state)
                desc = f'Selected {r}/{len(self._X.columns)} columns'
                for model_name, model in self._estimators:
                    try:
                        model_instance = model()
                        pbar.set_description(f'{desc} - Training model...')
                        model_instance.fit(x_train, y_train)
                        pbar.set_description(f'{desc} - Predicting...')
                        y_pred = model_instance.predict(x_test)
                        pbar.set_description(f'{desc} - Scoring...')
                        score = (model_name, columns, metric.__name__, metric(y_test, y_pred))

                        # Confusion matrix for classifiers
                        if self._problem_type == 'classification':
                            pbar.set_description(f'{desc} - Confusion matrix...')
                            cm = confusion_matrix(y_test, y_pred)
                            score += (cm,)

                        scores.append(score)
                    except Exception as e:
                        continue
                    pbar.update(1)

        pbar.set_description('Scoring columns completed.')
        pbar.close()
        return self.__find_best_score(scores)
    
    def filter_method(self, method='chi2'):
        """
        Perform filter method for feature selection using chi2 or anova.

        Parameters
        ----------
        method : str, optional
            The filter method to use ('chi2' or 'anova', default is 'chi2').
            The method determines how the features are scored. 'chi2' is used 
            for classification problems and evaluates the dependence between 
            categorical features, while 'anova' is used for regression problems 
            and evaluates the variance between features and the target.

        Returns
        -------
        tuple
            Scores and p-values of the features.

        Raises
        ------
        ValueError
            If an unknown method is specified or if the method is not suitable for the 
            problem type.

        Notes
        -----
        This method performs a filter-based feature selection using either chi-squared 
        test for classification or ANOVA for regression.

        Examples
        --------
        >>> fs = FeatureSelection(X, y)
        >>> scores, p_values = fs.filter_method(method='anova')
        """
        if method not in ['chi2', 'anova']:
            raise ValueError(f"Unknown method: {method}. Please choose 'chi2' or 'anova'.")

        if method == 'chi2' and self._problem_type == 'regression':
            raise ValueError("The method 'chi2' is not suitable for regression problems with continuous target values. Please use 'anova'.")

        if method == 'anova' and self._problem_type == 'classification':
            raise ValueError("The method 'anova' is not suitable for classification problems with categorical target values. Please use 'chi2'.")

        method_func = chi2 if method == 'chi2' else f_classif

        if method == 'chi2':
            X_positive = self._X - np.min(self._X)  # X'i pozitif yapmak için kaydırma
            return chi2(X_positive, self._y)
        else:
            scores, p_values = method_func(self._X, self._y)
            return scores, p_values

    def recursive_feature_elimination(self, estimator=RandomForestRegressor(), n_features_to_select=None, force=False):
        """
        Perform recursive feature elimination (RFE).

        Parameters
        ----------
        estimator : model object
            The model to use for RFE.

        n_features_to_select : int, optional
            The number of features to select (default is half of the features).

        force : bool, optional
            Force the use of the estimator regardless of problem type (default is False).
            Note: This parameter should be used cautiously. Forcing an estimator that 
            does not match the problem type (classification vs. regression) can lead to 
            suboptimal model performance and misleading feature importance scores.

        Returns
        -------
        DataFrame
            Ranked features with their corresponding importance scores.

        Raises
        ------
        ValueError
            If the estimator does not match the problem type and `force` is not set to True.

        Notes
        -----
        This method performs recursive feature elimination to select the most important features 
        based on their ranking by the specified estimator.

        Examples
        --------
        >>> fs = FeatureSelection(X, y)
        >>> ranked_features = fs.recursive_feature_elimination(estimator=RandomForestClassifier(), n_features_to_select=10)
        """

        if not force:
            if self.__is_classifier(estimator) and self._problem_type != 'classification':
                raise ValueError(
                    "You selected a classification model, but the target variable suggests a regression problem. Please choose the correct model or use force=True.")
            elif self.__is_regressor(estimator) and self._problem_type != 'regression':
                raise ValueError(
                    "You selected a regression model, but the target variable suggests a classification problem. Please choose the correct model or use force=True.")

        if n_features_to_select is None:
            n_features_to_select = self._X.shape[1] // 2

        # Apply RFE
        print('Applying RFE...')
        rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select)
        rfe.fit(self._X, self._y)

        # Get the ranking of features
        selected_ranked = rfe.ranking_.tolist()

        # Get the names of the features
        selected_features = self._X.columns

        # Combine the rankings and feature names
        ranked_features = list(zip(selected_features, selected_ranked))

        return self.__find_best_score(ranked_features, method='rfe')

    def embedded_method(self, method='lasso', alpha=1.0):
        """
        Perform embedded method for feature selection.

        Parameters
        ----------
        method : str, optional
            The embedded method to use ('lasso', 'ridge', 'decision_tree', 'random_forest', default is 'lasso').
            Note: Lasso and Ridge are regularization techniques that can help in feature selection 
            by penalizing the coefficients of less important features. Decision Tree and Random Forest 
            models provide feature importances inherently.

        alpha : float, optional
            The regularization strength for Lasso and Ridge (default is 1.0).
            Note: Higher alpha values increase the amount of shrinkage, thus selecting fewer features.

        Returns
        -------
        numpy.ndarray
            The importance scores of the features.

        Raises
        ------
        ValueError
            If an unknown method is specified or if the method is not suitable for the 
            problem type.

        Notes
        -----
        This method performs embedded feature selection using regularization techniques 
        or tree-based models to determine feature importance.

        Examples
        --------
        >>> fs = FeatureSelection(X, y)
        >>> importance_scores = fs.embedded_method(method='ridge', alpha=0.5)
        """
        method_dict = {
            'lasso': Lasso(alpha=alpha),
            'ridge': Ridge(alpha=alpha),
            'decision_tree': DecisionTreeClassifier() if self._problem_type == 'classification' else DecisionTreeRegressor(),
            'random_forest': RandomForestClassifier() if self._problem_type == 'classification' else RandomForestRegressor()
        }

        if method not in method_dict:
            raise ValueError(
                "Unknown embedded method. Please choose 'lasso', 'ridge', 'decision_tree', or 'random_forest'.")

        if (self._problem_type == 'classification' and method in ['lasso', 'ridge']) or (self._problem_type == 'regression' and method in ['decision_tree', 'random_forest']):
            raise ValueError(
                f"The method '{method}' is not suitable for a {self._problem_type} problem. Please choose an appropriate method."
            )

        model = method_dict[method]
        model.fit(self._X, self._y)
        return model.coef_ if hasattr(model, 'coef_') else model.feature_importances_

    def rfe_brute_force(self, estimator=RandomForestRegressor(), n_features_to_select=None, force=False):
        """
        Perform brute force recursive feature elimination (RFE).

        Parameters
        ----------
        estimator : model object
            The model to use for RFE.

        n_features_to_select : int, optional
            The number of features to select (default is half of the features).

        force : bool, optional
            Force the use of the estimator regardless of problem type (default is False).
            Note: This parameter should be used cautiously. Forcing an estimator that 
            does not match the problem type (classification vs. regression) can lead to 
            suboptimal model performance and misleading feature importance scores.

        Returns
        -------
        DataFrame
            The best feature combinations and their scores.

        Raises
        ------
        ValueError
            If the estimator does not match the problem type and `force` is not set to True.

        Notes
        -----
        This method performs brute force recursive feature elimination by iteratively 
        adding features and evaluating model performance.

        Examples
        --------
        >>> fs = FeatureSelection(X, y)
        >>> best_features = fs.rfe_brute_force(estimator=RandomForestClassifier(), n_features_to_select=5, force=True)
        """
        columns = self.recursive_feature_elimination(estimator=estimator, n_features_to_select=1, force=force).iloc[:, 0].values
        

        metric = accuracy_score if self._problem_type == 'classification' else r2_score

        selected_features = []

        scores = []

        pbar = tqdm(total=len(columns) * len(self._estimators))


        for column in columns:
            selected_features.append(column)
            desc = f'Selected {len(selected_features)}/{len(columns)} columns'
            pbar.set_description(desc)
            for model_name, model in self._estimators:
                model_instance = model()
                pbar.set_description(f'{desc} - Training model...')
                model_instance.fit(self._X_train[selected_features], self._y_train)
                pbar.set_description(f'{desc} - Predicting...')
                y_pred = model_instance.predict(self._X_test[selected_features])
                pbar.set_description(f'{desc} - Scoring...')
                score = (model_name, tuple(selected_features), metric.__name__, metric(self._y_test, y_pred))

                if self._problem_type == 'classification':
                    pbar.set_description(f'{desc} - Confusion matrix...')
                    cm = confusion_matrix(self._y_test, y_pred)
                    score += (cm,)

                scores.append(score)
                pbar.update(1)

        pbar.set_description('RFE brute force completed.')
        pbar.close()
        return self.__find_best_score(scores)


    def __determine_problem_type(self):
        """Determine the problem type (classification or regression) based on the target variable.

        **Returns**
        -------
        str
            'classification' if the target variable has less than 10 unique values, otherwise 'regression'.
        """
        if len(np.unique(self._y)) < 10:
            return 'classification'
        else:
            return 'regression'

    def __is_classifier(self, model):
        """Check if the given model is a classifier.

        **Parameters**
        ----------
        `model` : model object
            The model to check.

        **Returns**
        -------
        bool
            True if the model is a classifier, False otherwise.
        """
        return any(isinstance(model, cls) for _, cls in self._classifiers)

    def __is_regressor(self, model):
        """Check if the given model is a regressor.

        **Parameters**
        ----------
        `model` : model object
            The model to check.

        **Returns**
        -------
        bool
            True if the model is a regressor, False otherwise.
        """
        return any(isinstance(model, cls) for _, cls in self._regressors)

    def __perform_feature_selection(self):
        print(f'''
        Total entry:    {self._X.shape[0]}
        Train entry:    {self._X_train.shape[0]}
        Test entry:     {self._X_test.shape[0]}
        Problem Type :  {self._problem_type}
        Estimators({len(self._estimators)}):
        {self._estimators}
        ''')

    @staticmethod
    def __binom(n, r):
        """Calculate the binomial coefficient "n choose r".

        **Parameters**
        ----------
        `n` : int
            The number of items.
        `r` : int
            The number of items to choose.

        **Returns**
        -------
        int
            The binomial coefficient.
        """
        return math.factorial(n) / (math.factorial(r) * math.factorial(n - r))

    @staticmethod
    def __get_possible_combinations_count(columns, start_from=1):
        """Get the total number of possible combinations of columns starting from a given length.

        **Parameters**
        ----------
        `columns` : list
            The list of columns.
        `start_from` : int, optional
            The starting length of the combinations (default is 1).

        **Returns**
        -------
        int
            The total number of possible combinations.
        """
        return sum(FeatureSelection.__binom(len(columns), r) for r in range(start_from, len(columns) + 1))

    @staticmethod
    def __get_combinations(columns, r):
        """Get all possible combinations of a given length from the list of columns.

        **Parameters**
        ----------
        `columns` : list
            The list of columns.
        `r` : int
            The length of the combinations.

        **Returns**
        -------
        list
            The list of combinations.
        """
        return list(combinations(columns, r))

    @staticmethod
    def __find_best_score(scores, method='default'):
        """Find the best score from the given scores.

        **Parameters**
        ----------
        `scores` : list
            The list of scores.
        `method` : str, optional
            The method name to identify the caller (default is 'default').

        **Returns**
        -------
        DataFrame
            The best score and the corresponding feature combination.
        """
        if method == 'rfe':
            df = pd.DataFrame(scores, columns=['Feature', 'Rank'])
            return df.sort_values(by='Rank', ascending=True)
        else:
            columns = ['Model', 'Features', 'Metric', 'Score']
            if any(len(score) == 5 for score in scores):  # Check if confusion matrix is included
                columns.append('Confusion Matrix')
            df = pd.DataFrame(scores, columns=columns)
            return df.sort_values(by='Score', ascending=False)

    @staticmethod
    def __find_bad_feature(pvalues, columns):
        """Find the feature with the highest p-value.

        **Parameters**
        ----------
        `pvalues` : Series
            The p-values of the features.
        `columns` : list
            The list of columns.

        **Returns**
        -------
        str or None
            The feature with the highest p-value, or None if all p-values are significant.
        """
        max_pvalues = 0
        bad_feature = None

        for column in columns:
            if pvalues[column] > max_pvalues:
                max_pvalues = pvalues[column]
                bad_feature = column

        return bad_feature

    @staticmethod
    def __last_feature_is_okey(p_values, last_feature, significance_level=0.05):
        """Check if the last added feature is significant.

        **Parameters**
        ----------
        `p_values` : Series
            The p-values of the features.
        `last_feature` : str
            The last added feature.
        `significance_level` : float, optional
            The significance level to use (default is 0.05).

        **Returns**
        -------
        bool
            True if the last feature is significant, False otherwise.
        """
        return p_values[last_feature] <= significance_level