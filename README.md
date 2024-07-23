# Feature Selection Toolkit

## Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Installation](#installation)
4. [Usage](#usage)
    - [Initialization](#initialization)
    - [Filter Methods](#filter-methods)
        - [Chi-Squared Test](#chi-squared-test)
        - [ANOVA Test](#anova-test)
    - [Wrapper Methods](#wrapper-methods)
        - [Forward Selection](#forward-selection)
        - [Backward Elimination](#backward-elimination)
    - [Recursive Feature Elimination (RFE)](#recursive-feature-elimination-rfe)
    - [Embedded Methods](#embedded-methods)
        - [Lasso](#lasso)
        - [Ridge](#ridge)
        - [Decision Tree](#decision-tree)
        - [Random Forest](#random-forest)
    - [Brute Force Search](#brute-force-search)
        - [Scored Columns](#scored-columns)
        - [RFE Brute Force](#rfe-brute-force)
5. [Statistical Evidence](#statistical-evidence)
6. [Real-World Examples](#real-world-examples)
    - [Example 1: Iris Dataset](#example-1-iris-dataset)
    - [Example 2: Housing Prices](#example-2-housing-prices)
7. [Conclusion](#conclusion)
8. [Contributing](#contributing)
9. [License](#license)

## Overview
The Feature Selection Toolkit is designed to simplify the process of selecting the most significant features from a dataset. By utilizing various feature selection methods, this toolkit aims to enhance model performance and reduce computational complexity. This comprehensive toolkit supports both classification and regression tasks, providing a range of methods to fit different scenarios.

## Key Features
- Filter Methods: Utilize statistical tests like Chi-Squared and ANOVA to select features.
- Wrapper Methods: Implement Forward Selection and Backward Elimination to iteratively select or remove features.
- Embedded Methods: Integrate feature selection within the model training process using Lasso, Ridge, Decision Trees, and Random Forests.
- Recursive Feature Elimination (RFE): Use an iterative process with estimators to remove less important features.
- Brute Force Search: Evaluate all possible feature combinations to find the optimal subset for model performance.
- RFE Brute Force: Combines RFE and brute force search to ensure the best feature set is selected.

## Installation
To install the Feature Selection Toolkit, you can use pip:

```
pip install feature-selection-toolkit
```

## Usage

### Initialization
First, initialize the `FeatureSelection` class with your dataset:

```
from sklearn.datasets import load_iris
import pandas as pd
from feature_selection_toolkit import FeatureSelection

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

fs = FeatureSelection(X, y)
```

### Filter Methods
Filter methods assess each feature independently to determine its relevance to the target variable.

#### Chi-Squared Test
The Chi-Squared test evaluates the independence between categorical features and the target variable. It's particularly useful for classification tasks where both the features and target are categorical.

```
scores, p_values = fs.filter_method(method='chi2')
print("Chi-Squared Scores:", scores)
print("p-values:", p_values)
```

Benefits:
- Simple and fast computation.
- Effective for identifying significant categorical features.
- Useful in preliminary analysis to reduce feature space.

Use Case:
Ideal for datasets with categorical variables where the goal is to select features that are significantly associated with the target variable.

#### ANOVA Test
The ANOVA (Analysis of Variance) test assesses the difference between group means for continuous features relative to the target variable. It's suitable for classification tasks with continuous features.

```
scores, p_values = fs.filter_method(method='anova')
print("ANOVA Scores:", scores)
print("p-values:", p_values)
```

Benefits:
- Identifies features that contribute significantly to the variance between groups.
- Simple implementation and interpretation.
- Helps in reducing the dimensionality of continuous features.

Use Case:
Ideal for datasets with continuous features where the goal is to determine features that significantly differentiate between target classes.

### Wrapper Methods
Wrapper methods evaluate feature subsets using a specific model to iteratively select or remove features based on model performance.

#### Forward Selection
Forward Selection starts with an empty set of features and adds one feature at a time based on the model performance until the addition of new features does not improve the model.

```
selected_features = fs.forward_selection(significance_level=0.05)
print("Selected Features using Forward Selection:", selected_features)
```

Benefits:
- Builds the model incrementally, ensuring each added feature improves performance.
- Suitable for smaller feature sets.
- Provides interpretable results by showing the order of feature importance.

Use Case:
Useful when you have a relatively small number of features and want to build a model by iteratively adding the most significant features.

#### Backward Elimination
Backward Elimination starts with all features and iteratively removes the least significant feature based on model performance until only significant features remain.

```
selected_features = fs.backward_elimination(significance_level=0.05)
print("Selected Features using Backward Elimination:", selected_features)
```

Benefits:
- Considers the interaction between features by starting with all features.
- Effective in eliminating irrelevant or redundant features.
- Suitable for models with a larger number of features.

Use Case:
Ideal for datasets with a large number of features, where the goal is to iteratively remove the least significant ones to improve model performance.

#### Recursive Feature Elimination (RFE)
RFE removes the least important features iteratively based on a specified estimator until the desired number of features is reached.

```
support = fs.recursive_feature_elimination(estimator=RandomForestClassifier(), n_features_to_select=2)
print("RFE Support:", support)
```

Benefits:
- Provides a ranking of features based on their importance.
- Integrates with any estimator, offering flexibility.
- Iteratively removes less important features to improve model performance.

Use Case:
Suitable for situations where you want to rank features based on their importance and iteratively refine the feature set.

### Embedded Methods
Embedded methods perform feature selection during the model training process.

#### Lasso
Lasso (Least Absolute Shrinkage and Selection Operator) adds a penalty equal to the absolute value of the magnitude of coefficients, shrinking some coefficients to zero.

```
coefficients = fs.embedded_method(method='lasso', alpha=0.01)
print("Lasso Coefficients:", coefficients)
```

Benefits:
- Performs both feature selection and regularization.
- Reduces model complexity by eliminating less important features.
- Particularly useful for high-dimensional data.

Use Case:
Ideal for regression tasks with a large number of features where regularization and feature selection are required simultaneously.

#### Ridge
Ridge Regression adds a penalty equal to the square of the magnitude of coefficients, shrinking coefficients but keeping all features.

```
coefficients = fs.embedded_method(method='ridge', alpha=0.01)
print("Ridge Coefficients:", coefficients)
```

Benefits:
- Reduces overfitting by regularizing the model.
- Keeps all features but reduces their impact based on importance.
- Useful for multicollinear data.

Use Case:
Suitable for regression tasks where overfitting is a concern, and you want to regularize without eliminating features.

#### Decision Tree
Decision Trees provide feature importances inherently, which can be used for feature selection.

```
importances = fs.embedded_method(method='decision_tree')
print("Decision Tree Importances:", importances)
```

Benefits:
- Provides a clear importance ranking of features.
- Non-parametric method, useful for both classification and regression tasks.
- Handles interactions between features naturally.

Use Case:
Ideal for datasets where you want a quick and interpretable way to assess feature importance.

#### Random Forest
Random Forests aggregate the importance scores from multiple decision trees, providing a more robust feature importance measure.

```
importances = fs.embedded_method(method='random_forest')
print("Random Forest Importances:", importances)
```

Benefits:
- More stable and robust than a single decision tree.
- Captures interactions and non-linear relationships between features.
- Suitable for both classification and regression tasks.

Use Case:
Useful when you want a robust and stable measure of feature importance from an ensemble of trees.

### Brute Force Search
#### Scored Columns
Evaluate all possible feature combinations to find the best performing subset. This method ensures the selection of the optimal feature set by trying every possible combination.

```
best_scores = fs.scored_columns(test_size=0.2, random_state=1, r_start_on=2)
print("Best Scores:", best_scores)
```

Benefits:
- Guarantees finding the optimal feature subset.
- Provides a comprehensive evaluation of feature combinations.
- Useful for understanding the interactions between different feature sets.

Use Case:
Ideal for small to medium-sized datasets where computational resources allow for evaluating all feature combinations to find the best subset.

#### RFE Brute Force
Combines the strengths of Recursive Feature Elimination and Brute Force Search to ensure the best feature set is selected by evaluating all possible subsets generated by RFE.

```
best_features = fs.rfe_brute_force(estimator=RandomForestClassifier(), n_features_to_select=5, force=True)
print("Best Features from RFE Brute Force:", best_features)
```

Benefits:
- Ensures the selection of the optimal feature subset by combining RFE and brute force search.
- Provides a robust evaluation of feature importance and interactions.
- Suitable for datasets where feature importance and interactions are critical for model performance.

Use Case:
Ideal for complex datasets where both feature importance and interactions need to be evaluated comprehensively to select the best feature subset.

## Statistical Evidence
The methods included in this toolkit are based on well-established statistical techniques and have been extensively validated in academic research. For instance:
- Chi-Squared Test: Commonly used in hypothesis testing and feature selection in machine learning.
- ANOVA: Widely applied in statistical analysis to compare means and variance among groups.
- Lasso and Ridge: Regularization techniques that are crucial in high-dimensional data analysis.
- Recursive Feature Elimination: Proven method in model performance enhancement by iteratively removing less significant features.

## Real-World Examples

### Example 1: Iris Dataset
Using the Iris dataset, the toolkit can help in selecting the most important features for classifying different species of flowers. Forward Selection, for instance, can iteratively add features to find the optimal subset that maximizes classification accuracy.

### Example 2: Housing Prices
In a regression task like predicting housing prices, embedded methods like Lasso and Ridge can be used to handle high-dimensional data and identify key features that influence prices, leading to more accurate and interpretable models.

## Conclusion
The Feature Selection Toolkit is an essential tool for data scientists and machine learning practitioners looking to improve their models' performance by selecting the most relevant features. With its comprehensive range of methods and user-friendly interface, it provides a robust solution for feature selection in various machine learning tasks.

## Contributing
We welcome contributions to the Feature Selection Toolkit. If you have ideas for new features, bug fixes, or improvements, please follow these steps:
- Fork the repository.
- Create a new branch for your feature or bug fix.
- Make your changes, ensuring they are clear and concise.
- Commit your changes with descriptive commit messages.
- Push your branch to your forked repository on GitHub.
- Submit a pull request with a detailed description of your changes.
- For major changes, please open an issue first to discuss what you would like to change.

## License

The Feature Selection Toolkit is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.