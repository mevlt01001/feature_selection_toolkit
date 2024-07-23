# Changelog

## [1.0.7] - 2024-07-23

---

## Added

- **New Method: `rfe_brute_force`**
  - Implemented the `rfe_brute_force` method for exhaustive feature elimination.
  - This method iteratively adds features based on Recursive Feature Elimination (RFE) and evaluates model performance to find the best feature subset.
  - Example usage:
    ```python
    best_features = fs.rfe_brute_force(estimator=RandomForestClassifier(), n_features_to_select=5, force=True)
    print("Best Features:", best_features)
    ```

## Changed

- **Backward Elimination and Forward Selection Enhancements**
  - Added a `fast` parameter to both `backward_elimination` and `forward_selection` methods.
  - When `fast=True`, these methods use a quicker evaluation process, reducing computational time.
  - Example usage:
    ```python
    selected_features = fs.backward_elimination(significance_level=0.05, fast=True)
    print("Selected Features (Fast Backward Elimination):", selected_features)
    ```
  - Example usage for forward selection:
    ```python
    selected_features = fs.forward_selection(significance_level=0.05, fast=True)
    print("Selected Features (Fast Forward Selection):", selected_features)
    ```

## Fixed

- **Filter Method Improvements**
  - Improved the `filter_method` to raise appropriate errors when using `chi2` for regression problems or `anova` for classification problems.
  - Ensured better handling of data types and target variable compatibility.