import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, mean_squared_error

path = '../datasets/data_4000v/env_vital_signals.txt'

try:
    df = pd.read_csv(path, header=None)
    df.columns = [
        'vic_id', 'sistolic_presion', 'diastolic_pression', 'qPA', 
        'pulse', 'resp_freq', 'gravity_value', 'gravity_class'
    ]
except FileNotFoundError:
    print("Error: Dataset file not found. Path: " + path)
    exit()

X = df[['qPA', 'pulse', 'resp_freq']]
y_reg = df['gravity_value']
y_class = df['gravity_class'] - 1

print("Data loaded successfully.")
print("-" * 50)

print("--- Finding best Decision Tree REGRESSOR with GridSearchCV ---")
param_grid_reg = {
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_leaf': [2, 5, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search_reg = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid_reg,
    cv=5,
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=-1
)
grid_search_reg.fit(X, y_reg)
print(f"Best Regressor Params: {grid_search_reg.best_params_}")
best_rmse = np.sqrt(-grid_search_reg.best_score_)
print(f"Best Cross-Validated RMSE: {best_rmse:.4f}")


print("\n" + "-" * 50)

print("--- Finding best Decision Tree CLASSIFIER with GridSearchCV ---")
param_grid_class = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_leaf': [1, 2, 5, 10],
    'min_samples_split': [2, 5, 10]
}
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='macro', zero_division=0),
    'recall': make_scorer(recall_score, average='macro', zero_division=0)
}
grid_search_class = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid_class,
    cv=5,
    scoring=scoring,
    refit='accuracy',
    verbose=1,
    n_jobs=-1
)
grid_search_class.fit(X, y_class)
print(f"Best Classifier Params: {grid_search_class.best_params_}")

print("Best Classifier Cross-Validated Scores:")
results = grid_search_class.cv_results_
best_index = grid_search_class.best_index_
print(f"  Accuracy:  {results['mean_test_accuracy'][best_index]:.4f} (+/- {results['std_test_accuracy'][best_index]:.4f})")
print(f"  Precision: {results['mean_test_precision'][best_index]:.4f} (+/- {results['std_test_precision'][best_index]:.4f})")
print(f"  Recall:    {results['mean_test_recall'][best_index]:.4f} (+/- {results['std_test_recall'][best_index]:.4f})")


print("\n" + "=" * 50)
print("Grid search complete.")
print("The best parameters have been found for both models.")
print("The section below will train final models with these best parameters and save them.")
print("=" * 50)


best_regressor_config = grid_search_reg.best_params_
best_classifier_config = grid_search_class.best_params_

# Train final regressor on all data
final_regressor = DecisionTreeRegressor(random_state=42, **best_regressor_config)
final_regressor.fit(X, y_reg)
joblib.dump(final_regressor, 'best_regressor_dt.joblib')
print("\nSaved best DT regressor to 'best_regressor_dt.joblib'")

# Train final classifier on all data
final_classifier = DecisionTreeClassifier(random_state=42, **best_classifier_config)
final_classifier.fit(X, y_class)
joblib.dump(final_classifier, 'best_classifier_dt.joblib')
print("Saved best DT classifier to 'best_classifier_dt.joblib'")
