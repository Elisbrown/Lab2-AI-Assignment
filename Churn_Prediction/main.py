from src import preprocess, train_model, drift_detection, ensemble_model, utils

# Load and preprocess data
url = 'https://docs.google.com/spreadsheets/d/1bN2C5iD8uNG4BQrtXE5WNpGBhEdEjpcWBEJ8alT1oks/export?format=csv'
data = preprocess.load_data(url)
data = preprocess.handle_missing_values(data)

# Feature and target separation with class imbalance handling
X = data.drop(columns=['Churn'])
y = data['Churn']
X_resampled, y_resampled = preprocess.handle_class_imbalance(X, y)

# Split data into training and testing sets
X_train, y_train, X_test, y_test = preprocess.split_data(data)

# Train initial model and evaluate
model = train_model.train_initial_model(X_train, y_train)
train_model.evaluate_model(model, X_test, y_test)

# Detect drift
drift_detection.monitor_drift(model, X_test, y_test)

# Ensemble learning and evaluate
ensemble = ensemble_model.ensemble_learning(X_train, y_train)
train_model.evaluate_model(ensemble, X_test, y_test)
