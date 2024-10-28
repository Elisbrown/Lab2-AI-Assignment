from river.drift import ADWIN

def monitor_drift(model, X_test, y_test):
    """Detect and handle concept drift using ADWIN."""
    drift_detector = ADWIN()
    for prediction in model.predict(X_test):
        drift_detector.update(prediction)
        if drift_detector.change_detected:
            print("Drift detected; retraining the model.")
            model.fit(X_test, y_test)  # Retrain on detected drift
