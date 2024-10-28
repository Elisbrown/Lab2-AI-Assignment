from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

def train_initial_model(X_train, y_train):
    """Train a logistic regression model."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print performance metrics."""
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("AUC-ROC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
