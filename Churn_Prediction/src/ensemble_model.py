import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def time_weighted_learning(X_train, y_train, model):
    """Apply time-weighted learning to prioritize recent data."""
    weights = np.where(X_train['Date'].dt.year == X_train['Date'].dt.year.max(), 1.5, 1.0)
    model.fit(X_train, y_train, sample_weight=weights)
    return model

def online_learning(X_train, y_train, X_test, y_test):
    """Use SGD for online learning to update model with new data."""
    sgd_model = SGDClassifier(loss='log', random_state=42)
    sgd_model.partial_fit(X_train, y_train, classes=np.unique(y_train))
    sgd_model.partial_fit(X_test, y_test)
    return sgd_model

def ensemble_learning(X_train, y_train):
    """Combine models trained on different data periods using an ensemble approach."""
    model_1 = LogisticRegression(max_iter=1000, random_state=42)
    model_2 = DecisionTreeClassifier(random_state=42)
    ensemble = VotingClassifier(estimators=[('lr', model_1), ('dt', model_2)], voting='soft')
    ensemble.fit(X_train, y_train)
    return ensemble
