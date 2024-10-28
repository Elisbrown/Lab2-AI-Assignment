import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

def load_data(url):
    """Load dataset from URL."""
    data = pd.read_csv(url)
    return data

def handle_missing_values(data):
    """Impute missing values for numerical and categorical columns."""
    num_cols = data.select_dtypes(include=np.number).columns
    cat_cols = data.select_dtypes(exclude=np.number).columns
    
    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    
    data[num_cols] = num_imputer.fit_transform(data[num_cols])
    data[cat_cols] = cat_imputer.fit_transform(data[cat_cols])
    return data

def handle_class_imbalance(X, y):
    """Apply SMOTE to balance classes."""
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def split_data(data):
    """Split data by time periods into training and testing sets."""
    data['Date'] = pd.to_datetime(data['Date'])
    train_data = data[data['Date'].dt.year <= data['Date'].dt.year.max() - 1]
    test_data = data[data['Date'].dt.year == data['Date'].dt.year.max()]
    
    X_train, y_train = train_data.drop(columns=['Churn', 'Date']), train_data['Churn']
    X_test, y_test = test_data.drop(columns=['Churn', 'Date']), test_data['Churn']
    return X_train, y_train, X_test, y_test
