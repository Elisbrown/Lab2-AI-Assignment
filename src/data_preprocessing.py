import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load the Titanic dataset
df = pd.read_csv('')

# Inspect the first few rows of the DataFrame
print(df.head())

# Print the column names to check for unnecessary ones
print(df.columns)

# 2. Data Cleaning
print("\n2. Data Cleaning")

# Check for missing values
print("Missing Values:")
print(df.isnull().sum())

# Impute missing values in 'Age' with the mean
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Impute missing values in 'Embarked' with the mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# 3. Handling Outliers
print("\n3. Handling Outliers")

# Calculate upper and lower bounds for 'Fare' (e.g., 3 standard deviations)
upper_bound = df['Fare'].mean() + 3 * df['Fare'].std()
lower_bound = df['Fare'].mean() - 3 * df['Fare'].std()

# Cap values outside the threshold
df['Fare'] = np.clip(df['Fare'], lower_bound, upper_bound)

# 4. Data Normalization
print("\n4. Data Normalization")

# Min-Max Scaling
scaler_minmax = MinMaxScaler()
df['Age_scaled'] = scaler_minmax.fit_transform(df[['Age']])
df['Fare_scaled'] = scaler_minmax.fit_transform(df[['Fare']])

# Z-score Normalization
scaler_zscore = StandardScaler()
df['Age_zscore'] = scaler_zscore.fit_transform(df[['Age']])
df['Fare_zscore'] = scaler_zscore.fit_transform(df[['Fare']])

# 5. Feature Engineering
print("\n5. Feature Engineering")

# Create 'FamilySize' column
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Extract 'Title' from 'Name'
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

# 6. Feature Selection (using a model)
print("\n6. Feature Selection")

X = df[['Age', 'Fare', 'FamilySize']] 
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

feature_importances = model.feature_importances_
feature_names = X.columns

for importance, name in zip(feature_importances, feature_names):
    print(f'{name}: {importance}')

# 7. Model Building
print("\n7. Model Building")

# Split data
X = df[['Age', 'Fare', 'FamilySize']] 
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# (You can now use the trained model for predictions)

# Display the cleaned and preprocessed DataFrame (optional)
print("\nCleaned and Preprocessed DataFrame:")
print(df.head()) 

# Save the preprocessed data to a new CSV file (optional)
df.to_csv('titanic_preprocessed.csv', index=False)