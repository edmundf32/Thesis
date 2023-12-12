import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

#load the training and the test datasets
#javascript
#data = pd.read_csv('data\javascript_train_data.csv')
#new_data = pd.read_csv('data\javascript_new_data.csv')

#java
#data = pd.read_csv('data\java_train_data.csv')
#new_data = pd.read_csv('data\java_new_data.csv')

#python
data = pd.read_csv('data\python_train_data.csv')
new_data = pd.read_csv('data\python_new_data.csv')

# Define numerical columns
numerical_cols = [
    'score',
    'changed_files',
    'deletions',
    'additions',
    'review_comments',
    'repo_id'
]

# Define numerical and categorical columns
categorical_cols = [
    'action',
    'user_type',
    'merged',
    'state',
    'author_association'
]

# Define target column
target_col = 'score'

# Aggregate the data to have one row per feature
data_aggregated = data.groupby('repo_id').agg({
    'score': 'mean',
    'changed_files': 'mean',
    'deletions': 'mean',
    'additions': 'mean',
    'review_comments': 'mean',
    'action': 'first',
    'user_type': 'first',
    'merged': 'first',
    'state': 'first',
    'author_association': 'first'
}).reset_index()

# Define features and target
X = data_aggregated[numerical_cols + categorical_cols]
y = data_aggregated[target_col]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Preprocess the training data
X_train = preprocessor.fit_transform(X_train)

# Preprocess the testing data
X_test = preprocessor.transform(X_test)

# Create and train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2) Score: {r2}')

# Load and preprocess new data for prediction

new_data_aggregated = new_data.groupby('repo_id').agg({
    'score': 'mean',
    'changed_files': 'mean',
    'deletions': 'mean',
    'additions': 'mean',
    'review_comments': 'mean',
    'action': 'first',
    'user_type': 'first',
    'merged': 'first',
    'state': 'first',
    'author_association': 'first'
}).reset_index()

new_X = new_data_aggregated[numerical_cols + categorical_cols]
new_X = preprocessor.transform(new_X)

# Make predictions for the new data
new_predictions = model.predict(new_X)

new_data_aggregated['predicted_score'] = new_predictions

# Display the DataFrame with predictions
print(new_data_aggregated.head())