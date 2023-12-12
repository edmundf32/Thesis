import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

def train_and_predict(data, new_data, language):
    # Define numerical columns
    numerical_cols = [
        'score',
        'changed_files',
        'deletions',
        'additions',
        'review_comments'
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


    # Define features and target
    X = data[numerical_cols + categorical_cols]
    y = data[target_col]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing for numerical data (excluding 'score')
    numerical_features = [col for col in numerical_cols if col != 'score']
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
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Preprocess the training data
    X_train_preprocessed = preprocessor.fit_transform(X_train[numerical_features + categorical_cols])
    X_test_preprocessed = preprocessor.transform(X_test[numerical_features + categorical_cols])

    # Create and train a Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train_preprocessed, y_train)

    # Ridge Regression Model
    ridge_model = Ridge()
    ridge_model.fit(X_train_preprocessed, y_train)

    # Random Forest Regressor Model
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train_preprocessed, y_train)

    # Gradient Boosting Regressor Model
    gb_model = GradientBoostingRegressor(random_state=42)
    gb_model.fit(X_train_preprocessed, y_train)
    
    models = {
        'Linear Regression': lr_model,
        'Ridge Regression': ridge_model,
        'Random Forest': rf_model,
        'Gradient Boosting': gb_model
    }

    for model_name, model in models.items():
        y_pred = model.predict(X_test_preprocessed)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{model_name} - MSE: {mse}, R2 Score: {r2}")
        
        # Plotting actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.title(f"{language} - {model_name}: Actual vs Predicted")
        plt.xlabel('Actual Scores')
        plt.ylabel('Predicted Scores')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.show()

    # Define features (excluding 'score') for prediction
    prediction_features = numerical_features + categorical_cols

    # Aggregate new data for prediction
    new_data_aggregated = new_data.groupby('repo_id').agg({
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

    # Select features for prediction
    new_X = new_data_aggregated[prediction_features]

    # Preprocess the new data for prediction
    new_X_preprocessed = preprocessor.transform(new_X)

    # Make predictions for the new data using all models
    new_data_aggregated['lr_predicted_score'] = lr_model.predict(new_X_preprocessed)
    new_data_aggregated['ridge_predicted_score'] = ridge_model.predict(new_X_preprocessed)
    new_data_aggregated['rf_predicted_score'] = rf_model.predict(new_X_preprocessed)
    new_data_aggregated['gb_predicted_score'] = gb_model.predict(new_X_preprocessed)

    # Display the DataFrame with predictions
    print(new_data_aggregated.head())

datasets = {
    #'javascript': ('data\javascript_train_data.csv', 'data\javascript_new_data.csv', 1)
    #'java': ('data\java_train_data.csv', 'data\java_new_data.csv', 6)
    'python': ('data\python_train_data.csv', 'data\python_new_data.csv', 6)
}

# Run the training and prediction for each dataset
for language, (train_file, new_data_file, target_score) in datasets.items():
    print(f"Processing {language} dataset and target score {target_score}")
    data = pd.read_csv(train_file)
    new_data = pd.read_csv(new_data_file)
    train_and_predict(data, new_data, language)
    print("--#" * 30)
    print("\n") 
    print("\n") 
