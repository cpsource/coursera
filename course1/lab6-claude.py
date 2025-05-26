"""
Regression Tree Analysis for Taxi Trip Data
This script analyzes taxi trip data to predict tip amounts using Decision Tree Regression.
"""
from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data(url):
    """
    Load data from URL and perform initial exploratory analysis.
    
    Args:
        url (str): URL to the CSV data file
        
    Returns:
        pd.DataFrame: The loaded dataframe
    """
    print("Loading data...")
    raw_data = pd.read_csv(url)
    
    # Display basic information about the dataset
    print(f"Dataset shape: {raw_data.shape}")
    print("\nFirst 5 rows:")
    print(raw_data.head())
    
    print("\nData Types:")
    print(raw_data.dtypes)
    
    print("\nMissing Values:")
    print(raw_data.isnull().sum())
    
    print("\nDescriptive Statistics:")
    print(raw_data.describe())
    
    return raw_data

def visualize_data(data):
    """
    Visualize the dataset to understand relationships.
    
    Args:
        data (pd.DataFrame): The dataframe to visualize
    """
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()
    
    # Correlation with target variable
    correlation_values = data.corr()['tip_amount'].drop('tip_amount')
    plt.figure(figsize=(10, 8))
    correlation_values.sort_values().plot(kind='barh')
    plt.title('Feature Correlation with Tip Amount')
    plt.xlabel('Correlation Coefficient')
    plt.tight_layout()
    plt.show()
    
    # Distribution of target variable
    plt.figure(figsize=(10, 6))
    sns.histplot(data['tip_amount'], kde=True)
    plt.title('Distribution of Tip Amount')
    plt.xlabel('Tip Amount')
    plt.tight_layout()
    plt.show()

def preprocess_data(data):
    """
    Preprocess the data for modeling.
    
    Args:
        data (pd.DataFrame): Raw dataframe
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print("Preprocessing data...")
    
    # Identify low correlation features
    correlation_values = data.corr()['tip_amount'].drop('tip_amount')
    low_corr_features = correlation_values[abs(correlation_values) < 0.05].index.tolist()
    
    print(f"Features with low correlation to target: {low_corr_features}")
    
    # Check for categorical variables
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print(f"Categorical columns found: {categorical_cols}")
        # One-hot encode categorical variables if needed
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    # Extract target variable
    y = data[['tip_amount']].values.astype('float32')
    
    # Drop target from feature matrix
    X_data = data.drop(['tip_amount'], axis=1)
    
    # Get feature names for later use
    feature_names = X_data.columns.tolist()
    
    # Convert to numpy array
    X = X_data.values
    
    # Option 1: Normalize features (L1 norm)
    # X = normalize(X, axis=1, norm='l1', copy=False)
    
    # Option 2: Standardize features (often better for tree-based models)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, shuffle=True
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, feature_names

def build_and_evaluate_model(X_train, X_test, y_train, y_test, feature_names):
    """
    Build and evaluate decision tree regression model.
    
    Args:
        X_train, X_test, y_train, y_test: Training and testing data
        feature_names: List of feature names for visualization
        
    Returns:
        tuple: Trained model and performance metrics
    """
    print("Building and evaluating model...")
    
    # Create base model
    base_model = DecisionTreeRegressor(
        criterion='squared_error',
        max_depth=6,  # Starting with a moderate depth
        random_state=42
    )
    
    # Perform cross-validation
    cv_scores = cross_val_score(
        base_model, X_train, y_train, 
        cv=5, scoring='neg_mean_squared_error'
    )
    
    print(f"Cross-validation MSE: {-cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    
    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'max_depth': [4, 6, 8, 10, 12],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(
        DecisionTreeRegressor(criterion='squared_error', random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train.ravel())
    
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print metrics
    print("\nModel Performance Metrics:")
    print(f"MSE: {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"R² Score: {r2:.3f}")
    
    return best_model, (mse, rmse, mae, r2)

def visualize_model(model, feature_names, X_train):
    """
    Visualize decision tree and feature importance.
    
    Args:
        model: Trained decision tree model
        feature_names: List of feature names
        X_train: Training data for feature importance
    """
    # Plot feature importance
    feature_importance = model.feature_importances_
    
    # Create DataFrame for better visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
    plt.title('Top 10 Feature Importance')
    plt.tight_layout()
    plt.show()
    
    # Plot decision tree (limited to depth 3 for clarity)
    plt.figure(figsize=(20, 10))
    plot_tree(
        model, 
        max_depth=3, 
        feature_names=feature_names,
        filled=True, 
        rounded=True, 
        fontsize=10
    )
    plt.title('Decision Tree Visualization (Limited to Depth 3)')
    plt.tight_layout()
    plt.show()

def compare_models(X_train, X_test, y_train, y_test):
    """
    Compare different max_depth values and their effect on model performance.
    
    Args:
        X_train, X_test, y_train, y_test: Training and testing data
    """
    depths = [2, 4, 6, 8, 10, 12, 15]
    train_scores = []
    test_scores = []
    mse_scores = []
    
    for depth in depths:
        model = DecisionTreeRegressor(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        
        # Calculate scores
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        train_scores.append(train_score)
        test_scores.append(test_score)
        mse_scores.append(mse)
    
    # Plot results
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(depths, train_scores, 'o-', label='Training R²')
    plt.plot(depths, test_scores, 'o-', label='Testing R²')
    plt.xlabel('Max Depth')
    plt.ylabel('R² Score')
    plt.title('R² Score vs. Tree Depth')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(depths, mse_scores, 'o-', color='red')
    plt.xlabel('Max Depth')
    plt.ylabel('MSE')
    plt.title('MSE vs. Tree Depth')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to execute the entire analysis pipeline.
    """
    # Dataset URL
    url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/pu9kbeSaAtRZ7RxdJKX9_A/yellow-tripdata.csv'
    
    # Load and explore data
    data = load_and_explore_data(url)
    
    # Visualize data
    visualize_data(data)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(data)
    
    # Build and evaluate model
    best_model, metrics = build_and_evaluate_model(X_train, X_test, y_train, y_test, feature_names)
    
    # Visualize model
    visualize_model(best_model, feature_names, X_train)
    
    # Compare different tree depths
    compare_models(X_train, X_test, y_train, y_test)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
