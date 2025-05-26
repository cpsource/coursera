# weather.py
#
# 1) file -> download (FinalProject_AUSWeather.ipy)
# 2) This notebook will then be gtraded using AI grader in the subsequest section.
# 3) Copy/paste your makrdown responses in the subsequent AI Mark assighnment
#

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# Load the data

url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
df = pd.read_csv(url)
print(df.head())
print(df.count())

# Drop all rows with missing values

df = df.dropna()
print(df.info())
print(df.columns)

#update names
df = df.rename(columns={'RainToday': 'RainYesterday',
                        'RainTomorrow': 'RainToday'
                        })

# Data Granularity

# Location Selecton
df = df[df.Location.isin(['Melbourne','MelbourneAirport','Watsonia',])]
print(df. info())


def date_to_season(date):
    month = date.month
    if (month == 12) or (month == 1) or (month == 2):
        return 'Summer'
    elif (month == 3) or (month == 4) or (month == 5):
        return 'Autumn'
    elif (month == 6) or (month == 7) or (month == 8):
        return 'Winter'
    elif (month == 9) or (month == 10) or (month == 11):
        return 'Spring'

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Apply the function to the 'Date' column
df['Season'] = df['Date'].apply(date_to_season)

# Drop the 'Date' column
df = df.drop(columns='Date')

# Show the resulting DataFrame
print(df)

# Exercise 2. Define the feature and target dataframes

X = df.drop(columns='RainToday', axis=1)
y = df['RainToday']

# Exercise 3. How balanced are the classes?

# Check class balance in the target
print(y.value_counts())

# ## Write your response here and convert the cell to a markdown.
#How often does it rain annualy in the Melborne area? 23%
#How accurate would you be if you just assumed it wont rain every day? 76%
#Is this a balanced dataset? No, this dataset is not fully balanced.
#Next steps? resampling: oversample rainy days or undersample dry days, use class weights in your classifier, evaluate with precission, recall, and f1-score, engineer new features, try tree based models

# Exercise 5. Split data into training and test sets, ensuring target stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# transform
numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Define separate transformers for both feature types and combine them into a single preprocessing transformer
# Scale the numeric features
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# One-hot encode the categoricals
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Exercise 7 - combine
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Exercise 8. Create a pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define a parameter grid to use in a cross validation grid search model optimizer
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

# Select a cross-validation method, ensuring target stratification during validation¶
cv = StratifiedKFold(n_splits=5, shuffle=True)

grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

# print the best parameters and best crossvalidation score
print("\nBest parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Exercise 10. Display your model's estimated score
test_score = grid_search.score(X_test, y_test)
print("Test set score: {:.2f}".format(test_score))

# Exercise 11. Get the model predictions ...
y_pred = grid_search.predict(X_test)

# Exercise 12. Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Exercise 13. Plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# points to note 2
# what is the true positive rate
print(classification_report(y_test, y_pred))

# Exercise 14. Extract the feature importances
feature_importances = grid_search.best_estimator_['classifier'].feature_importances_

# Now let's extract the feature importances and plot them as a bar graph.
# Combine numeric and categorical feature names
feature_names = numeric_features + list(grid_search.best_estimator_['preprocessor']
                                        .named_transformers_['cat']
                                        .named_steps['onehot']
                                        .get_feature_names_out(categorical_features))

feature_importances = grid_search.best_estimator_['classifier'].feature_importances_

N = 20  # Change this number to display more or fewer features

importance_df = pd.DataFrame({'Feature': feature_names,
                              'Importance': feature_importances
                             }).sort_values(by='Importance', ascending=False)
top_features = importance_df.head(N)

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature on top
plt.title(f'Top {N} Most Important Features in predicting whether it will rain today')
plt.xlabel('Importance Score')
plt.show()

# Point to note - 3 - Identify the most important feature for predicting whether it will rain based on the feature importance bar graph.

# Get the fitted preprocessor from the pipeline
fitted_preprocessor = grid_search.best_estimator_['preprocessor']

# Extract the feature names
numeric_features = fitted_preprocessor.transformers_[0][2]
categorical_encoder = fitted_preprocessor.transformers_[1][1]
categorical_features_encoded = categorical_encoder.get_feature_names_out(categorical_features)

# Combine all feature names
feature_names = list(numeric_features) + list(categorical_features_encoded)

# Get importances from the fitted classifier
importances = grid_search.best_estimator_['classifier'].feature_importances_

# Create a DataFrame to sort and visualize
import pandas as pd
import matplotlib.pyplot as plt

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot top 10
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'][:10], importance_df['Importance'][:10])
plt.xlabel('Feature Importance')
plt.title('Top 10 Features Predicting Rain')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Try another model
# Replace RandomForestClassifier with LogisticRegression
pipeline.set_params(classifier=LogisticRegression(random_state=42))

# Update the model's estimator to use the new pipeline
grid_search.estimator = pipeline

# Define a new grid with Logistic Regression parameters
param_grid = {
    'classifier__solver': ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight': [None, 'balanced']
}

# Update grid_search's parameter grid
grid_search.param_grid = param_grid

# Fit the updated pipeline with LogisticRegression
grid_search.fit(X_train, y_train)

# Make predictions
y_pred = grid_search.predict(X_test)

# compare the results to your previous model
print(classification_report(y_test, y_pred))

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')

# Set the title and labels
plt.title('Titanic Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.tight_layout()
plt.show()

from sklearn.metrics import accuracy_score, recall_score

# Evaluate Random Forest
rf_y_pred = grid_search.predict(X_test)  # This now contains LogisticRegression, so we need to re-fit with RF first
pipeline.set_params(classifier=RandomForestClassifier(random_state=42))
grid_search.estimator = pipeline
grid_search.param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}
grid_search.fit(X_train, y_train)
rf_y_pred = grid_search.predict(X_test)
rf_acc = accuracy_score(y_test, rf_y_pred)
rf_tpr = recall_score(y_test, rf_y_pred, pos_label='Yes')

# Evaluate Logistic Regression (already fitted)
pipeline.set_params(classifier=LogisticRegression(random_state=42))
grid_search.estimator = pipeline
grid_search.param_grid = {
    'classifier__solver': ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight': [None, 'balanced']
}
grid_search.fit(X_train, y_train)
lr_y_pred = grid_search.predict(X_test)
lr_acc = accuracy_score(y_test, lr_y_pred)
lr_tpr = recall_score(y_test, lr_y_pred, pos_label='Yes')

# Final comparison
print("\nComparison of Model Performance:")
print(f"{'Model':<25} {'Accuracy':<10} {'TPR (Recall on Rain)':<20}")
print(f"{'-'*55}")
print(f"{'Random Forest':<25} {rf_acc:.2f}      {rf_tpr:.2f}")
print(f"{'Logistic Regression':<25} {lr_acc:.2f}      {lr_tpr:.2f}")

