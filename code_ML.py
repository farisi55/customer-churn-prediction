import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Step 1: Load the dataset
url = "https://raw.githubusercontent.com/farisi55/customer-churn-prediction/main/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
data = pd.read_csv(url)

# Step 2: Data Preprocessing
# Handle missing values
data.replace(" ", pd.NA, inplace=True)  # Convert empty strings to NaN
data.dropna(inplace=True)  # Drop rows with missing values

# Split data into features and target variable
X = data.drop('Churn', axis=1)
y = data['Churn']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps for numerical and categorical features
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                        'PhoneService', 'MultipleLines', 'InternetService',
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                        'TechSupport', 'StreamingTV', 'StreamingMovies',
                        'Contract', 'PaperlessBilling', 'PaymentMethod']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
])

# Define models
models = {
    'Logistic_Regression': LogisticRegression(),
    'Support_Vector_Machine': SVC(),
    'Random_Forest': RandomForestClassifier(),
    'K-Nearest_Neighbors': KNeighborsClassifier()
}

# Define parameter grids for hyperparameter tuning
param_grids = {
    'Logistic_Regression': {'classifier__C': [0.1, 1, 10, 100]},
    'Support_Vector_Machine': {'classifier__C': [0.1, 1, 10, 100], 'classifier__kernel': ['linear', 'rbf']},
    'Random_Forest': {'classifier__n_estimators': [50, 100, 200]},
    'K-Nearest_Neighbors': {'classifier__n_neighbors': [3, 5, 7]}
}

# Train and tune models
for name, model in models.items():
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
    try:
        grid_search = GridSearchCV(clf, param_grids[name], cv=5)
        grid_search.fit(X_train, y_train)
        print(f"Best parameters for {name}: {grid_search.best_params_}")
        print(f"Best score for {name}: {grid_search.best_score_}")

        # Export model
        joblib.dump(grid_search, f"{name}_model.joblib")
    except Exception as e:
        print(f"Error occurred during fitting for {name}: {e}")