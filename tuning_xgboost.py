import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder

# Load data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Data cleaning & feature engineering (simple)
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

le = LabelEncoder()
train_df['Sex'] = le.fit_transform(train_df['Sex'])
train_df['Embarked'] = le.fit_transform(train_df['Embarked'])

# Fitur minimal
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train_df[features]
y = train_df['Survived']

# Grid Search for XGBoost
params = {
    'max_depth': [3, 4, 5],
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2]
}

xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)

grid_search = GridSearchCV(estimator=xgb_model, param_grid=params,
                           scoring='accuracy', cv=5, verbose=1)
grid_search.fit(X, y)

print("Best Params:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)