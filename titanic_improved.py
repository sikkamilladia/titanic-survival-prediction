import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Combine buat feature engineering bareng
train_df['TrainSet'] = True
test_df['TrainSet'] = False
test_df['Survived'] = np.nan
combined = pd.concat([train_df, test_df], sort=False)

# FEATURE ENGINEERING
combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1
combined['IsAlone'] = (combined['FamilySize'] == 1).astype(int)
combined['Title'] = combined['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
combined['Title'] = combined['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 
                                               'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
combined['Title'] = combined['Title'].replace('Mlle', 'Miss')
combined['Title'] = combined['Title'].replace('Ms', 'Miss')
combined['Title'] = combined['Title'].replace('Mme', 'Mrs')

# Handle missing values
combined['Age'] = combined['Age'].fillna(combined['Age'].median())
combined['Fare'] = combined['Fare'].fillna(combined['Fare'].median())
combined['Embarked'] = combined['Embarked'].fillna(combined['Embarked'].mode()[0])

# Encode kategorikal
label = LabelEncoder()
combined['Sex'] = label.fit_transform(combined['Sex'])
combined['Embarked'] = label.fit_transform(combined['Embarked'])
combined['Title'] = label.fit_transform(combined['Title'])

# Drop kolom ga penting
combined.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)

# Split kembali
train_df = combined[combined['TrainSet'] == True].drop(['TrainSet'], axis=1)
test_df = combined[combined['TrainSet'] == False].drop(['TrainSet', 'Survived'], axis=1)

# Fitur & Label
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']

# Train/Test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# MODEL: XGBoost
model = xgb.XGBClassifier(
    learning_rate=0.1,
    max_depth=5,
    n_estimators=200,
    eval_metric='logloss',
    random_state=42
)
# Train model
model.fit(X_train, y_train)

# Validation
val_preds = model.predict(X_val)
acc = accuracy_score(y_val, val_preds)
print(f"Validation Accuracy: {acc:.4f}")

# Cross-validation score (bonus)
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-Validation Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Predict test set
final_preds = model.predict(test_df)

# Submission
submission = pd.read_csv('test.csv')[['PassengerId']]
submission['Survived'] = final_preds.astype(int)
submission.to_csv('submission.csv', index=False)
print("Improved submission file saved as 'submission.csv'")