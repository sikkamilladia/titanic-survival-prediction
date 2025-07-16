import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Drop kolom yang ga perlu
train_df = train_df.drop(['Cabin', 'Ticket', 'Name'], axis=1)
test_df = test_df.drop(['Cabin', 'Ticket', 'Name'], axis=1)

# Isi missing values
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

# Encode kolom kategori secara manual
sex_mapping = {'male': 0, 'female': 1}
embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}

train_df['Sex'] = train_df['Sex'].map(sex_mapping)
test_df['Sex'] = test_df['Sex'].map(sex_mapping)

train_df['Embarked'] = train_df['Embarked'].map(embarked_mapping)
test_df['Embarked'] = test_df['Embarked'].map(embarked_mapping)

# Pisahin fitur dan label
X = train_df.drop(['Survived', 'PassengerId'], axis=1)
y = train_df['Survived']

# Split jadi train/validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

# Prediksi test data
X_test = test_df.drop(['PassengerId'], axis=1)
test_preds = model.predict(X_test)

# Save ke CSV
output = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': test_preds
})
output.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created!")