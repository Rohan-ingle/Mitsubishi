import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

# Load dataset
DATA_PATH = 'archive/creditcard.csv'
df = pd.read_csv(DATA_PATH)

# Features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train linear SVM for 10 epochs
model = SGDClassifier(loss='hinge', max_iter=10, tol=1e-3, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'svm_model.joblib')

# Export weights and bias for zkML
np.save('svm_weights.npy', model.coef_)
np.save('svm_bias.npy', model.intercept_)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")
