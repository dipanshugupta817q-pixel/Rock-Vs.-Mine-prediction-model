# ---------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ---------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------------------------------------------
# 2. LOAD SONAR DATASET (FIXED)
# ---------------------------------------------------------------
data = pd.read_csv(
    "sonar.csv",
    header=None,
    sep=",",
    encoding='latin1',
    on_bad_lines='skip'
)

print("Dataset Loaded Successfully!\n")
print(data.head())

# ---------------------------------------------------------------
# 3. FEATURES & LABELS
# ---------------------------------------------------------------
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Convert labels: R = 0, M = 1
y = y.map({"R": 0, "M": 1})

# ---------------------------------------------------------------
# 4. TRAIN–TEST SPLIT
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------
# 5. FEATURE SCALING
# ---------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 6. TRAIN MODEL
# ---------------------------------------------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)

# ---------------------------------------------------------------
# 7. MODEL EVALUATION
# ---------------------------------------------------------------
y_pred = model.predict(X_test_scaled)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------------------------------------------------
# 8. TEST MODEL ON RANDOM SAMPLE
# ---------------------------------------------------------------
sample = np.random.rand(60).reshape(1, -1)
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)

print("\nRandom Sample Prediction:", "MINE" if prediction[0] == 1 else "ROCK")