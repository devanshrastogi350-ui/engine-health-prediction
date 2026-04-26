# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 2: Dummy dataset create karte hain
data = {
    'temperature': [80, 95, 70, 110, 85, 120, 60, 105],
    'vibration': [30, 50, 20, 60, 35, 70, 15, 65],
    'pressure': [40, 55, 35, 65, 45, 75, 30, 70],
    'rpm': [3000, 4000, 2500, 4500, 3200, 4800, 2000, 4600],
    'status': [0, 1, 0, 1, 0, 1, 0, 1]  # 0 = Healthy, 1 = Faulty
}

df = pd.DataFrame(data)

# Step 3: Split data
X = df.drop('status', axis=1)
y = df['status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 4: Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 5: Prediction
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 6: Test custom input
sample = pd.DataFrame([[100, 60, 60, 4200]], 
columns=['temperature','vibration','pressure','rpm'])
result = model.predict(sample)

if result[0] == 1:
    print("Engine is Faulty ⚠️")
else:
    print("Engine is Healthy ✅")