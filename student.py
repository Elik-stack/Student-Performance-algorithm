
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Step 1: Generate large dataset
# -------------------------------
num_students = 500  # total number of students
student_names = [f"Student_{i+1}" for i in range(num_students)]

data = []
for name in student_names:
    hours_studied = random.randint(1, 10)  # 1-10 hours
    attendance = max(35, min(100, int(hours_studied*10 + random.gauss(0,10))))
    previous_grade = max(25, min(95, int(attendance*0.8 + random.gauss(0,10))))
    # Simple pass/fail rule
    pass_exam = 1 if (0.5*hours_studied + 0.5*previous_grade/10) > 5 else 0
    data.append([name, hours_studied, attendance, previous_grade, pass_exam])

# -------------------------------
# Step 2: Create DataFrame and save CSV
# -------------------------------
df = pd.DataFrame(data, columns=[
    "name", "hours_studied", "attendance", "previous_grade", "pass_exam"
])
df.to_csv("students_large.csv", index=False)
print("Generated 'students_large.csv' with 500 students.\n")
print(df.head())
print("\nTotal rows:", len(df))

# -------------------------------
# Step 3: Data Exploration
# -------------------------------
print("\nSummary Statistics:")
print(df.describe())

# Correlation heatmap (numeric columns only)
plt.figure(figsize=(6,4))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# -------------------------------
# Step 4: Prepare data for ML
# -------------------------------
X = df[['hours_studied', 'attendance', 'previous_grade']]
y = df['pass_exam']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# Step 5: Train Logistic Regression Model
# -------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# -------------------------------
# Step 6: Evaluate Model
# -------------------------------
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize prediction distribution
sns.histplot(y_pred, kde=False, color='green')
plt.title("Prediction Distribution (0=Fail, 1=Pass)")
plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.show()

# -------------------------------
# Step 7: Save the trained model
# -------------------------------
joblib.dump(model, 'student_performance_model_large.pkl')
print("\nModel saved as 'student_performance_model_large.pkl'")

# -------------------------------
# Step 8: Predict for a new student
# -------------------------------
new_student = pd.DataFrame({
    'hours_studied': [7],
    'attendance': [85],
    'previous_grade': [75]
})

new_scaled = scaler.transform(new_student)
new_pred = model.predict(new_scaled)
print("\nNew Student Prediction (1=Pass, 0=Fail):", new_pred[0])
