import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt


df = pd.read_csv("diabetes_large_dataset (1).csv")
df = df.drop_duplicates()

numeric_cols = df.select_dtypes(include=['int64','float64']).columns
df[numeric_cols] = df[numeric_cols].replace(0, np.nan)
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

def cap_outliers(col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower, lower,
                       np.where(df[col] > upper, upper, df[col]))

for col in numeric_cols:
    cap_outliers(col)

df["hypertension"] = np.where(
    (df["blood_pressure_systolic"] >= 140) | 
    (df["blood_pressure_diastolic"] >= 90), 1, 0
)
plt.figure()
plt.hist(df["age"], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.figure()
plt.hist(df["bmi"], bins=20)
plt.title("BMI Distribution")
plt.xlabel("BMI")
plt.ylabel("Frequency")

plt.figure()
plt.scatter(df["bmi"], df["blood_pressure_systolic"])
plt.title("BMI vs Systolic Blood Pressure")
plt.xlabel("BMI")
plt.ylabel("Systolic BP")

plt.figure()
corr = df.corr()
plt.imshow(corr, cmap="viridis")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap")
plt.show()
features = df[[
    "age",
    "gender",
    "height_cm",
    "weight_kg",
    "bmi",
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "heart_rate"
]]

X = features
y = df["hypertension"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(model, "hypertension_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and Scaler saved successfully!")