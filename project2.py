import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
df = pd.read_csv("data.csv")

print("--- Dataset Info ---")
print(df.info())

# 2. Preprocessing Dates & Age
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], format="mixed")

# Convert age from days to years
if "age" in df.columns:
    df["age_years"] = (df["age"] / 365).astype(int)

# Check missing and duplicate values
print("\nNull values:")
print(df.isnull().sum())

print("\nDuplicate rows count:", df.duplicated().sum())
# Drop duplicate rows if any exist
df = df.drop_duplicates()

# 3. Clean Outliers in Blood Pressure (Clinical Validity)
# Normal human systolic blood pressure (ap_hi) stays between 60 and 240
if "ap_hi" in df.columns and "ap_lo" in df.columns:
    df = df[(df["ap_hi"] >= 60) & (df["ap_hi"] <= 240)]
    df = df[(df["ap_lo"] >= 40) & (df["ap_lo"] <= 180)]

# 4. Exploratory Data Analysis (EDA)
print("\nSummary Statistics:")
print(df.describe())

# Target Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="disease", data=df, palette="pastel")
plt.title("Presence of Heart Disease in Patients")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Patient Count")
plt.show()

# Lifestyle Factors vs Disease
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
if "smoke" in df.columns:
    sns.countplot(x="smoke", hue="disease", data=df, ax=axes[0], palette="Set2")
    axes[0].set_title("Smoking vs Heart Disease")

if "cholesterol" in df.columns:
    sns.countplot(x="cholesterol", hue="disease", data=df, ax=axes[1], palette="Set2")
    axes[1].set_title("Cholesterol Level vs Heart Disease")
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# 5. Feature Selection & Train-Test Split
# Drop target, ID, raw 'age' in days (since age_years exists), and unused categorical fields
cols_to_drop = ["disease", "id", "date", "country", "occupation", "age"]
cols_present = [col for col in cols_to_drop if col in df.columns]

X = df.drop(columns=cols_present)
y = df["disease"]

# Convert any remaining categorical text columns (e.g., gender) to numeric dummies
X = pd.get_dummies(X, drop_first=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Stratify=y keeps the disease ratio balanced in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 6. Feature Scaling (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Model Training & Evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Support Vector Machine": SVC(kernel="rbf", random_state=42)
}

results = {}

for name, clf in models.items():
    # Fit model on scaled data
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    
    print(f"\n=================== {name} ===================")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot Confusion Matrix
    plt.figure(figsize=(5, 3.5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Healthy", "Disease"],
                yticklabels=["Healthy", "Disease"])
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# 8. Summary Comparison Table for Viva/Lab File
summary_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy (%)"])
summary_df["Accuracy (%)"] = (summary_df["Accuracy (%)"] * 100).round(2)
print("\n--- Model Performance Comparison ---")
print(summary_df.sort_values(by="Accuracy (%)", ascending=False).to_string(index=False))

# 1. Naye patient ke poore features define karein
new_patient = pd.DataFrame([{
    'age_years': 55,      # Umra (saal)
    'gender': 2,          # 1: Female, 2: Male
    'height': 168,        # Height (cm)
    'weight': 74.0,       # Weight (kg)
    'ap_hi': 145,         # Systolic BP (mmHg)
    'ap_lo': 95,          # Diastolic BP (mmHg)
    'cholesterol': 2,     # 1: Normal, 2: Above Normal, 3: High
    'gluc': 1,            # 1: Normal, 2: Above Normal, 3: High
    'smoke': 1,           # 0: No, 1: Yes
    'alco': 0,            # 0: No, 1: Yes
    'active': 1           # 0: No, 1: Yes
}])

# 2. Columns ka exact order match karein jo X mein tha
new_patient = new_patient[X.columns]

# 3. Scaler se transform karein
new_patient_scaled = scaler.transform(new_patient)

# 4. Model se predict karein
best_model = models["Decision Tree"]

# Prediction (0 = Healthy, 1 = Disease)
prediction = best_model.predict(new_patient_scaled)
probability = best_model.predict_proba(new_patient_scaled)[0][1] * 100

print(f"Prediction: {prediction[0]}")
if prediction[0] == 1:
    print(f"Result: High Risk of Heart Disease! (Risk Probability: {probability:.2f}%)")
else:
    print(f"Result: Low Risk / Healthy. (Risk Probability: {probability:.2f}%)")

