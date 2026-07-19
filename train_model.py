import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

print("="*60)
print("STROKESHIELD AI - Model Training")
print("="*60)

# Check if dataset exists
dataset_path = 'healthcare-dataset-stroke-data.csv'
if not os.path.exists(dataset_path):
    print(f"\n❌ ERROR: Dataset file not found: {dataset_path}")
    print("Please download the dataset from Kaggle and place it in this folder.")
    print("URL: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset")
    exit(1)

print("\n1. Loading dataset...")
df = pd.read_csv(dataset_path)
print(f"   ✅ Loaded: {df.shape[0]} patients, {df.shape[1]} columns")
print(f"   Stroke cases: {df['stroke'].sum()} ({df['stroke'].mean()*100:.1f}%)")

print("\n2. Cleaning data...")
df['bmi'] = df['bmi'].fillna(df['bmi'].median())
df = df.drop('id', axis=1)
print("   ✅ Missing BMI values filled, ID column removed")

print("\n3. Encoding categories...")
le_gender = LabelEncoder()
le_married = LabelEncoder()
le_work = LabelEncoder()
le_residence = LabelEncoder()
le_smoking = LabelEncoder()

df['gender'] = le_gender.fit_transform(df['gender'])
df['ever_married'] = le_married.fit_transform(df['ever_married'])
df['work_type'] = le_work.fit_transform(df['work_type'])
df['Residence_type'] = le_residence.fit_transform(df['Residence_type'])
df['smoking_status'] = le_smoking.fit_transform(df['smoking_status'])
print("   ✅ Text categories converted to numbers")

print("\n4. Training model...")
X = df.drop('stroke', axis=1)
y = df['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train_scaled, y_train)
print("   ✅ Model trained!")

print("\n5. Evaluating model...")
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]
print(f"   AUC-ROC Score: {roc_auc_score(y_test, y_prob):.3f}")

print("\n6. Saving model files...")
joblib.dump(model, 'stroke_risk_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_gender, 'le_gender.pkl')
joblib.dump(le_married, 'le_married.pkl')
joblib.dump(le_work, 'le_work.pkl')
joblib.dump(le_residence, 'le_residence.pkl')
joblib.dump(le_smoking, 'le_smoking.pkl')
print("   ✅ All files saved!")

print("\n7. Testing prediction function...")

def get_risk_score(age, gender, hypertension, heart_disease, ever_married,
                   work_type, Residence_type, avg_glucose_level, bmi, smoking_status):
    """Predict stroke risk from user health data."""
    
    g = le_gender.transform([gender])[0]
    m = le_married.transform([ever_married])[0]
    w = le_work.transform([work_type])[0]
    r = le_residence.transform([Residence_type])[0]
    s = le_smoking.transform([smoking_status])[0]
    
    features = pd.DataFrame([{
        'gender': g, 'age': age, 'hypertension': hypertension,
        'heart_disease': heart_disease, 'ever_married': m,
        'work_type': w, 'Residence_type': r,
        'avg_glucose_level': avg_glucose_level, 'bmi': bmi,
        'smoking_status': s
    }])
    
    scaled = scaler.transform(features)
    prob = model.predict_proba(scaled)[0][1]
    score = int(prob * 100)
    
    # Boost for obvious risk factors
    if age > 60: score += 15
    if age > 75: score += 10
    if avg_glucose_level > 140: score += 10
    if avg_glucose_level > 200: score += 15
    if bmi > 30: score += 5
    if smoking_status == 'smokes': score += 10
    if hypertension == 1: score += 5
    if heart_disease == 1: score += 10
    
    score = min(score, 99)
    
    if score < 30: cat = "Low"
    elif score < 60: cat = "Moderate"
    elif score < 80: cat = "High"
    else: cat = "Critical"
    
    return {"risk_score": score, "risk_category": cat}

# Test users
tests = [
    ("Healthy 25-year-old", 25, 'Female', 0, 0, 'No', 'Private', 'Urban', 85, 22, 'never smoked'),
    ("Pre-diabetic 52-year-old smoker", 52, 'Male', 1, 0, 'Yes', 'Private', 'Urban', 142, 31, 'smokes'),
    ("Elderly diabetic with heart disease", 75, 'Male', 1, 1, 'Yes', 'Self-employed', 'Rural', 220, 35, 'smokes'),
]

for name, *params in tests:
    result = get_risk_score(*params)
    print(f"\n   {name}:")
    print(f"   Score: {result['risk_score']}/100 | {result['risk_category']}")

print("\n" + "="*60)
print("✅ MODEL TRAINING COMPLETE!")
print("="*60)