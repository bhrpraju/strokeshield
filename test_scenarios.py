import joblib
import pandas as pd
import numpy as np

print("="*60)
print("STROKESHIELD AI - Extended Testing")
print("="*60)

# Load saved model files
print("\nLoading model files...")
model = joblib.load('stroke_risk_model.pkl')
scaler = joblib.load('scaler.pkl')
le_gender = joblib.load('le_gender.pkl')
le_married = joblib.load('le_married.pkl')
le_work = joblib.load('le_work.pkl')
le_residence = joblib.load('le_residence.pkl')
le_smoking = joblib.load('le_smoking.pkl')
print("✅ All files loaded!")

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

# ============================================
# EXTENDED TEST SCENARIOS
# ============================================

tests = [
    # Young & healthy
    ("Young athlete", 22, 'Female', 0, 0, 'No', 'Private', 'Urban', 80, 21, 'never smoked'),
    ("Young smoker", 25, 'Male', 0, 0, 'No', 'Private', 'Urban', 90, 24, 'smokes'),
    
    # Middle age variations
    ("Healthy 40s", 45, 'Female', 0, 0, 'Yes', 'Private', 'Urban', 95, 25, 'never smoked'),
    ("Overweight 40s", 48, 'Male', 0, 0, 'Yes', 'Private', 'Urban', 110, 29, 'formerly smoked'),
    ("Pre-diabetic 50s", 55, 'Male', 1, 0, 'Yes', 'Private', 'Urban', 145, 32, 'smokes'),
    
    # High risk combinations
    ("Diabetic smoker", 60, 'Male', 1, 0, 'Yes', 'Self-employed', 'Urban', 180, 33, 'smokes'),
    ("Heart disease only", 65, 'Female', 0, 1, 'Yes', 'Govt_job', 'Rural', 120, 28, 'never smoked'),
    ("Triple threat", 70, 'Male', 1, 1, 'Yes', 'Private', 'Urban', 200, 35, 'smokes'),
    
    # Edge cases
    ("Very elderly healthy", 80, 'Female', 0, 0, 'Yes', 'Private', 'Rural', 100, 23, 'never smoked'),
    ("Very elderly smoker", 82, 'Male', 1, 1, 'Yes', 'Self-employed', 'Rural', 210, 36, 'smokes'),
    ("Extreme glucose", 50, 'Male', 0, 0, 'Yes', 'Private', 'Urban', 250, 27, 'formerly smoked'),
]

print("\n" + "="*60)
print("RUNNING ALL TEST SCENARIOS")
print("="*60)

for name, *params in tests:
    result = get_risk_score(*params)
    print(f"\n{name}:")
    print(f"  Score: {result['risk_score']}/100 | Category: {result['risk_category']}")

print("\n" + "="*60)
print("✅ ALL TESTS COMPLETE!")
print("="*60)