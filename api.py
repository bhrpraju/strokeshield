from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import joblib
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="StrokeShield AI - Proactive Prevention API")

# Allow Lovable to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# LOAD MODEL FILES
# ============================================
print("Loading StrokeShield model...")
model = joblib.load('stroke_risk_model.pkl')
scaler = joblib.load('scaler.pkl')
le_gender = joblib.load('le_gender.pkl')
le_married = joblib.load('le_married.pkl')
le_work = joblib.load('le_work.pkl')
le_residence = joblib.load('le_residence.pkl')
le_smoking = joblib.load('le_smoking.pkl')
print("✅ Model loaded!")

# ============================================
# DATA MODELS
# ============================================
class HealthData(BaseModel):
    age: float
    gender: str
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str

class LabValues(BaseModel):
    hba1c: Optional[float] = None
    ldl: Optional[float] = None
    hdl: Optional[float] = None
    triglycerides: Optional[float] = None
    crp: Optional[float] = None
    homocysteine: Optional[float] = None
    fasting_insulin: Optional[float] = None
    apob: Optional[float] = None

# ============================================
# CORE PREDICTION FUNCTION
# ============================================
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
    
    # Boost for risk factors
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
    
    importance = model.feature_importances_
    feature_names = ['gender', 'age', 'hypertension', 'heart_disease',
                     'ever_married', 'work_type', 'Residence_type',
                     'avg_glucose_level', 'bmi', 'smoking_status']
    top_indices = np.argsort(importance)[-3:][::-1]
    top_drivers = [feature_names[i] for i in top_indices]
    
    return score, cat, top_drivers

# ============================================
# LAB SUGGESTION ENGINE (The Attia Layer)
# ============================================
def get_lab_suggestions(score, category, top_drivers, data):
    """Generate proactive lab test suggestions with evidence."""
    
    labs = []
    
    # HbA1c — Always suggest if glucose elevated or age >45
    if data.avg_glucose_level > 110 or data.age > 45 or "avg_glucose_level" in top_drivers:
        labs.append({
            "test": "HbA1c",
            "full_name": "Hemoglobin A1c",
            "why": f"Your glucose is {data.avg_glucose_level}. HbA1c shows your 3-month average — fasting glucose alone misses 30% of pre-diabetes cases.",
            "target": "< 5.7% (normal), < 6.5% (pre-diabetic threshold)",
            "priority": "HIGH" if data.avg_glucose_level > 140 else "MEDIUM",
            "action_if_high": "Cut refined carbs, walk 30 min post-meal, consider metformin discussion with doctor"
        })
    
    # Fasting Insulin + HOMA-IR — If glucose or BMI elevated
    if data.avg_glucose_level > 100 or data.bmi > 28:
        labs.append({
            "test": "Fasting Insulin + HOMA-IR",
            "full_name": "Insulin Resistance Panel",
            "why": f"Your BMI is {data.bmi} and glucose is {data.avg_glucose_level}. Insulin resistance precedes diabetes by 10-15 years. HOMA-IR catches it before HbA1c rises.",
            "target": "HOMA-IR < 2.0 (ideal), < 2.5 (acceptable)",
            "priority": "HIGH" if data.bmi > 30 else "MEDIUM",
            "action_if_high": "Time-restricted eating (16:8), eliminate liquid sugar, increase resistance training"
        })
    
    # Lipid Panel + ApoB — Always for anyone with risk
    if score > 30:
        labs.append({
            "test": "Lipid Panel + ApoB",
            "full_name": "Advanced Lipid Profile",
            "why": "Standard LDL is incomplete. ApoB counts the actual number of atherogenic particles — the causal agents of plaque. One ApoB particle = one potential plaque.",
            "target": "ApoB < 60 mg/dL (optimal for high-risk), < 80 (acceptable)",
            "priority": "HIGH" if data.age > 60 or data.heart_disease == 1 else "MEDIUM",
            "action_if_high": "Eliminate trans fats, increase omega-3, consider statin if ApoB > 100 + family history"
        })
    
    # hs-CRP — Inflammation marker
    if score > 40 or data.bmi > 30 or data.smoking_status == 'smokes':
        labs.append({
            "test": "hs-CRP",
            "full_name": "High-Sensitivity C-Reactive Protein",
            "why": "Inflammation is the silent driver of arterial damage. hs-CRP is an independent stroke predictor — even when cholesterol is normal.",
            "target": "< 1.0 mg/L (low risk), < 3.0 (average)",
            "priority": "MEDIUM",
            "action_if_high": "Eliminate seed oils, increase omega-3 (2g EPA/DHA daily), sleep 7-8 hours, manage stress"
        })
    
    # Homocysteine — If age >50 or any vascular risk
    if data.age > 50 or score > 50:
        labs.append({
            "test": "Homocysteine",
            "full_name": "Homocysteine Level",
            "why": "Elevated homocysteine damages blood vessel walls directly. 40% of stroke patients have levels >12. Often due to MTHFR gene variant — fixable with B-vitamins.",
            "target": "< 10 μmol/L (optimal), < 12 (acceptable)",
            "priority": "MEDIUM",
            "action_if_high": "Methylated B-complex (B6, B9, B12), reduce alcohol, increase leafy greens"
        })
    
    # NT-proBNP — Heart strain, especially if hypertension or age >65
    if data.hypertension == 1 or data.age > 65 or data.heart_disease == 1:
        labs.append({
            "test": "NT-proBNP",
            "full_name": "N-terminal pro-B-type Natriuretic Peptide",
            "why": "Detects early heart strain before symptoms appear. Elevated levels predict AFib (which causes 1 in 4 strokes) and heart failure years in advance.",
            "target": "< 125 pg/mL (age <75), < 450 (age 75+)",
            "priority": "HIGH" if data.heart_disease == 1 else "MEDIUM",
            "action_if_high": "Cardiology referral, BP optimization, sodium restriction <2g/day"
        })
    
    # Sleep Apnea Screen — If BMI high, age >50, or hypertension
    if data.bmi > 30 or data.age > 50 or data.hypertension == 1:
        labs.append({
            "test": "Sleep Apnea Screening (STOP-BANG)",
            "full_name": "Obstructive Sleep Apnea Assessment",
            "why": "Untreated sleep apnea increases stroke risk 2-3x. Causes nocturnal hypertension and AFib. 80% of cases are undiagnosed.",
            "target": "STOP-BANG score < 3 (low risk)",
            "priority": "MEDIUM",
            "action_if_high": "Sleep study (polysomnography), CPAP if AHI >15, weight loss priority #1"
        })
    
    # Carotid Ultrasound — If age >60 or high risk
    if data.age > 60 or score > 70:
        labs.append({
            "test": "Carotid Ultrasound",
            "full_name": "Carotid Artery Doppler",
            "why": "Direct visualization of plaque in your neck arteries. Shows stroke risk that blood tests miss. 'See the disease' — Peter Attia.",
            "target": "IMT < 0.9mm, no plaque",
            "priority": "HIGH" if score > 70 else "LOW",
            "action_if_high": "Aggressive lipid lowering, aspirin discussion, vascular surgery consult if >70% stenosis"
        })
    
    # Sort by priority
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    labs.sort(key=lambda x: priority_order[x["priority"]])
    
    return labs

# ============================================
# PREVENTION PLAN GENERATOR
# ============================================
def get_prevention_plan(score, category, top_drivers, data):
    """Generate personalized 30-day prevention plan."""
    
    plan = {
        "overview": f"Your stroke risk is {score}/100 ({category}). Based on your top risk factors ({', '.join(top_drivers)}), here's your personalized plan.",
        "weeks": []
    }
    
    # Week 1: Quick wins
    week1_actions = []
    if "avg_glucose_level" in top_drivers and data.avg_glucose_level > 140:
        week1_actions.append("Eliminate all liquid sugar (soda, juice, sweetened coffee). Replace with water or black coffee.")
    if "bmi" in top_drivers and data.bmi > 30:
        week1_actions.append("Start 16:8 time-restricted eating. No calories between 8 PM and 12 PM.")
    if "smoking_status" in top_drivers and data.smoking_status == 'smokes':
        week1_actions.append("Set a quit date within 14 days. Download QuitGuide app. Tell 3 people your plan.")
    if "age" in top_drivers and data.age > 60:
        week1_actions.append("Schedule all 7 recommended lab tests this week. Book within 7 days.")
    if "hypertension" in top_drivers and data.hypertension == 1:
        week1_actions.append("Buy home BP monitor. Take readings twice daily (morning/evening). Log in app.")
    
    if not week1_actions:
        week1_actions.append("Maintain current healthy habits. Focus on prevention labs this week.")
    
    plan["weeks"].append({
        "week": 1,
        "theme": "Quick Wins & Baseline",
        "actions": week1_actions,
        "labs_to_get": [lab["test"] for lab in get_lab_suggestions(score, category, top_drivers, data) if lab["priority"] == "HIGH"][:3]
    })
    
    # Week 2: Build habits
    week2_actions = []
    if "avg_glucose_level" in top_drivers:
        week2_actions.append("Add 10-minute walk after every meal. Lowers post-meal glucose spikes by 30%.")
    if "bmi" in top_drivers:
        week2_actions.append("Replace one processed meal/day with whole foods (vegetables + protein + healthy fat).")
    if data.smoking_status == 'smokes':
        week2_actions.append("Use nicotine replacement (patch/gum) if cravings hit. Cut cigarettes by 25%.")
    if data.hypertension == 1:
        week2_actions.append("Reduce sodium to <2,300mg/day. Check labels. No restaurant food this week.")
    
    if not week2_actions:
        week2_actions.append("Add 2 strength training sessions (20 min each). Bodyweight squats, push-ups, planks.")
    
    plan["weeks"].append({
        "week": 2,
        "theme": "Build the Habit Stack",
        "actions": week2_actions,
        "labs_to_get": [lab["test"] for lab in get_lab_suggestions(score, category, top_drivers, data) if lab["priority"] == "MEDIUM"][:2]
    })
    
    # Week 3: Intensify
    week3_actions = []
    if data.age > 50:
        week3_actions.append("Increase to 150 min/week moderate exercise (brisk walking, swimming, cycling).")
    if data.bmi > 28:
        week3_actions.append("Add resistance training 3x/week. Muscle is metabolic currency — more muscle = better glucose control.")
    if data.avg_glucose_level > 140:
        week3_actions.append("Eliminate all refined grains (white bread, white rice, pasta). Replace with quinoa, lentils, vegetables.")
    if data.smoking_status == 'smokes':
        week3_actions.append("Cut to 50% of baseline. Use 4 D's: Delay, Deep breathe, Drink water, Do something else.")
    
    if not week3_actions:
        week3_actions.append("Optimize sleep: 7-8 hours, consistent bedtime, no screens 1 hour before bed.")
    
    plan["weeks"].append({
        "week": 3,
        "theme": "Intensify & Optimize",
        "actions": week3_actions,
        "labs_to_get": ["Review all lab results with doctor. Adjust plan based on findings."]
    })
    
    # Week 4: Lock in
    plan["weeks"].append({
        "week": 4,
        "theme": "Lock In & Measure",
        "actions": [
            "Re-check home BP, weight, and waist circumference.",
            "Log all habits for 7 consecutive days.",
            "Schedule follow-up with doctor to review progress.",
            "Re-calculate stroke risk in StrokeShield to see improvement."
        ],
        "labs_to_get": ["Repeat any abnormal labs. Target improvement from baseline."]
    })
    
    return plan

# ============================================
# DOCTOR REPORT GENERATOR
# ============================================
def generate_doctor_report(score, category, top_drivers, data, labs):
    """Generate clinical summary for physician review."""
    
    report = {
        "patient_summary": {
            "age": data.age,
            "gender": data.gender,
            "bmi": data.bmi,
            "glucose": data.avg_glucose_level,
            "hypertension": "Yes" if data.hypertension == 1 else "No",
            "heart_disease": "Yes" if data.heart_disease == 1 else "No",
            "smoking": data.smoking_status
        },
        "risk_assessment": {
            "score": score,
            "category": category,
            "top_drivers": top_drivers,
            "interpretation": f"Patient at {category.lower()} risk for ischemic stroke based on {', '.join(top_drivers)}."
        },
        "recommended_labs": labs,
        "clinical_concerns": [],
        "recommended_actions": []
    }
    
    # Clinical concerns
    if data.avg_glucose_level > 140:
        report["clinical_concerns"].append(f"Pre-diabetic range glucose ({data.avg_glucose_level}). Recommend HbA1c and fasting insulin.")
    if data.bmi > 30:
        report["clinical_concerns"].append(f"Obesity (BMI {data.bmi}). Weight loss of 5-10% reduces stroke risk 20%.")
    if data.smoking_status == 'smokes':
        report["clinical_concerns"].append("Active smoking — strongest modifiable risk factor. Urgent cessation referral.")
    if data.hypertension == 1:
        report["clinical_concerns"].append("Hypertension — target <130/80 per ACC/AHA guidelines.")
    if data.heart_disease == 1:
        report["clinical_concerns"].append("Known heart disease — cardiology follow-up recommended.")
    if data.age > 75:
        report["clinical_concerns"].append("Advanced age — comprehensive geriatric assessment recommended.")
    
    # Recommended actions
    if score > 70:
        report["recommended_actions"].append("Urgent: Comprehensive stroke prevention evaluation within 2 weeks.")
    if "avg_glucose_level" in top_drivers and data.avg_glucose_level > 140:
        report["recommended_actions"].append("Consider metformin if HbA1c > 6.5%. Lifestyle intervention mandatory.")
    if data.bmi > 30:
        report["recommended_actions"].append("Refer to registered dietitian for structured weight loss program.")
    if data.smoking_status == 'smokes':
        report["recommended_actions"].append("Prescribe varenicline or bupropion + behavioral counseling.")
    if data.hypertension == 1:
        report["recommended_actions"].append("Optimize antihypertensive regimen. Target home BP <130/80.")
    if data.heart_disease == 1:
        report["recommended_actions"].append("Cardiology referral for AFib screening and CHA2DS2-VASc score.")
    
    if not report["recommended_actions"]:
        report["recommended_actions"].append("Continue current management. Annual cardiovascular risk reassessment.")
    
    return report

# ============================================
# API ENDPOINTS
# ============================================
@app.post("/predict")
def predict_risk(data: HealthData):
    """Main prediction endpoint — returns risk + labs + plan."""
    
    score, category, top_drivers = get_risk_score(
        data.age, data.gender, data.hypertension, data.heart_disease,
        data.ever_married, data.work_type, data.Residence_type,
        data.avg_glucose_level, data.bmi, data.smoking_status
    )
    
    labs = get_lab_suggestions(score, category, top_drivers, data)
    plan = get_prevention_plan(score, category, top_drivers, data)
    
    return {
        "risk_score": score,
        "risk_category": category,
        "top_drivers": top_drivers,
        "lab_suggestions": labs,
        "prevention_plan": plan,
        "message": f"Your stroke risk is {score}/100 ({category}). {len(labs)} proactive lab tests recommended."
    }

@app.post("/labs")
def get_labs_only(data: HealthData):
    """Get lab suggestions only."""
    
    score, category, top_drivers = get_risk_score(
        data.age, data.gender, data.hypertension, data.heart_disease,
        data.ever_married, data.work_type, data.Residence_type,
        data.avg_glucose_level, data.bmi, data.smoking_status
    )
    
    labs = get_lab_suggestions(score, category, top_drivers, data)
    
    return {
        "risk_score": score,
        "risk_category": category,
        "lab_suggestions": labs
    }

@app.post("/plan")
def get_plan_only(data: HealthData):
    """Get prevention plan only."""
    
    score, category, top_drivers = get_risk_score(
        data.age, data.gender, data.hypertension, data.heart_disease,
        data.ever_married, data.work_type, data.Residence_type,
        data.avg_glucose_level, data.bmi, data.smoking_status
    )
    
    plan = get_prevention_plan(score, category, top_drivers, data)
    
    return {
        "risk_score": score,
        "risk_category": category,
        "prevention_plan": plan
    }

@app.post("/report")
def get_doctor_report(data: HealthData):
    """Generate doctor-ready clinical report."""
    
    score, category, top_drivers = get_risk_score(
        data.age, data.gender, data.hypertension, data.heart_disease,
        data.ever_married, data.work_type, data.Residence_type,
        data.avg_glucose_level, data.bmi, data.smoking_status
    )
    
    labs = get_lab_suggestions(score, category, top_drivers, data)
    report = generate_doctor_report(score, category, top_drivers, data, labs)
    
    return {
        "clinical_report": report,
        "generated_at": "StrokeShield AI v1.0",
        "disclaimer": "This report is for informational purposes and does not replace professional medical advice."
    }

@app.get("/")
def health_check():
    """Check if API is running."""
    return {
        "status": "StrokeShield AI is live",
        "version": "1.0",
        "endpoints": ["/predict", "/labs", "/plan", "/report"],
        "description": "Proactive stroke prevention: risk prediction, lab suggestions, prevention plans, doctor reports"
    }

@app.get("/test")
def test_prediction():
    """Test endpoint with sample high-risk user."""
    test_data = HealthData(
        age=52, gender='Male', hypertension=1, heart_disease=0,
        ever_married='Yes', work_type='Private', Residence_type='Urban',
        avg_glucose_level=142, bmi=31, smoking_status='smokes'
    )
    return predict_risk(test_data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
