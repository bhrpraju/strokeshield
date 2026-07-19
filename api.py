from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict
import joblib
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="StrokeShield AI - Proactive Prevention API v2.0")

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
class CoreHealthData(BaseModel):
    """Tier 1: Core medical data that feeds the ML model"""
    age: float
    gender: str
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: Optional[float] = None
    glucose_known: bool = True
    height_cm: float
    weight_kg: float
    smoking_status: str
    cigarettes_per_day: Optional[int] = 0

class RiskModifiers(BaseModel):
    """Tier 2: Evidence-based risk modifiers"""
    hookah_frequency: Optional[str] = "never"
    chewing_tobacco: Optional[str] = "never"
    beedi_smoking: Optional[str] = "never"
    vaping: Optional[str] = "never"
    recreational_drugs: Optional[bool] = False
    sleep_hours: Optional[str] = "7-8"
    sleep_regularity: Optional[str] = "regular"
    loud_snoring: Optional[bool] = False
    witnessed_apneas: Optional[bool] = False
    daytime_sleepiness: Optional[bool] = False
    late_night_eating: Optional[bool] = False
    meal_regularity: Optional[str] = "regular"
    processed_food: Optional[str] = "rarely"
    fruit_veg_servings: Optional[str] = "5+"
    salt_intake: Optional[str] = "moderate"
    cooking_oil: Optional[str] = "mustard"
    physical_activity: Optional[str] = "moderate"
    prolonged_sitting: Optional[bool] = False
    stress_level: Optional[str] = "low"
    anxiety_depression: Optional[bool] = False
    work_life_balance: Optional[str] = "balanced"
    family_stroke_history: Optional[str] = "none"
    family_diabetes_history: Optional[bool] = False
    previous_stroke_tia: Optional[bool] = False
    air_quality: Optional[str] = "good"
    cooking_fuel: Optional[str] = "lpg"

class LabValues(BaseModel):
    """Tier 3: Lab results for feedback loop"""
    hba1c: Optional[float] = None
    fasting_glucose: Optional[float] = None
    ldl: Optional[float] = None
    hdl: Optional[float] = None
    triglycerides: Optional[float] = None
    total_cholesterol: Optional[float] = None
    apob: Optional[float] = None
    crp: Optional[float] = None
    homocysteine: Optional[float] = None
    fasting_insulin: Optional[float] = None
    nt_probnp: Optional[float] = None
    ahi: Optional[float] = None

class FullAssessmentRequest(BaseModel):
    core: CoreHealthData
    modifiers: Optional[RiskModifiers] = None
    labs: Optional[LabValues] = None
# ============================================
# CORE PREDICTION FUNCTION (ML Layer)
# ============================================
def calculate_bmi(height_cm, weight_kg):
    """Calculate BMI from height and weight"""
    height_m = height_cm / 100
    return weight_kg / (height_m ** 2)

def get_ml_risk_score(core_data):
    """Predict stroke risk from core health data using trained ML model."""
    
    bmi = calculate_bmi(core_data.height_cm, core_data.weight_kg)
    
    # Handle unknown glucose
    if not core_data.glucose_known or core_data.avg_glucose_level is None:
        estimated_glucose = 90 + (bmi - 22) * 2 + max(0, core_data.age - 40) * 0.5
        if core_data.hypertension == 1:
            estimated_glucose += 15
        glucose = min(estimated_glucose, 300)
    else:
        glucose = core_data.avg_glucose_level
    
    g = le_gender.transform([core_data.gender])[0]
    m = le_married.transform([core_data.ever_married])[0]
    w = le_work.transform([core_data.work_type])[0]
    r = le_residence.transform([core_data.Residence_type])[0]
    s = le_smoking.transform([core_data.smoking_status])[0]
    
    features = pd.DataFrame([{
        'gender': g, 'age': core_data.age, 'hypertension': core_data.hypertension,
        'heart_disease': core_data.heart_disease, 'ever_married': m,
        'work_type': w, 'Residence_type': r,
        'avg_glucose_level': glucose, 'bmi': bmi,
        'smoking_status': s
    }])
    
    scaled = scaler.transform(features)
    prob = model.predict_proba(scaled)[0][1]
    score = int(prob * 100)
    
    # Boost for risk factors
    if core_data.age > 60: score += 15
    if core_data.age > 75: score += 10
    if glucose > 140: score += 10
    if glucose > 200: score += 15
    if bmi > 30: score += 5
    if core_data.smoking_status == 'smokes': score += 10
    if core_data.hypertension == 1: score += 5
    if core_data.heart_disease == 1: score += 10
    
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
    
    return score, cat, top_drivers, bmi, glucose

# ============================================
# RISK MODIFIER ENGINE (Evidence-Based Layer)
# ============================================
def calculate_risk_modifiers(modifiers):
    """Calculate evidence-based risk modifiers with citations."""
    
    if modifiers is None:
        return [], 0
    
    modifier_list = []
    total_points = 0
    
    # Tobacco & Substances
    if modifiers.hookah_frequency == "daily":
        modifier_list.append({
            "factor": "Daily hookah use",
            "points": 12,
            "evidence": "OR 3.2 for ischemic stroke (Shiraz case-control study)",
            "citation": "PMID: 29105666",
            "action": "Quit hookah completely. One session = 100+ cigarettes in smoke volume."
        })
        total_points += 12
    elif modifiers.hookah_frequency == "weekly":
        modifier_list.append({
            "factor": "Weekly hookah use",
            "points": 8,
            "evidence": "OR 3.2 for ischemic stroke",
            "citation": "PMID: 29105666",
            "action": "Reduce to zero. Hookah causes arterial stiffness in 30 minutes."
        })
        total_points += 8
    
    if modifiers.chewing_tobacco == "daily":
        modifier_list.append({
            "factor": "Daily chewing tobacco (gutka/paan)",
            "points": 8,
            "evidence": "OR 2.3 for ischemic stroke (adjusted, Indian case-control)",
            "citation": "Agashe & Gawde, TISS Mumbai",
            "action": "Quit immediately. 68% of stroke patients in South Asia use smokeless tobacco."
        })
        total_points += 8
    
    if modifiers.beedi_smoking == "current":
        modifier_list.append({
            "factor": "Current beedi smoking",
            "points": 10,
            "evidence": "OR 2.39 for all stroke (INTERSTROKE, 32 countries)",
            "citation": "INTERSTROKE Study, Lancet 2016",
            "action": "Beedis are MORE harmful than filtered cigarettes. Quit with nicotine replacement."
        })
        total_points += 10
    
    if modifiers.vaping == "daily":
        modifier_list.append({
            "factor": "Daily vaping/e-cigarettes",
            "points": 5,
            "evidence": "Arterial stiffness, endothelial dysfunction (UCLA study)",
            "citation": "UCLA, American Journal of Cardiology",
            "action": "Not a safe alternative. Quit all nicotine products."
        })
        total_points += 5
    
    if modifiers.recreational_drugs:
        modifier_list.append({
            "factor": "Recreational drug use",
            "points": 15,
            "evidence": "Cocaine: OR 5.7; Methamphetamine: hemorrhagic stroke",
            "citation": "Multiple case-control studies",
            "action": "URGENT: Seek addiction counseling. Drugs cause acute vessel damage."
        })
        total_points += 15
    
    # Sleep & Circadian
    if modifiers.sleep_hours in ["<5", "5-6"]:
        modifier_list.append({
            "factor": "Short sleep duration",
            "points": 3,
            "evidence": "U-shaped risk curve. <6 hours = higher stroke risk",
            "citation": "UK Biobank, Taiwan cohort studies",
            "action": "Target 7-8 hours. Sleep <6 hours increases BP and inflammation."
        })
        total_points += 3
    elif modifiers.sleep_hours == ">9":
        modifier_list.append({
            "factor": "Long sleep duration",
            "points": 3,
            "evidence": ">9 hours associated with higher stroke risk",
            "citation": "Meta-analysis, Sleep Medicine",
            "action": "Long sleep may indicate underlying conditions. Get checked."
        })
        total_points += 3
    
    if modifiers.sleep_regularity == "varies_>2hrs":
        modifier_list.append({
            "factor": "Highly irregular sleep schedule",
            "points": 5,
            "evidence": "26% higher cardiovascular death, heart attack, stroke",
            "citation": "72,269 people, 8-year follow-up (2021)",
            "action": "Fix sleep schedule. Same bedtime/waketime daily, even weekends."
        })
        total_points += 5
    
    # Sleep Apnea Screening (STOP-BANG simplified)
    apnea_score = 0
    if modifiers.loud_snoring: apnea_score += 1
    if modifiers.witnessed_apneas: apnea_score += 2
    if modifiers.daytime_sleepiness: apnea_score += 1
    
    if apnea_score >= 3:
        modifier_list.append({
            "factor": "High sleep apnea risk (STOP-BANG)",
            "points": 8,
            "evidence": "60-70% of stroke patients have OSA. Independent predictor.",
            "citation": "AAN 2022 Review; Age+Witnessed Apneas 73% accuracy",
            "action": "Get sleep study (HSAT or polysomnography). CPAP reduces stroke recurrence."
        })
        total_points += 8
    
    if modifiers.late_night_eating:
        modifier_list.append({
            "factor": "Regular late-night eating (>10 PM)",
            "points": 3,
            "evidence": "Circadian disruption, glucose intolerance, metabolic syndrome",
            "citation": "Chronobiology research, metabolic studies",
            "action": "Last meal before 8 PM. Fast 12-14 hours overnight."
        })
        total_points += 3
    
    # Diet & Lifestyle
    if modifiers.meal_regularity == "skips_breakfast":
        modifier_list.append({
            "factor": "Regularly skips breakfast",
            "points": 3,
            "evidence": "Skipping breakfast associated with higher stroke risk",
            "citation": "Japanese cohort study, Stroke journal",
            "action": "Eat within 1 hour of waking. Stabilizes glucose and metabolism."
        })
        total_points += 3
    
    if modifiers.processed_food == "daily":
        modifier_list.append({
            "factor": "Daily processed/junk food",
            "points": 3,
            "evidence": "Arterial stiffness, inflammation, trans fat intake",
            "citation": "INTERSTROKE dietary analysis",
            "action": "Replace with whole foods. Eliminate trans fats completely."
        })
        total_points += 3
    
    if modifiers.fruit_veg_servings == "<3":
        modifier_list.append({
            "factor": "Low fruit/vegetable intake (<3 servings/day)",
            "points": 3,
            "evidence": "12% of stroke deaths in India attributable to low fruit intake",
            "citation": "GBD 2021 India",
            "action": "Target 5+ servings/day. Focus on leafy greens, berries, citrus."
        })
        total_points += 3
    
    if modifiers.salt_intake == "high":
        modifier_list.append({
            "factor": "High salt intake",
            "points": 3,
            "evidence": ">=4000 mg/day = HR 2.59 vs <=1500 mg/day (Northern Manhattan Study)",
            "citation": "NOMAS, Stroke 2012",
            "action": "Target <2300 mg/day. Eliminate pickles, processed snacks, outside food."
        })
        total_points += 3
    
    if modifiers.cooking_oil == "dalda":
        modifier_list.append({
            "factor": "Dalda/vanaspati (trans fat) cooking",
            "points": 5,
            "evidence": "Trans fats directly damage endothelium, raise LDL, lower HDL",
            "citation": "WHO trans fat guidelines",
            "action": "Switch to mustard oil, coconut oil, or olive oil immediately."
        })
        total_points += 5
    
    if modifiers.physical_activity == "sedentary":
        modifier_list.append({
            "factor": "Sedentary lifestyle",
            "points": 3,
            "evidence": "Physical inactivity = independent stroke risk factor",
            "citation": "Stroke Riskometer validation study",
            "action": "Start with 10-min walks after meals. Build to 150 min/week."
        })
        total_points += 3
    
    if modifiers.prolonged_sitting:
        modifier_list.append({
            "factor": "Prolonged sitting (>8 hrs/day)",
            "points": 3,
            "evidence": "Independent vascular risk even with exercise",
            "citation": "Sedentary behavior meta-analysis",
            "action": "Stand every 30 min. Walk 2 min every hour."
        })
        total_points += 3
    
    # Stress & Mental Health
    if modifiers.stress_level == "severe":
        modifier_list.append({
            "factor": "Severe chronic stress",
            "points": 5,
            "evidence": "Chronic stress = inflammation, BP spikes, cortisol dysregulation",
            "citation": "Psychosomatic Medicine, multiple cohorts",
            "action": "Seek mental health support. Try meditation, yoga, counseling."
        })
        total_points += 5
    elif modifiers.stress_level == "high":
        modifier_list.append({
            "factor": "High stress level",
            "points": 3,
            "evidence": "Linked to higher stroke risk through multiple pathways",
            "citation": "Stroke Riskometer beta factors",
            "action": "Stress management: breathing exercises, regular breaks, boundaries."
        })
        total_points += 3
    
    if modifiers.anxiety_depression:
        modifier_list.append({
            "factor": "Anxiety/depression symptoms",
            "points": 3,
            "evidence": "Mental health disorders increase stroke risk",
            "citation": "Meta-analysis, Neurology",
            "action": "Mental health is physical health. Seek professional support."
        })
        total_points += 3
    
    # Family & History
    if modifiers.family_stroke_history in ["both_parents", "sibling"]:
        modifier_list.append({
            "factor": "Strong family history of stroke",
            "points": 5,
            "evidence": "Genetic + shared environment = 1.5-2x risk",
            "citation": "Stroke Riskometer family history factor",
            "action": "Earlier screening, aggressive risk factor control."
        })
        total_points += 5
    elif modifiers.family_stroke_history == "one_parent":
        modifier_list.append({
            "factor": "Family history of stroke (one parent)",
            "points": 3,
            "evidence": "Moderate genetic risk",
            "citation": "Framingham family studies",
            "action": "Be extra vigilant with modifiable risk factors."
        })
        total_points += 3
    
    if modifiers.family_diabetes_history:
        modifier_list.append({
            "factor": "Family history of diabetes",
            "points": 3,
            "evidence": "Insulin resistance risk, shared dietary habits",
            "citation": "Diabetes genetics consortium",
            "action": "Monitor glucose closely. Prioritize metabolic health."
        })
        total_points += 3
    
    if modifiers.previous_stroke_tia:
        modifier_list.append({
            "factor": "Previous stroke or TIA",
            "points": 20,
            "evidence": "STRONGEST predictor of recurrence. 10-15% within 90 days.",
            "citation": "AHA/ASA Secondary Prevention Guidelines",
            "action": "URGENT: Strict secondary prevention. Cardiology follow-up essential."
        })
        total_points += 20
    
    # Environment
    if modifiers.air_quality in ["poor", "very_poor"]:
        modifier_list.append({
            "factor": "Poor air quality exposure",
            "points": 5,
            "evidence": "41% of stroke deaths in India attributable to air pollution",
            "citation": "GBD 2021 India",
            "action": "Use air purifier indoors. N95 mask outdoors. Monitor AQI daily."
        })
        total_points += 5
    
    if modifiers.cooking_fuel in ["wood", "dung", "coal"]:
        modifier_list.append({
            "factor": "Solid fuel cooking (wood/dung/coal)",
            "points": 5,
            "evidence": "22% of stroke deaths in India from household air pollution",
            "citation": "GBD 2021 India",
            "action": "Switch to LPG/biogas. Ensure ventilation. Use exhaust fan."
        })
        total_points += 5
    
    return modifier_list, min(total_points, 30)
# ============================================
# LAB INTERPRETATION ENGINE
# ============================================
def interpret_labs(labs, age=50):
    """Interpret lab values against evidence-based targets."""
    
    if labs is None:
        return [], [], []
    
    interpretations = []
    abnormal_flags = []
    doctor_triggers = []
    
    # HbA1c
    if labs.hba1c is not None:
        if labs.hba1c < 5.5:
            status = "optimal"
            action = "Maintain current habits."
        elif labs.hba1c < 6.0:
            status = "good"
            action = "Monitor annually."
        elif labs.hba1c < 6.5:
            status = "borderline"
            action = "Lifestyle intervention. Recheck in 3 months."
        elif labs.hba1c < 7.0:
            status = "high"
            action = "Doctor consultation. Medication review."
            doctor_triggers.append("HbA1c diabetic range")
        else:
            status = "critical"
            action = "URGENT: See doctor within 1 week. Poor glycemic control."
            doctor_triggers.append("HbA1c critically high")
        
        interpretations.append({
            "test": "HbA1c",
            "value": f"{labs.hba1c}%",
            "status": status,
            "target": "<6.5% for stroke prevention",
            "action": action,
            "evidence": "HbA1c >=6.1% predicts >2x stroke recurrence (Wu et al., 2013)"
        })
        
        if status in ["high", "critical"]:
            abnormal_flags.append("HbA1c")
    
    # Fasting Glucose
    if labs.fasting_glucose is not None:
        if labs.fasting_glucose < 100:
            status = "optimal"
            action = "Maintain."
        elif labs.fasting_glucose < 126:
            status = "borderline"
            action = "Pre-diabetic. Lifestyle changes. Recheck in 3 months."
        elif labs.fasting_glucose < 160:
            status = "high"
            action = "Diabetic range. Doctor consultation."
            doctor_triggers.append("Fasting glucose high")
        else:
            status = "critical"
            action = "URGENT: See doctor within 1 week."
            doctor_triggers.append("Fasting glucose critically high")
        
        interpretations.append({
            "test": "Fasting Glucose",
            "value": f"{labs.fasting_glucose} mg/dL",
            "status": status,
            "target": "<100 mg/dL",
            "action": action,
            "evidence": "Fasting glucose >126 = diabetic threshold (ADA guidelines)"
        })
        
        if status in ["high", "critical"]:
            abnormal_flags.append("Fasting Glucose")
    
    # LDL-C
    if labs.ldl is not None:
        if labs.ldl < 70:
            status = "optimal"
            action = "Excellent for high-risk patients."
        elif labs.ldl < 100:
            status = "good"
            action = "Acceptable. Monitor."
        elif labs.ldl < 130:
            status = "borderline"
            action = "Diet changes. Consider statin if other risks."
        else:
            status = "high"
            action = "Doctor consultation. Statin therapy likely needed."
            doctor_triggers.append("LDL-C high")
        
        interpretations.append({
            "test": "LDL Cholesterol",
            "value": f"{labs.ldl} mg/dL",
            "status": status,
            "target": "<100 mg/dL (<70 for high-risk)",
            "action": action,
            "evidence": "LDL-C >130 = high risk. Each 39 mg/dL reduction = 22% stroke reduction (CTT meta-analysis)"
        })
        
        if status == "high":
            abnormal_flags.append("LDL-C")
    
    # ApoB
    if labs.apob is not None:
        if labs.apob < 60:
            status = "optimal"
            action = "Excellent."
        elif labs.apob < 80:
            status = "good"
            action = "Acceptable."
        elif labs.apob < 100:
            status = "borderline"
            action = "Monitor. Consider lifestyle intensification."
        else:
            status = "high"
            action = "Doctor consultation. Aggressive lipid lowering."
            doctor_triggers.append("ApoB high")
        
        interpretations.append({
            "test": "ApoB",
            "value": f"{labs.apob} mg/dL",
            "status": status,
            "target": "<60 mg/dL (optimal for high-risk)",
            "action": action,
            "evidence": "ApoB is better predictor than LDL-C. Each particle = potential plaque (Attia, Outlive)"
        })
        
        if status == "high":
            abnormal_flags.append("ApoB")
    
    # hs-CRP
    if labs.crp is not None:
        if labs.crp < 1.0:
            status = "optimal"
            action = "Low inflammation. Maintain."
        elif labs.crp < 3.0:
            status = "average"
            action = "Moderate. Anti-inflammatory diet."
        elif labs.crp < 10.0:
            status = "high"
            action = "High inflammation. Doctor consultation."
            doctor_triggers.append("hs-CRP elevated")
        else:
            status = "critical"
            action = "URGENT: Rule out infection. See doctor."
            doctor_triggers.append("hs-CRP critically high")
        
        interpretations.append({
            "test": "hs-CRP",
            "value": f"{labs.crp} mg/L",
            "status": status,
            "target": "<1.0 mg/L (low risk)",
            "action": action,
            "evidence": "Independent stroke predictor. Even with normal cholesterol. (Ridker et al.)"
        })
        
        if status in ["high", "critical"]:
            abnormal_flags.append("hs-CRP")
    
    # Homocysteine
    if labs.homocysteine is not None:
        if labs.homocysteine < 10:
            status = "optimal"
            action = "Excellent."
        elif labs.homocysteine < 12:
            status = "borderline"
            action = "B-vitamin supplementation."
        else:
            status = "high"
            action = "Doctor consultation. MTHFR testing. Methylated B-complex."
            doctor_triggers.append("Homocysteine elevated")
        
        interpretations.append({
            "test": "Homocysteine",
            "value": f"{labs.homocysteine} umol/L",
            "status": status,
            "target": "<10 umol/L (optimal)",
            "action": action,
            "evidence": "40% of stroke patients have >12. MTHFR variant common in Indians. Fixable with B-vitamins."
        })
        
        if status == "high":
            abnormal_flags.append("Homocysteine")
    
    # HOMA-IR
    if labs.fasting_insulin is not None and labs.fasting_glucose is not None:
        homa_ir = (labs.fasting_glucose * labs.fasting_insulin) / 405
        
        if homa_ir < 2.0:
            status = "optimal"
            action = "Excellent insulin sensitivity."
        elif homa_ir < 2.5:
            status = "acceptable"
            action = "Monitor. Lifestyle changes."
        elif homa_ir < 3.5:
            status = "high"
            action = "Insulin resistant. Doctor consultation. Metformin discussion."
            doctor_triggers.append("HOMA-IR elevated")
        else:
            status = "critical"
            action = "URGENT: Severe insulin resistance. Endocrinology referral."
            doctor_triggers.append("HOMA-IR critically high")
        
        interpretations.append({
            "test": "HOMA-IR (Insulin Resistance)",
            "value": f"{homa_ir:.2f}",
            "status": status,
            "target": "<2.0 (ideal), <2.5 (acceptable)",
            "action": action,
            "evidence": "Insulin resistance precedes diabetes by 10-15 years. Catches risk before HbA1c rises. (Attia, Outlive)"
        })
        
        if status in ["high", "critical"]:
            abnormal_flags.append("HOMA-IR")
    
    # NT-proBNP
    if labs.nt_probnp is not None:
        threshold = 450 if age > 75 else 125
        
        if labs.nt_probnp < threshold:
            status = "optimal"
            action = "No heart strain detected."
        elif labs.nt_probnp < threshold * 2:
            status = "elevated"
            action = "Mild heart strain. Cardiology follow-up."
            doctor_triggers.append("NT-proBNP elevated")
        else:
            status = "high"
            action = "URGENT: Significant heart strain. Cardiology referral."
            doctor_triggers.append("NT-proBNP high")
        
        interpretations.append({
            "test": "NT-proBNP",
            "value": f"{labs.nt_probnp} pg/mL",
            "status": status,
            "target": f"<{threshold} pg/mL (age-adjusted)",
            "action": action,
            "evidence": "Detects early heart strain before symptoms. Predicts AFib (1 in 4 strokes). (ESC guidelines)"
        })
        
        if status in ["elevated", "high"]:
            abnormal_flags.append("NT-proBNP")
    
    # Sleep Apnea (AHI)
    if labs.ahi is not None:
        if labs.ahi < 5:
            status = "optimal"
            action = "No sleep apnea."
        elif labs.ahi < 15:
            status = "mild"
            action = "Mild OSA. Lifestyle changes, positional therapy."
        elif labs.ahi < 30:
            status = "moderate"
            action = "Moderate OSA. CPAP consultation."
            doctor_triggers.append("Moderate OSA")
        else:
            status = "severe"
            action = "URGENT: Severe OSA. CPAP treatment essential."
            doctor_triggers.append("Severe OSA")
        
        interpretations.append({
            "test": "Sleep Apnea (AHI)",
            "value": f"{labs.ahi} events/hour",
            "status": status,
            "target": "<5 events/hour",
            "action": action,
            "evidence": "Untreated OSA = 2-3x stroke risk. 60-70% of stroke patients have OSA. CPAP reduces recurrence."
        })
        
        if status in ["moderate", "severe"]:
            abnormal_flags.append("Sleep Apnea")
    
    return interpretations, abnormal_flags, doctor_triggers

# ============================================
# LAB SUGGESTION ENGINE
# ============================================
def get_lab_suggestions(ml_score, adjusted_score, category, top_drivers, core_data, modifiers):
    """Generate personalized lab test suggestions with evidence."""
    
    bmi = calculate_bmi(core_data.height_cm, core_data.weight_kg)
    labs = []
    
    # HbA1c
    if (not core_data.glucose_known or core_data.avg_glucose_level is None or 
        (core_data.avg_glucose_level and core_data.avg_glucose_level > 110) or 
        core_data.age > 45):
        priority = "HIGH" if (not core_data.glucose_known or 
                              (core_data.avg_glucose_level and core_data.avg_glucose_level > 140)) else "MEDIUM"
        labs.append({
            "test": "HbA1c",
            "full_name": "Hemoglobin A1c",
            "why": "Your glucose is unknown" if not core_data.glucose_known else f"Your glucose is {core_data.avg_glucose_level}. HbA1c shows 3-month average.",
            "target": "<5.7% normal, <6.5% pre-diabetic threshold",
            "priority": priority,
            "cost_inr": "300-500",
            "action_if_high": "Cut refined carbs, walk 30 min post-meal, consider metformin",
            "citation": "Wu et al., 2013; ADA guidelines"
        })
    
    # Fasting Insulin + HOMA-IR
    if (core_data.avg_glucose_level and core_data.avg_glucose_level > 100) or bmi > 28:
        labs.append({
            "test": "Fasting Insulin + HOMA-IR",
            "full_name": "Insulin Resistance Panel",
            "why": f"Your BMI is {bmi:.1f}. Insulin resistance precedes diabetes by 10-15 years.",
            "target": "HOMA-IR <2.0 (ideal), <2.5 (acceptable)",
            "priority": "HIGH" if bmi > 30 else "MEDIUM",
            "cost_inr": "800-1200",
            "action_if_high": "Time-restricted eating (16:8), eliminate liquid sugar, resistance training",
            "citation": "Attia, Outlive; Reaven, 1988"
        })
    
    # Lipid Panel + ApoB
    if adjusted_score > 30:
        labs.append({
            "test": "Lipid Panel + ApoB",
            "full_name": "Advanced Lipid Profile",
            "why": "Standard LDL is incomplete. ApoB counts actual atherogenic particles.",
            "target": "ApoB <60 mg/dL (optimal high-risk), <80 (acceptable)",
            "priority": "HIGH" if core_data.age > 60 or core_data.heart_disease == 1 else "MEDIUM",
            "cost_inr": "1500-2500",
            "action_if_high": "Eliminate trans fats, increase omega-3, consider statin if ApoB >100",
            "citation": "Attia, Outlive; CTT meta-analysis"
        })
    
    # hs-CRP
    if adjusted_score > 40 or bmi > 30 or core_data.smoking_status == 'smokes':
        labs.append({
            "test": "hs-CRP",
            "full_name": "High-Sensitivity C-Reactive Protein",
            "why": "Inflammation is silent driver of arterial damage. Independent stroke predictor.",
            "target": "<1.0 mg/L (low risk), <3.0 (average)",
            "priority": "MEDIUM",
            "cost_inr": "400-600",
            "action_if_high": "Eliminate seed oils, increase omega-3 (2g EPA/DHA daily), sleep 7-8 hours",
            "citation": "Ridker et al.; JUPITER trial"
        })
    
    # Homocysteine
    if core_data.age > 50 or adjusted_score > 50:
        labs.append({
            "test": "Homocysteine",
            "full_name": "Homocysteine Level",
            "why": "Elevated homocysteine damages blood vessel walls directly. 40% of stroke patients >12.",
            "target": "<10 umol/L (optimal), <12 (acceptable)",
            "priority": "MEDIUM",
            "cost_inr": "600-900",
            "action_if_high": "Methylated B-complex (B6, B9, B12), reduce alcohol, leafy greens",
            "citation": "Clarke et al.; MTHFR studies in Indian population"
        })
    
    # NT-proBNP
    if core_data.hypertension == 1 or core_data.age > 65 or core_data.heart_disease == 1:
        labs.append({
            "test": "NT-proBNP",
            "full_name": "N-terminal pro-B-type Natriuretic Peptide",
            "why": "Detects early heart strain before symptoms. Predicts AFib (1 in 4 strokes).",
            "target": "<125 pg/mL (age <75), <450 (age 75+)",
            "priority": "HIGH" if core_data.heart_disease == 1 else "MEDIUM",
            "cost_inr": "1200-1800",
            "action_if_high": "Cardiology referral, BP optimization, sodium restriction <2g/day",
            "citation": "ESC guidelines 2023; Januzzi et al."
        })
    
    # Sleep Apnea Screen
    if modifiers:
        apnea_score = 0
        if modifiers.loud_snoring: apnea_score += 1
        if modifiers.witnessed_apneas: apnea_score += 2
        if modifiers.daytime_sleepiness: apnea_score += 1
        if bmi > 30: apnea_score += 1
        if core_data.age > 50: apnea_score += 1
        
        if apnea_score >= 2 or bmi > 30 or core_data.age > 50:
            labs.append({
                "test": "Sleep Study (Polysomnography or HSAT)",
                "full_name": "Obstructive Sleep Apnea Assessment",
                "why": "Untreated OSA increases stroke risk 2-3x. Causes nocturnal hypertension and AFib.",
                "target": "AHI <5 events/hour (normal)",
                "priority": "HIGH" if apnea_score >= 3 else "MEDIUM",
                "cost_inr": "3000-8000 (home), 15000-25000 (lab)",
                "action_if_high": "CPAP if AHI >15, weight loss priority #1, positional therapy",
                "citation": "AAN 2022; 60-70% of stroke patients have OSA"
            })
    
    # Carotid Ultrasound
    if core_data.age > 60 or adjusted_score > 70:
        labs.append({
            "test": "Carotid Ultrasound",
            "full_name": "Carotid Artery Doppler",
            "why": "Direct visualization of plaque in neck arteries. Shows risk blood tests miss.",
            "target": "IMT <0.9mm, no plaque",
            "priority": "HIGH" if adjusted_score > 70 else "LOW",
            "cost_inr": "2000-3500",
            "action_if_high": "Aggressive lipid lowering, aspirin discussion, vascular surgery if >70% stenosis",
            "citation": "Attia, Outlive; NASCET trial"
        })
    
    # Hookah-specific: Liver Function Test
    if modifiers and modifiers.hookah_frequency in ["daily", "weekly"]:
        labs.append({
            "test": "Liver Function Test (LFT)",
            "full_name": "Liver Function Panel",
            "why": "Hookah smoke contains heavy metals and toxins. One session = 100+ cigarettes in smoke volume.",
            "target": "ALT <40 U/L, AST <40 U/L, GGT <55 U/L",
            "priority": "MEDIUM",
            "cost_inr": "600-1000",
            "action_if_high": "Quit hookah immediately. Hepatology referral if enzymes >2x normal.",
            "citation": "Shiraz case-control: OR 3.2 for stroke; UCLA arterial stiffness study"
        })
    
    # Sort by priority
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    labs.sort(key=lambda x: priority_order[x["priority"]])
    
    return labs
# ============================================
# PREVENTION PLAN GENERATOR
# ============================================
def get_prevention_plan(ml_score, adjusted_score, category, top_drivers, core_data, modifiers, lab_interpretations):
    """Generate personalized 30-day prevention plan."""
    
    bmi = calculate_bmi(core_data.height_cm, core_data.weight_kg)
    
    plan = {
        "overview": f"Your base stroke risk is {ml_score}/100. With lifestyle factors, your adjusted risk is {adjusted_score}/100 ({category}). Here's your personalized plan.",
        "weeks": []
    }
    
    # Identify top risk factors for targeting
    risk_factors = []
    if "age" in top_drivers and core_data.age > 60:
        risk_factors.append("age")
    if "avg_glucose_level" in top_drivers or (core_data.avg_glucose_level and core_data.avg_glucose_level > 140):
        risk_factors.append("glucose")
    if "bmi" in top_drivers or bmi > 30:
        risk_factors.append("bmi")
    if core_data.smoking_status == 'smokes':
        risk_factors.append("smoking")
    if modifiers:
        if modifiers.hookah_frequency in ["daily", "weekly"]:
            risk_factors.append("hookah")
        if modifiers.chewing_tobacco == "daily":
            risk_factors.append("chewing_tobacco")
        if modifiers.loud_snoring or modifiers.witnessed_apneas:
            risk_factors.append("sleep_apnea")
        if modifiers.stress_level in ["high", "severe"]:
            risk_factors.append("stress")
        if modifiers.late_night_eating:
            risk_factors.append("late_eating")
    
    # Week 1: Quick wins + urgent labs
    week1_actions = []
    week1_labs = []
    
    if "glucose" in risk_factors or not core_data.glucose_known:
        week1_actions.append("Eliminate all liquid sugar (soda, juice, sweetened coffee). Replace with water or black coffee.")
        week1_actions.append("Start 16:8 time-restricted eating. No calories between 8 PM and 12 PM.")
        week1_labs.append("HbA1c + Fasting Glucose")
    
    if "bmi" in risk_factors:
        week1_actions.append(f"Your BMI is {bmi:.1f}. Target 5% weight loss = {core_data.weight_kg * 0.05:.1f} kg. Start by eliminating one processed food item daily.")
        week1_labs.append("Fasting Insulin + HOMA-IR")
    
    if "smoking" in risk_factors:
        week1_actions.append("Set a quit date within 14 days. Download QuitGuide app. Tell 3 people your plan. Consider nicotine replacement.")
    
    if "hookah" in risk_factors:
        week1_actions.append("QUIT hookah completely — not reduce. One session = 100+ cigarettes. Dispose of all hookah equipment. Find alternative social activity.")
        week1_labs.append("Liver Function Test")
    
    if "chewing_tobacco" in risk_factors:
        week1_actions.append("Quit gutka/paan TODAY. 68% of stroke patients in South Asia use smokeless tobacco. Use nicotine gum if cravings hit.")
    
    if "sleep_apnea" in risk_factors:
        week1_actions.append("Sleep on your side. Elevate head of bed 30 degrees. No alcohol after 6 PM. Record snoring with Sleep Cycle app.")
        week1_labs.append("Sleep Study (HSAT)")
    
    if "late_eating" in risk_factors:
        week1_actions.append("Last meal before 8 PM. Fast 12-14 hours overnight. No food within 3 hours of bedtime.")
    
    if core_data.hypertension == 1:
        week1_actions.append("Buy home BP monitor. Take readings twice daily (morning/evening). Log in app. Target <130/80.")
    
    if not week1_actions:
        week1_actions.append("Maintain current healthy habits. Focus on prevention labs this week.")
    
    plan["weeks"].append({
        "week": 1,
        "theme": "Quick Wins & Baseline",
        "actions": week1_actions,
        "labs_to_get": week1_labs if week1_labs else ["HbA1c", "Lipid Panel"]
    })
    
    # Week 2: Build habits
    week2_actions = []
    week2_labs = []
    
    if "glucose" in risk_factors:
        week2_actions.append("Add 10-minute walk after every meal. Lowers post-meal glucose spikes by 30%.")
    
    if "bmi" in risk_factors:
        week2_actions.append("Replace one processed meal/day with whole foods (vegetables + protein + healthy fat).")
        week2_actions.append("Add resistance training 2x/week. Muscle is metabolic currency.")
    
    if "stress" in risk_factors:
        week2_actions.append("Start 10-minute daily meditation (Headspace/Calm). Box breathing: 4-4-4-4 pattern.")
    
    if core_data.hypertension == 1:
        week2_actions.append("Reduce sodium to <2300mg/day. No restaurant food this week. Cook at home with mustard/coconut oil.")
    
    if not week2_actions:
        week2_actions.append("Add 2 strength training sessions (20 min). Bodyweight squats, push-ups, planks.")
    
    plan["weeks"].append({
        "week": 2,
        "theme": "Build the Habit Stack",
        "actions": week2_actions,
        "labs_to_get": week2_labs if week2_labs else ["Lipid Panel + ApoB", "hs-CRP"]
    })
    
    # Week 3: Intensify
    week3_actions = []
    
    if core_data.age > 50:
        week3_actions.append("Increase to 150 min/week moderate exercise (brisk walking, swimming, cycling).")
    
    if "bmi" in risk_factors:
        week3_actions.append("Increase resistance training to 3x/week. Progressive overload.")
    
    if "glucose" in risk_factors:
        week3_actions.append("Eliminate all refined grains (white bread, white rice, pasta). Replace with quinoa, lentils, vegetables.")
    
    if "smoking" in risk_factors or "hookah" in risk_factors:
        week3_actions.append("If still using nicotine: Cut to 50% of baseline. Use 4 D's: Delay, Deep breathe, Drink water, Do something else.")
    
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
def generate_doctor_report(ml_score, adjusted_score, category, top_drivers, core_data, modifiers, lab_interpretations, doctor_triggers):
    """Generate clinical summary for physician review."""
    
    bmi = calculate_bmi(core_data.height_cm, core_data.weight_kg)
    
    report = {
        "patient_summary": {
            "age": core_data.age,
            "gender": core_data.gender,
            "bmi": round(bmi, 1),
            "height_cm": core_data.height_cm,
            "weight_kg": core_data.weight_kg,
            "glucose": core_data.avg_glucose_level if core_data.glucose_known else "Unknown (estimated)",
            "hypertension": "Yes" if core_data.hypertension == 1 else "No",
            "heart_disease": "Yes" if core_data.heart_disease == 1 else "No",
            "smoking": core_data.smoking_status,
            "cigarettes_per_day": core_data.cigarettes_per_day
        },
        "risk_assessment": {
            "ml_risk_score": ml_score,
            "adjusted_risk_score": adjusted_score,
            "risk_category": category,
            "top_drivers": top_drivers,
            "interpretation": f"Base ML risk: {ml_score}/100. Adjusted with lifestyle factors: {adjusted_score}/100 ({category})."
        },
        "risk_modifiers": [],
        "lab_results": [],
        "clinical_concerns": [],
        "recommended_actions": [],
        "urgent_flags": doctor_triggers
    }
    
    # Add risk modifiers
    if modifiers:
        mod_list, _ = calculate_risk_modifiers(modifiers)
        report["risk_modifiers"] = mod_list
    
    # Add lab interpretations
    if lab_interpretations:
        report["lab_results"] = lab_interpretations
    
    # Clinical concerns
    if not core_data.glucose_known:
        report["clinical_concerns"].append("Glucose level unknown — estimated from BMI/age. HbA1c strongly recommended.")
    elif core_data.avg_glucose_level and core_data.avg_glucose_level > 140:
        report["clinical_concerns"].append(f"Pre-diabetic/diabetic range glucose ({core_data.avg_glucose_level}). Recommend HbA1c and fasting insulin.")
    
    if bmi > 30:
        report["clinical_concerns"].append(f"Obesity (BMI {bmi:.1f}). Weight loss of 5-10% reduces stroke risk 20%.")
    
    if core_data.smoking_status == 'smokes':
        report["clinical_concerns"].append("Active smoking — strongest modifiable risk factor. Urgent cessation referral.")
    
    if core_data.hypertension == 1:
        report["clinical_concerns"].append("Hypertension — target <130/80 per ACC/AHA guidelines.")
    
    if core_data.heart_disease == 1:
        report["clinical_concerns"].append("Known heart disease — cardiology follow-up recommended.")
    
    if modifiers:
        if modifiers.hookah_frequency in ["daily", "weekly"]:
            report["clinical_concerns"].append(f"Hookah use ({modifiers.hookah_frequency}) — OR 3.2 for ischemic stroke. Arterial stiffness in 30 minutes.")
        if modifiers.chewing_tobacco == "daily":
            report["clinical_concerns"].append("Daily chewing tobacco — OR 2.3 for stroke. 68% of South Asian stroke patients use SLT.")
        if modifiers.previous_stroke_tia:
            report["clinical_concerns"].append("Previous stroke/TIA — 10-15% recurrence within 90 days. Strict secondary prevention essential.")
    
    # Recommended actions
    if adjusted_score > 70:
        report["recommended_actions"].append("URGENT: Comprehensive stroke prevention evaluation within 2 weeks.")
    
    if not core_data.glucose_known or (core_data.avg_glucose_level and core_data.avg_glucose_level > 140):
        report["recommended_actions"].append("Order HbA1c and fasting glucose. Consider metformin if HbA1c >6.5%.")
    
    if bmi > 30:
        report["recommended_actions"].append("Refer to registered dietitian for structured weight loss program. Target 5-10% loss.")
    
    if core_data.smoking_status == 'smokes':
        report["recommended_actions"].append("Prescribe varenicline or bupropion + behavioral counseling. Nicotine replacement therapy.")
    
    if core_data.hypertension == 1:
        report["recommended_actions"].append("Optimize antihypertensive regimen. Target home BP <130/80. Consider ACE-I + thiazide/CCB.")
    
    if core_data.heart_disease == 1:
        report["recommended_actions"].append("Cardiology referral for AFib screening and CHA2DS2-VASc score.")
    
    if modifiers:
        if modifiers.loud_snoring or modifiers.witnessed_apneas:
            report["recommended_actions"].append("Sleep study (polysomnography or HSAT). If OSA confirmed, CPAP trial.")
        if modifiers.hookah_frequency in ["daily", "weekly"]:
            report["recommended_actions"].append("Addiction counseling for hookah cessation. Liver function test.")
    
    if not report["recommended_actions"]:
        report["recommended_actions"].append("Continue current management. Annual cardiovascular risk reassessment.")
    
    return report

# ============================================
# API ENDPOINTS
# ============================================
@app.post("/assess")
def full_assessment(request: FullAssessmentRequest):
    """Complete assessment: ML score + modifiers + labs + plan + report."""
    
    # Layer 1: ML Risk Score
    ml_score, category, top_drivers, bmi, glucose = get_ml_risk_score(request.core)
    
    # Layer 2: Risk Modifiers
    modifiers_list, modifier_points = calculate_risk_modifiers(request.modifiers)
    adjusted_score = min(ml_score + modifier_points, 99)
    
    # Recalculate category if adjusted
    if adjusted_score < 30: adj_category = "Low"
    elif adjusted_score < 60: adj_category = "Moderate"
    elif adjusted_score < 80: adj_category = "High"
    else: adj_category = "Critical"
    
    # Layer 3: Lab Suggestions
    labs = get_lab_suggestions(ml_score, adjusted_score, adj_category, top_drivers, request.core, request.modifiers)
    
    # Layer 4: Lab Interpretation (if provided)
    lab_interpretations, abnormal_flags, doctor_triggers = interpret_labs(request.labs, request.core.age)
    
    # Layer 5: Prevention Plan
    plan = get_prevention_plan(ml_score, adjusted_score, adj_category, top_drivers, request.core, request.modifiers, lab_interpretations)
    
    # Layer 6: Doctor Report
    report = generate_doctor_report(ml_score, adjusted_score, adj_category, top_drivers, request.core, request.modifiers, lab_interpretations, doctor_triggers)
    
    return {
        "ml_risk_score": ml_score,
        "adjusted_risk_score": adjusted_score,
        "risk_category": adj_category,
        "top_drivers": top_drivers,
        "bmi": round(bmi, 1),
        "estimated_glucose": round(glucose, 1) if not request.core.glucose_known else None,
        "risk_modifiers": modifiers_list,
        "modifier_points": modifier_points,
        "lab_suggestions": labs,
        "lab_interpretations": lab_interpretations,
        "abnormal_flags": abnormal_flags,
        "doctor_triggers": doctor_triggers,
        "prevention_plan": plan,
        "doctor_report": report,
        "message": f"Base risk: {ml_score}/100. Adjusted: {adjusted_score}/100 ({adj_category}). {len(labs)} lab tests recommended."
    }

@app.post("/predict")
def predict_only(core_data: CoreHealthData):
    """Quick prediction with core data only."""
    ml_score, category, top_drivers, bmi, glucose = get_ml_risk_score(core_data)
    return {
        "ml_risk_score": ml_score,
        "risk_category": category,
        "top_drivers": top_drivers,
        "bmi": round(bmi, 1),
        "estimated_glucose": round(glucose, 1) if not core_data.glucose_known else None
    }

@app.post("/analyze-labs")
def analyze_labs_only(core_data: CoreHealthData, labs: LabValues):
    """Analyze uploaded lab results."""
    ml_score, category, top_drivers, bmi, glucose = get_ml_risk_score(core_data)
    interpretations, abnormal_flags, doctor_triggers = interpret_labs(labs, core_data.age)
    
    return {
        "ml_risk_score": ml_score,
        "lab_interpretations": interpretations,
        "abnormal_flags": abnormal_flags,
        "doctor_triggers": doctor_triggers,
        "recommend_doctor": len(doctor_triggers) > 0,
        "doctor_report_preview": generate_doctor_report(ml_score, ml_score, category, top_drivers, core_data, None, interpretations, doctor_triggers) if doctor_triggers else None
    }

@app.get("/")
def health_check():
    return {
        "status": "StrokeShield AI v2.0 is live",
        "version": "2.0",
        "layers": ["ML Risk Prediction", "Evidence-Based Modifiers", "Lab Suggestions", "Lab Interpretation", "Prevention Plan", "Doctor Report"],
        "endpoints": ["/assess", "/predict", "/analyze-labs"],
        "description": "Proactive stroke prevention with ML + evidence-based modifiers + lab feedback loop"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
