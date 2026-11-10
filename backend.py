from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import joblib
import uvicorn
import math

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

features = joblib.load('models/features.pkl')

class PredictRequest(BaseModel):
    models: List[str]
    inputs: Dict[str, float]

@app.post('/predict')
async def predict(req: PredictRequest):
    try:
        input_data = {f: 0.0 for f in features}
        for k, v in req.inputs.items():
            if k in input_data:
                input_data[k] = v

        Monthly_Income = input_data.get('Monthly_Income_Est_Rs', 25000)
        Monthly_Savings = input_data.get('Monthly_Savings_Rs', 7000)
        Repayment_OnTime = req.inputs.get('Repayment_OnTime_Pct', 90.0)
        Consistency_Score = input_data.get('Consistency_Score', 0.92)
        Social_Trust_Score = input_data.get('Social_Trust_Score', 0.92)
        UPI_Txn = input_data.get('UPI_Txn_Per_Day', 100)
        Days_Active = input_data.get('Days_Active_Per_Week', 5)
        Years_Vending = input_data.get('Years_Vending', 2)

        X_df = pd.DataFrame([[Monthly_Income, Monthly_Savings, Repayment_OnTime, Consistency_Score, Social_Trust_Score, UPI_Txn, Days_Active, Years_Vending]], 
                            columns=['Income', 'Savings', 'Repay', 'Cons', 'Trust', 'UPI_Day', 'Act_Days', 'Years'])

        models = {
            'Random Forest': joblib.load('models/rf_reg.pkl'),
            'AdaBoost': joblib.load('models/ada_reg.pkl'),
            'XGBoost': joblib.load('models/xgb_reg.pkl'),
            'LightGBM': joblib.load('models/lgbm_reg.pkl'),
            'Gradient Boosting': joblib.load('models/gb_reg.pkl')
        }

        results = {}
        scores = []
        for name, model in models.items():
            pred = int(model.predict(X_df)[0])
            results[name] = {'score': pred}
            scores.append(pred)

        ensemble_score = int(np.mean(scores))
        
        # V2.5 FLUID MULTI-TIER UNDERWRITING
        # Behavioral Additive/Subtractive Adjustments
        adjustment = 0
        if Repayment_OnTime < 50.0 or Monthly_Income < 5000:
            adjustment -= 250
        elif 50.0 <= Repayment_OnTime < 65.0:
            adjustment -= 120
            
        if Repayment_OnTime >= 90.0:
            adjustment += 40
            
        savings_ratio = (Monthly_Savings / Monthly_Income) if Monthly_Income > 0 else 0
        if savings_ratio >= 0.20:
            adjustment += 30

        # Hard Boundary Clamping
        ensemble_score = max(300, min(850, ensemble_score + adjustment))
        for name in results:
            results[name]['score'] = max(300, min(850, results[name]['score'] + adjustment))

        # Four-Tier Classification & Decision Matrix
        if ensemble_score < 500 or Repayment_OnTime < 50.0:
            risk = "High Risk"
            decision = "Declined"
            cls_name = "declined"
            icon = "ti-circle-x"
            summaryClass = "danger"
            summaryIcon = "\u274c"
            summaryTitle = "High Default Risk"
            limit_val = 0
            trancheNote = ""
            note = f"Critical default risk identified. Repayment punctuality is at {Repayment_OnTime}%, falling short of the 50% safety minimum. With an income throughput of \u20b9{Monthly_Income:,.0f} and savings buffer of \u20b9{Monthly_Savings:,.0f}, the profile is disqualified from PM SVANidhi progression."

        elif ensemble_score <= 649:
            risk = "Moderate Risk"
            decision = "Under Review"
            cls_name = "review"
            icon = "ti-alert-circle"
            summaryClass = "warn"
            summaryIcon = "\u26a0\ufe0f"
            summaryTitle = "Manual Verification Required"
            limit_val = 10000
            trancheNote = "Reviewing for initial PM SVANidhi 1st tranche (\u20b910k) eligibility."
            note = f"Operational friction requiring manual underwriting verification. Profile holds a consistency index of {Consistency_Score} and savings-to-income ratio of {savings_ratio*100:.1f}%. Must be assessed before tranche progression."

        elif ensemble_score <= 759:
            risk = "Moderate-Low Risk"
            decision = "Reliable"
            cls_name = "approved"
            icon = "ti-circle-check"
            summaryClass = ""
            summaryIcon = "\u2705"
            summaryTitle = "Reliable Mid-Tier Vendor"
            limit_val = 20000
            trancheNote = "To borrow the 2nd tranche (\u20b920k), the 1st tranche must be completely paid off."
            note = f"Strong mid-tier performance demonstrating a consistent digital footprint ({UPI_Txn} daily UPI txns) and dependable savings rates. Profile securely supports steady micro-credit expansion."

        else:
            risk = "Low Risk"
            decision = "Accepted"
            cls_name = "approved"
            icon = "ti-circle-check"
            summaryClass = ""
            summaryIcon = "\u2705"
            summaryTitle = "Micro-Loan Approved"
            if Monthly_Income >= 30000 and Repayment_OnTime >= 92.0:
                limit_val = 50000
                trancheNote = "To borrow the 3rd tranche (\u20b950k), the 1st & 2nd tranches must be completely paid off."
            else:
                limit_val = 20000
                trancheNote = "To borrow the 2nd tranche (\u20b920k), the 1st tranche must be completely paid off."
            note = f"Prime-tier vendor status. Exceptional financial discipline with \u20b9{Monthly_Income:,.0f} income throughput, robust savings of \u20b9{Monthly_Savings:,.0f}, and {Repayment_OnTime}% repayment punctuality."

        limitStr = f"\u20b9{int(limit_val):,}" if limit_val > 0 else "\u20b90"

        feature_importance_dict = {
            'Repayment_OnTime_Pct': 25.0,
            'Monthly_Savings_Rs': 20.0,
            'Monthly_Income_Est_Rs': 15.0,
            'Consistency_Score': 15.0,
            'Social_Trust_Score': 10.0,
            'UPI_Txn_Per_Day': 10.0,
            'Years_Vending': 3.0,
            'Days_Active_Per_Week': 2.0
        }
        sorted_fi = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))

        return {
            'status': 'success',
            'ensemble_score': ensemble_score,
            'verdict': {
                'cls': cls_name,
                'decision': decision,
                'icon': icon,
                'limitStr': limitStr,
                'risk': risk,
                'summaryClass': summaryClass,
                'summaryIcon': summaryIcon,
                'summaryTitle': summaryTitle,
                'summaryText': note,
                'trancheNote': trancheNote
            },
            'model_scores': results,
            'feature_importances': sorted_fi
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'message': str(e)}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
