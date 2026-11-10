
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
import shap

os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

df = pd.read_excel('C:/Streetcred/street_vendor_credit_dataset.xlsx')

plt.figure(figsize=(10, 6))
sns.scatterplot(x='UPI_Txn_StdDev', y='Consistency_Score', hue='Repayment_Label', data=df)
plt.title('Revenue Volatility vs Consistency Score')
plt.savefig('plots/eda_volatility.png')
plt.close()

df['CoV_Income'] = df['UPI_Txn_StdDev'] / (df['UPI_Avg_Credit_Rs'] + 1e-5)

features = [
    'Years_Vending', 'Num_Dependents', 'UPI_Txn_Per_Day', 'UPI_Avg_Credit_Rs', 
    'UPI_Active_Days_Per_Month', 'UPI_Txn_StdDev', 'Monthly_UPI_Credit_Rs', 
    'Morning_Txn_Ratio', 'Evening_Txn_Ratio', 'Total_Daily_Sales_Rs', 
    'Days_Active_Per_Week', 'Monthly_Income_Est_Rs', 'Monthly_Expense_Rs', 
    'Monthly_Savings_Rs', 'Consistency_Score', 'Social_Trust_Score', 'CoV_Income'
]

X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'models/scaler.pkl')

label_map = {'Repaid': 1, 'Defaulted': 0, 'UNLABELED': -1}
df['y_semi'] = df['Repayment_Label'].map(label_map)
y = df['y_semi'].values

X_labeled = X_scaled[df['Is_Labeled'] == 1]
y_labeled = y[df['Is_Labeled'] == 1]
X_unlabeled = X_scaled[df['Is_Labeled'] == 0]

X_train, X_test, y_train, y_test = train_test_split(X_labeled, y_labeled, test_size=0.2, random_state=42)

X_semi_train = np.vstack([X_train, X_unlabeled])
y_semi_train = np.concatenate([y_train, np.full(X_unlabeled.shape[0], -1)])

base_clfs = {
    'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1),
    'LightGBM': LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
}

trained_semi_models = {}
for name, clf in base_clfs.items():
    semi_clf = SelfTrainingClassifier(clf, criterion='threshold', threshold=0.85)
    semi_clf.fit(X_semi_train, y_semi_train)
    trained_semi_models[name] = semi_clf

lgbm_pseudo = trained_semi_models['LightGBM'].predict(X_semi_train)
xgb_pseudo = trained_semi_models['XGBoost'].predict(X_semi_train)

lgbm_pseudo_labels = lgbm_pseudo[y_semi_train == -1]
xgb_pseudo_labels = xgb_pseudo[y_semi_train == -1]
agree_mask = lgbm_pseudo_labels == xgb_pseudo_labels

X_meta_train = X_unlabeled[agree_mask]
y_meta_train = lgbm_pseudo_labels[agree_mask]

meta_clf = LogisticRegression()
if len(np.unique(y_meta_train)) > 1:
    meta_clf.fit(X_meta_train, y_meta_train)
else:
    meta_clf.fit(X_train, y_train)

joblib.dump(trained_semi_models['XGBoost'], 'models/XGBoost.pkl')
joblib.dump(trained_semi_models['LightGBM'], 'models/LightGBM.pkl')
joblib.dump(trained_semi_models['RandomForest'], 'models/RandomForest.pkl')
joblib.dump(trained_semi_models['GradientBoosting'], 'models/GradientBoosting.pkl')
joblib.dump(meta_clf, 'models/MetaClassifier.pkl')
joblib.dump(features, 'models/features.pkl')

baseline_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
baseline_xgb.fit(X_train, y_train)
explainer = shap.TreeExplainer(baseline_xgb)
shap_values = explainer.shap_values(X_train)
plt.figure()
shap.summary_plot(shap_values, X_train, feature_names=features, show=False)
plt.savefig('plots/shap_summary.png', bbox_inches='tight')
