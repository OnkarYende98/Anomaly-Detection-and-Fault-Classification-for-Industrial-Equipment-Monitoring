import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)
from sklearn.ensemble import IsolationForest


!pip install xgboost
from xgboost import XGBClassifier


df = pd.read_csv("equipment_anomaly_data.csv")

print("Dataset shape:", df.shape)
print(df.head())


le_equipment = LabelEncoder()
df['equipment'] = le_equipment.fit_transform(df['equipment'])

le_location = LabelEncoder()
df['location'] = le_location.fit_transform(df['location'])


X = df.drop('faulty', axis=1)
y = df['faulty']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

print("\n--- Fault Classification Report (XGBoost) ---")

print(classification_report(y_test, y_pred_xgb))



cm_xgb = confusion_matrix(y_test, y_pred_xgb)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=xgb.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - XGBoost")
plt.show()



y_prob_xgb = xgb.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob_xgb)
roc_auc = auc(fpr, tpr)



plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGBoost')
plt.legend(loc='lower right')
plt.show()



iso = IsolationForest(contamination=0.099, random_state=42)

anomaly_labels = iso.fit_predict(X_scaled)



df['anomaly'] = np.where(anomaly_labels == -1, 1, 0)

print("\nTotal anomalies detected:", df['anomaly'].sum())



plt.figure(figsize=(6,4))
sns.scatterplot(
    x=df['temperature'], y=df['vibration'],
    hue=df['anomaly'], style=df['faulty'], palette="coolwarm"
)
plt.title("Anomaly Detection (Isolation Forest)")
plt.show()



cm_iso = confusion_matrix(df['faulty'], df['anomaly'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm_iso, display_labels=[0,1])
disp.plot(cmap=plt.cm.Oranges)
plt.title("Confusion Matrix - Isolation Forest")
plt.show()



iso_scores = -iso.decision_function(X_scaled)
fpr_iso, tpr_iso, _ = roc_curve(df['faulty'], iso_scores)
roc_auc_iso = auc(fpr_iso, tpr_iso)



plt.figure()
plt.plot(fpr_iso, tpr_iso, color='red', lw=2, label=f'ROC curve (AUC = {roc_auc_iso:.3f})')
plt.plot([0,1], [0,1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Isolation Forest')
plt.legend(loc='lower right')
plt.show()



equip_fault_rate = df.groupby('equipment')['faulty'].mean()
equip_fault_rate.plot(kind='bar', title="Fault Rate by Equipment")
plt.ylabel("Fault Rate")
plt.show()



loc_fault_rate = df.groupby('location')['faulty'].mean()
loc_fault_rate.plot(kind='bar', title="Fault Rate by Location", color="orange")
plt.ylabel("Fault Rate")
plt.show()



corr = df.drop(['faulty','anomaly'], axis=1).corrwith(df['faulty'])
corr.plot(kind='bar', title="Correlation of Features with Faulty Label")
plt.show()



total_anomalies = (df['anomaly'] == 1).sum()
true_faulty_in_anomalies = df[(df['anomaly'] == 1) & (df['faulty'] == 1)].shape[0]
false_alarms = df[(df['anomaly'] == 1) & (df['faulty'] == 0)].shape[0]



print("\n--- Anomaly Detection Evaluation ---")
print(f"Total anomalies detected: {total_anomalies}")
print(f"True faulty among an omalies: {true_faulty_in_anomalies}")
print(f"False alarms among anomalies: {false_alarms}")



precision = precision_score(df['faulty'], df['anomaly'])
recall = recall_score(df['faulty'], df['anomaly'])
f1 = f1_score(df['faulty'], df['anomaly'])



print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-score: {f1:.3f}")
