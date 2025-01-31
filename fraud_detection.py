import kagglehub
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# Download latest version
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
#print("Path to dataset files:", path)
filepath = path + "/creditcard.csv"
#print(filepath)

data = pd.read_csv(filepath)
#print(data.head())
#print(data['Class'].value_counts()) #distribution imbalanced
'''
plt.figure(figsize=(10, 6))
sns.histplot(data[data['Class'] == 0]["Amount"], bins = 50, color='blue', label='Non-Fraud')
sns.histplot(data[data['Class'] == 1]["Amount"], bins = 50, color='red', label='Fraud')
plt.yscale('log')  # Apply logarithmic scale
plt.legend()
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency (Log Scale)")
plt.title("Distribution of Transaction Amounts for Fraud vs Non-Fraud")
plt.show()
'''
#normalise data
scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data['Time'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))
#print(data.head())
X = data.drop('Class', axis=1)
y = data['Class']
#print(y.head())
X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.2, random_state=69, stratify=y)
y_train = y_train.astype(int)
#Balance the data set to make sure it contains as many fraud data and nonfraud data 
smote = SMOTE(random_state = 69)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train) 
#print(pd.Series(y_train_res).value_counts())
#logistic_model = LogisticRegression(random_state=69)
#logistic_model.fit(X_train_res, y_train_res)

model = RandomForestClassifier(random_state=69, n_estimators=100)
model.fit(X_train_res, y_train_res)

#xgb_model = xgb.XGBClassifier(random_state = 69, use_label_encoder = False, eval_metric = 'logloss')
#xgb_model.fit(X_train_res, y_train_res)

y_pred = model.predict(X_test)
y_pred_prob = np.clip(model.predict_proba(X_test)[:, 1], 1e-6, 1)
print(f"This is the model outpu: {y_pred}")
print(f"This is y_pred_prod: {y_pred_prob}")

print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_pred_prob)

print(f'ROC-AUC score: {roc_auc:.4f}')

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

'''
Logistic regression: 
             precision    recall  f1-score   support

           0       1.00      0.97      0.99     56864
           1       0.06      0.95      0.11        98

    accuracy                           0.97     56962
   macro avg       0.53      0.96      0.55     56962
weighted avg       1.00      0.97      0.99     56962

ROC-AUC score: 0.9887

Random Forest:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.89      0.84      0.86        98

    accuracy                           1.00     56962
   macro avg       0.95      0.92      0.93     56962
weighted avg       1.00      1.00      1.00     56962

ROC-AUC score: 0.9819

the xgboosting model evaluation is left as an exercise 
'''