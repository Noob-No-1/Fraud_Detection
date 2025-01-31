Performance of different models:

Logistic regression: 

             precision    recall  f1-score   support

           0       1.00      0.97      0.99     56864
           1       0.06      0.95      0.11        98

    accuracy                           0.97     56962
   macro avg       0.53      0.96      0.55     56962
weighted avg       1.00      0.97      0.99     56962

ROC-AUC score: 0.9887
![logostic_reg_fraud](https://github.com/user-attachments/assets/71ff824c-c9c6-439c-825a-62518cd3052f)





Random Forest:

              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.89      0.84      0.86        98

    accuracy                           1.00     56962
   macro avg       0.95      0.92      0.93     56962
weighted avg       1.00      1.00      1.00     56962

ROC-AUC score: 0.9819

the xgboosting model evaluation is left as an exercise 
![random_forest_fraud](https://github.com/user-attachments/assets/8c457245-92c5-424e-bf7d-37c1f3be9be2)
