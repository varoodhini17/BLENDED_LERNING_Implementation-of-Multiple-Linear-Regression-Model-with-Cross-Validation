# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries and load the dataset using pandas.
2.Remove unnecessary columns (car_ID, CarName) and convert categorical variables into dummy variables.
3.Separate the dataset into independent variables (X) and dependent variable (price).
4.Split the data into training and testing sets (80% training, 20% testing).
5.Train a Linear Regression model using the training data.
6.Perform 5-fold cross-validation and calculate the average R² score.
7.Predict on the test set, evaluate using MSE, MAE, and R² score, and plot actual vs predicted prices. 

## Program:
```
/*
Program to implement the multiple linear regression model for predicting car prices with cross-validation.
Developed by: Varoodhini.M
RegisterNumber: 212225220118
*/
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import matplotlib.pyplot as plt
data=pd.read_csv('CarPrice_Assignment.csv')

data=data.drop(['car_ID','CarName'],axis=1)
data=pd.get_dummies(data,drop_first=True)
X=data.drop('price',axis=1)
y=data['price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
print('Name:')
print('Reg No:')
print("\n=== Cross-Validation ===")
cv_scores=cross_val_score(model,X,y,cv=5)
print("Fold R2 scores:",[f"{score:.4f}" for score in cv_scores])
print(f"Average R2: {cv_scores.mean():.4f}")
y_pred=model.predict(X_test)
print("\n=== Test Set Performance ===")
print(f"MSE: {mean_squared_error(y_test,y_pred):.2f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred):.2f}")
print(f"R2: {r2_score(y_test,y_pred):.4f}")
plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y_test.min(),y_test.max()],
        [y_test.min(),y_test.max()],'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()


```

## Output:

<img width="827" height="623" alt="image" src="https://github.com/user-attachments/assets/764f620d-f0ba-4808-9a77-dfb474fc22d7" />



## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
