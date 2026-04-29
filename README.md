# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and preprocess it by converting categorical values like placement status into numerical form
2. Select important features as input (X) and the placement status as output (y)
3. Train the Logistic Regression model using the dataset to learn patterns
4. Use the model to predict placement status and evaluate its performance using accuracy

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: JAYA SURYA R
RegisterNumber:  212225230114
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("untitled.csv")
data = data.drop("salary", axis=1)
data = pd.get_dummies(data, drop_first=True)

x = data.drop("status_Placed", axis=1)
y = data["status_Placed"]

model = LogisticRegression(max_iter=1000)
model.fit(x, y)
print("Accuracy:", model.score(x, y))

x1 = x.iloc[:, 0].values.reshape(-1, 1)

model.fit(x1, y)

plt.scatter(x1, y, label="Data")
x_val = np.linspace(x1.min(), x1.max(), 100)
y_val = model.predict_proba(x_val.reshape(-1, 1))[:, 1]  
plt.plot(x_val, y_val, color="red", label="Logistic curve")
plt.xlabel("Feature 1")
plt.ylabel("Probability of Placement")
plt.legend()
plt.show()

```

## Output:
<img width="806" height="605" alt="Screenshot 2026-04-29 111328" src="https://github.com/user-attachments/assets/ac820ec7-80e1-4bd7-a44b-b041a2ea3c8f" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
