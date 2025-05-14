# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: DHANUSHKUMAR SIVAKUMAR
RegisterNumber:  212224040067

```
```
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
data=pd.read_csv("Employee.csv")
data.head()
```
```
data.tail()
```
```
data.isnull().sum()
```
```
data.info()
```
```
data["left"].value_counts()
```
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
```
```
x=data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", "time_spend_company", "Work_accident", "promotion_last_5years","salary"]]
x
```
```
y=data["left"]
y
```
```
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=100)
```
```
dt = DecisionTreeClassifier(criterion='entropy', random_state=100)
dt.fit(x_train, y_train)
```
```
y_pred = dt.predict(x_test)
```
```
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy=accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100}%")
```
```
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree  
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```

## Output:

Head Values

![Screenshot 2025-05-14 211224](https://github.com/user-attachments/assets/5949d6b9-6609-448f-9a6e-ce43aae70898)

Head Values

![Screenshot 2025-05-14 211301](https://github.com/user-attachments/assets/e0779969-8ba7-4933-91c7-f6837d98fa0a)

Sum - Null Values

![Screenshot 2025-05-14 211309](https://github.com/user-attachments/assets/3ad5ddf7-e43e-48ef-a5f6-889b6635efaf)

Data Info

![Screenshot 2025-05-14 211318](https://github.com/user-attachments/assets/77277136-39a5-48d8-a869-20aa8b997c8e)

Values count in left column

![Screenshot 2025-05-14 211324](https://github.com/user-attachments/assets/480d3e1e-9e6c-4c0c-8162-ee397deef583)

X values

![Screenshot 2025-05-14 211337](https://github.com/user-attachments/assets/bf14d934-37a2-4ed5-8b76-4be6d0cf409b)

Y Values

![Screenshot 2025-05-14 211344](https://github.com/user-attachments/assets/5141434d-c33d-4a87-9c59-494674312f07)

Training the model

![Screenshot 2025-05-14 211354](https://github.com/user-attachments/assets/50227ecb-e188-43d6-a8e7-a8f87bc81a11)

Accuracy

![Screenshot 2025-05-14 211404](https://github.com/user-attachments/assets/cd8cf377-a002-4729-b416-c1396c167927)

Data Prediction

![Screenshot 2025-05-14 211416](https://github.com/user-attachments/assets/f7070bff-c434-491f-81d7-0a2b09eec449)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
