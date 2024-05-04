import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression


t_data = pd.read_csv("C:\\Users\\DELL\\OneDrive\\Desktop\\data set\\tested.csv")
t_data
t_data.head()
t_data.tail()
t_data.shape
t_data.info()
t_data.isnull().sum()


#we need to handle missing values
# to handle missing values in the cabin column we will drop it as they are numerous
t_data.drop(columns='Cabin', axis=1, inplace=True)
# to handle missing values in the age and fare column we will replace them with the mean age and fare
Age = t_data['Age'].mean()
t_data['Age'].fillna(Age, inplace = True)
Fare = t_data['Fare'].mean()
t_data['Fare'].fillna(Fare, inplace = True)
t_data.info()

# Our data is now consistent
t_data

#Data Visualization
sns.set()
sns.countplot(x='Sex', data=t_data)

t_data['Survived'].value_counts()
sns.countplot(x='Survived', data=t_data)
sns.countplot (x='Sex', hue = 'Survived', data = t_data)

#this is clearly visible that those who survived were only females
t_data[['Survived', 'Sex']]
t_data[['Survived', 'Pclass' ]]

sns.countplot(x ='Pclass',hue= 'Survived', data=t_data)

#converting the categorical variables into numerical data
t_data.replace({'Sex':{'male':0, 'female':1},'Embarked':{'S':0, 'C':1, 'Q':2}}, inplace=True)
t_data

# Now drop the columns which are irrelevant for the survival prediction, such as PassengerId, Name and Ticket
t_data.drop(columns={'PassengerId','Name','Ticket'},axis=1, inplace=True)
t_data

#separating features and target
X = t_data.drop(columns='Survived', axis=1)
Y = t_data['Survived']
print(X)
print(Y)

#splitting the data into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
X_train.isnull().sum()

#Model Training
# we're using logistic regression model that uses binary classification for the prediction
model = LogisticRegression()
#training the model with the training data
model.fit(X_train,Y_train)

#Model Evaluation

X_test_prediction = model.predict(X_test)
# accuracy score for training data
# accuracy score for test data
print(X_test_prediction)

testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of test data is : ',testing_data_accuracy)
# precision score
test_data_precision = precision_score(Y_test, X_test_prediction)
print('test data precion is :', test_data_precision)
# recall score
test_data_recall = recall_score(Y_train, X_train_prediction)
print('test data recall is :', test_data_recall)

from sklearn import metrics
score = model.score(X_test,Y_test)
print(score)
cm = metrics.confusion_matrix(Y_train, X_train_prediction)
print(cm)

sns.heatmap(cm, annot = True, fmt = "d", square = True, cmap= "inferno")
plt.ylabel('Actual label')
plt.xlabel('predicted label')
title = ('Accuracy Score :',score)
plt.title(title, size = 10)

classification_report(X_test_pred, Y_test)

#Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
model_2 = RandomForestClassifier(n_estimators=100)
model_2.fit(X_train,Y_train)

X_test_pred = model_2.predict(X_test)
model_2.score(X_train, Y_train)
acc_score = round(model_2.score(X_test, Y_test) * 100, 2)
model_2_acc = accuracy_score(X_test_pred, Y_test)
model_2_acc


acc_score
precision = precision_score(X_test_pred, Y_test)
recall = recall_score(X_test_pred, Y_test)
print(precision)
print(recall)

classification_report(X_test_pred, Y_test)

#KNN
from sklearn.neighbors import KNeighborsClassifier
model_3 = KNeighborsClassifier(n_neighbors=3)
model_3.fit(X_train, Y_train)

X_test_pred = model_3.predict(X_test)
model_3_acc = accuracy_score(X_test_pred, Y_test, normalize=True)
model_3_score = round(model_3.score(X_train, Y_train) * 100, 2)
model_3_precision = precision_score(X_test_pred, Y_test)
model_3_recall = recall_score(X_test_pred, Y_test)
model_3_score

model_3_acc
print(model_3_precision)
print(model_3_recall)
classification_report(X_test_pred, Y_test)

#Predicting values
print(X) 
print(Y)

p1 = model.predict([[3, 0, 34.5, 0, 0, 7.8292, 2]])
p2 = model_2.predict([[3, 0, 34.5, 0, 0, 7.8292, 2]])
p3 = model_3.predict([[3, 0, 34.5, 0, 0, 7.8292, 2]])

print(p1)
print(p2)
print(p3)

p1 = model.predict([[2, 0, 38.5, 0, 0, 7.2500, 0]])
p2 = model_2.predict([[2, 0, 38.5, 0, 0, 7.2500, 0]])
p3 = model_3.predict([[2, 0, 38.5, 0, 0, 7.2500, 0]])

print(p1)
print(p2)
print(p3)




 


