#Import necessary librabries and load data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv("iphone_purchase_records.csv")

dataset.info()

dataset.drop('Gender', axis=1, inplace=True)

dataset.info()

X=dataset.drop('Purchase Iphone', axis=1).values
Y=dataset['Purchase Iphone'].values

#Exploring data
pd.plotting.scatter_matrix(dataset, c=Y, figsize=[8,8], s=150)

sns.displot(dataset, x='Salary', hue='Purchase Iphone')

#Split data into training and test set
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3, random_state=18, stratify=Y)

#Make predictions using KNN algorithm
accuracy=[]
for i in range(1,10):
  knn = KNeighborsClassifier(n_neighbors=i).fit(X_train,Y_train)
  Y_pred = knn.predict(X_test)
  accuracy.append(metrics.accuracy_score(Y_test, Y_pred))
plt.plot(range(1,10),accuracy, color='blue', linestyle='dashed', marker='*')
plt.xlabel('k value')
plt.ylabel('accuracy')

#Check Accuracy of the Predictions
knn=KNeighborsClassifier(n_neighbors=8).fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
metrics.accuracy_score(Y_test, Y_pred)
