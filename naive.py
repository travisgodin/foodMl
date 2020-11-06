# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 20:29:29 2020

@author: Travis
"""
import pandas as pd
import numpy as np
food_data = pd.read_csv('foodProd.csv')

food_data = food_data.dropna()

def choose_class(data):
    for x in data:
        if x > 0 and x < 2:
            data = data.replace(x,1)
        elif x >= 2 and x < 10:
            data = data.replace(x,2)
        elif x >=10:
            data = data.replace(x,3)
    return data


food_data['Greenhouse gas emissions per 1000kcal']= choose_class(food_data['Greenhouse gas emissions per 1000kcal'])



# Create training/testing datasets
from sklearn.model_selection import train_test_split
train, test = train_test_split(food_data, test_size=0.2)#, random_state=42)
train_labels = train['Greenhouse gas emissions per 1000kcal']
test_labels = test['Greenhouse gas emissions per 1000kcal']
train_data = train.iloc[:,3:17]
test_data = test.iloc[:,3:17]

# Graph Greenhouse emissions vs 
import matplotlib.pyplot as plt
landClass1 = test.loc[test['Greenhouse gas emissions per 1000kcal'] == 1].iloc[:,17]
landClass2 = test.loc[test['Greenhouse gas emissions per 1000kcal'] == 2].iloc[:,17]
landClass3 = test.loc[test['Greenhouse gas emissions per 1000kcal'] == 3].iloc[:,17]
farmClass1 = test.loc[test['Greenhouse gas emissions per 1000kcal'] == 1].iloc[:,3]
farmClass2 = test.loc[test['Greenhouse gas emissions per 1000kcal'] == 2].iloc[:,3]
farmClass3 = test.loc[test['Greenhouse gas emissions per 1000kcal'] == 3].iloc[:,3]

plt.scatter(landClass1, farmClass1, color='r', label='Class 1') #, marker='^', s = 70
plt.scatter(landClass2, farmClass2, color='b', label='Class 2') #, marker='v', s = 70
plt.scatter(landClass3, farmClass3, color='g', label='Class 3') #, marker='v', s = 70
plt.legend()
plt.xlabel("Land use per 1000kcal")
plt.ylabel("Farm Emissions")


# Fit a Gaussian Naive Bayes classifier and see its performance
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
pred = gnb.fit(train_data, train_labels).predict(test_data)
#print("Number of mislabeled points out of a total %d points: %d. Accuracy: %f" 
#     % (len(train_data),(train_labels != pred).sum(), 1-(train_labels != pred).sum()/len(train_data)))
print("Number of mislabeled points out of a total %d points: %d. Accuracy: %f" 
     % (len(test_data),(test_labels != pred).sum(), 1-(test_labels != pred).sum()/len(test_data)))

from sklearn.metrics import confusion_matrix
print('Confusion matrix (TN,FP/FN,TP):\n', confusion_matrix(test_labels, pred))
from sklearn.metrics import precision_score, recall_score
print('Precision:', precision_score(test_labels, pred,average='weighted',zero_division=0))
print('Recall:', recall_score(test_labels, pred,average='weighted',zero_division=0))
from sklearn.metrics import f1_score
print('F1 Score:', f1_score(test_labels, pred,average='weighted',zero_division=0))


xp = np.arange(-1, 31, 2)
yp = np.arange(-0, 31, 2)
zp = np.ndarray((16,16))

for x in range(0, len(xp)):
    for y in range(0, len(yp)):
        zp[x][y] = xp[x]**2 + yp[y]**2
        
contours=plt.contour(xp,yp,zp)

plt.show()
