
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#Read the data
df = pd.read_csv("nasa.csv")

#Remove redundant/irrelevant data as much as possible
df = df.drop(['Neo Reference ID', 'Name', 'Absolute Magnitude', 'Est Dia in M(min)', 'Est Dia in M(max)', 'Est Dia in Miles(min)', 'Est Dia in Miles(max)', 'Est Dia in Feet(min)', 'Est Dia in Feet(max)', 'Close Approach Date', 'Relative Velocity km per hr', 'Miles per hour', 'Miss Dist.(lunar)', 'Miss Dist.(kilometers)', 'Miss Dist.(miles)', 'Orbiting Body', 'Equinox'], axis=1)

#Check correlation for remaining data and modify database
correlation = df.corr()
correlation = abs(correlation['Hazardous'])
strong_corr = correlation[correlation > 0.24]
df = df[strong_corr.index]

#Split the data to test
train, test, train_labels, test_labels = train_test_split(df, df['Hazardous'], test_size=0.25, random_state=0)  

#Train the model
model = svm.SVC()
model.fit(train, train_labels)

#Predict from the testing data
prediction = model.predict(test)
print('Accuracy = ' + str(accuracy_score(test_labels, prediction)))
