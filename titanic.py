import numpy as np
import pandas as pd

# Importing the dataset 
original_dataset = pd.read_csv('train.csv')
dataset = pd.read_csv('train.csv')

# Looking at the complete nformation of the dataset
dataset.info()

# droping the unnecessary columns
dataset.pop('Name')
dataset.pop('Cabin')
dataset.pop('Ticket')
dataset.pop('PassengerId')

# filling the Nan values in the age column
from sklearn.preprocessing import Imputer
imp_train = Imputer(strategy="median")
dataset.iloc[:,4:5] = imp_train.fit_transform(dataset.iloc[:,4:5])

# filling the 2 Nan Value in embarked column
dataset.fillna(method = "ffill",inplace=True)

dataset.info()

# converting categorical variables
from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
dataset['Sex'] = lab.fit_transform(dataset['Sex'])
dataset['Embarked'] = lab.fit_transform(dataset['Embarked'])

dataset.info()

# creating correlation matrix 
tc = dataset.corr()

# visualizing the correlation between diffferent columns
import seaborn as sns
sns.heatmap(tc,annot=True)

# creating the X and th Y
dataset.columns
X = dataset[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
Y = dataset['Survived']

# creating one hot representation of embarked column
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[-1])
X = ohe.fit_transform(X).toarray()

# creating the test and train data
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size = 0.8,random_state = 42)


# Prediction
# Fitting the Classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(x_train,y_train)

# checking the predictions
print(round(classifier.score(x_test,y_test) * 100, 2))
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
print(confusion_matrix(y_test,classifier.predict(x_test)))
print(accuracy_score(y_test, classifier.predict(x_test)))
print(classification_report(y_test, classifier.predict(x_test)))

from sklearn.model_selection import cross_val_score
res = cross_val_score(classifier,x_test, y_test, cv=5, scoring='accuracy')
print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))
