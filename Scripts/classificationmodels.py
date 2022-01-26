### --- 0. the utilities.py conatin the function preproccessAndSplit() and scores()
from utilities import *
import warnings

from imblearn.over_sampling import RandomOverSampler # need to be installed !pip install imblearn
from sklearn.linear_model import LogisticRegression,Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier 
warnings.filterwarnings('ignore')



#### ----  1. Logistique Regression :
def logisticregression(path,features,target):
    x_train,x_test,y_train,y_test = preproccessAndSplit(path,features,target)
    x_train,y_train = RandomOverSampler(random_state=123).fit_resample(x_train, y_train)
    model = LogisticRegression(random_state = 123,n_jobs = -1)
    model.fit(x_train, y_train)
    return scores(x_test,y_test,model ), model

#### ----  2. KNN (KNeighborsClassifier) :
def KNClassifier(path,features ,target   ):
    x_train,x_test,y_train,y_test = preproccessAndSplit(path,features,target)
    x_train,y_train = RandomOverSampler(random_state=123).fit_resample(x_train, y_train)
    model=KNeighborsClassifier(n_jobs=-1)
    model.fit(x_train,y_train)
    return scores(x_test,y_test,model ), model

#### ----  3. Decision Tree :
def DTClassifier(path,features,target):
    x_train,x_test,y_train,y_test = preproccessAndSplit(path,features,target)
    x_train,y_train = RandomOverSampler(random_state=123).fit_resample(x_train, y_train)
    model = DecisionTreeClassifier(random_state=123)
    model.fit(x_train,y_train)
    return scores(x_test,y_test,model ), model

#### ----  4. Perceptron :
def perceptron(path,features,target):
    x_train,x_test,y_train,y_test = preproccessAndSplit(path,features,target)
    x_train,y_train = RandomOverSampler(random_state=123).fit_resample(x_train, y_train)
    model = Perceptron(random_state=123)
    model.fit(x_train, y_train)
    return scores(x_test,y_test,model ), model

#### ----  5. SVC :
def scv(path,features,target):
    x_train,x_test,y_train,y_test = preproccessAndSplit(path,features,target)
    x_train,y_train = RandomOverSampler(random_state=123).fit_resample(x_train, y_train)
    model = SVC(random_state = 123)
    model.fit(x_train,y_train)
    return model, x_train,x_test,y_train,y_test

#### ----  6. Linear SVC:
def LinearSvc(path,features,target):
    x_train,x_test,y_train,y_test = preproccessAndSplit(path,features,target)
    x_train,y_train = RandomOverSampler(random_state=123).fit_resample(x_train, y_train)
    model=LinearSVC(random_state=123)
    model.fit(x_train,y_train)
    return scores(x_test,y_test,model ), model

#### ----  7. Random forest Classifier :
def RFClassifier(path , features, target):
    x_train,x_test,y_train,y_test = preproccessAndSplit(path,features,target)
    model = RandomForestClassifier(n_jobs = -1,random_state = 123)
    model.fit(x_train, y_train)
    return scores(x_test,y_test,model ), model