import numpy as np
import pandas as pd
import seaborn as se
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,plot_confusion_matrix
warnings.filterwarnings('ignore')

file_path= ""
features=[]
target=''
df=pd.read_csv(file_path);
df.head()
X=df[features]
Y=df[target]

def NullClearner(df):
    if(isinstance(df, pd.Series) and (df.dtype in ["float64","int64"])):
        df.fillna(df.mean(),inplace=True)
        return df
    elif(isinstance(df, pd.Series)):
        df.fillna(df.mode()[0],inplace=True)
        return df
    else:return df


def EncodeX(df):
    return pd.get_dummies(df)



def EncodeY(df):
    if len(df.unique())<=2:
        return df
    else:
        un_EncodedT=np.sort(pd.unique(df), axis=-1, kind='mergesort')
        df=LabelEncoder().fit_transform(df)
        EncodedT=[xi for xi in range(len(un_EncodedT))]
        print("Encoded Target: {} to {}".format(un_EncodedT,EncodedT))
        return df


x=X.columns.to_list()
for i in x:
    X[i]=NullClearner(X[i])  
X=EncodeX(X)
Y=EncodeY(NullClearner(Y))
X.head()


f,ax = plt.subplots(figsize=(18, 18))
matrix = np.triu(X.corr())
se.heatmap(X.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax, mask=matrix)


plt.figure(figsize = (10,6))
se.countplot(Y)

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=123)


x_train,y_train = RandomOverSampler(random_state=123).fit_resample(x_train, y_train)

model = DecisionTreeClassifier(random_state=123)
model.fit(x_train,y_train)

print("Accuracy score {:.2f} %\n".format(model.score(x_test,y_test)*100))

plot_confusion_matrix(model,x_test,y_test,cmap=plt.cm.Blues)

print(classification_report(y_test,model.predict(x_test)))

plt.figure(figsize=(8,6))
n_features = len(X.columns)
plt.barh(range(n_features), model.feature_importances_, align='center')
plt.yticks(np.arange(n_features), X.columns)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.ylim(-1, n_features)


fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (3,3), dpi=400)
cls_target = [str(x) for x in pd.unique(y_train)]
cls_target.sort()
plot_tree(model,feature_names = X.columns, class_names=cls_target,filled = True)
fig.savefig('./tree.png')