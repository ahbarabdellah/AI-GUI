#### ------ pour importet les fichier python en dehor de ce dossier  ------- ####
import sys 
import os
sys.path.append(os.path.abspath("./2. Notebook to script/1- regression"))
sys.path.append(os.path.abspath("./2. Notebook to script/2- classification"))
import regressionmodels
import classificationmodels

 
#################################
#### regression
path="./3. Test Data/Regression/House prices/House prices.csv"
features=['MSSubClass','MSZoning','LotFrontage','LotArea','Street','Neighborhood','HouseStyle','YearBuilt','RoofStyle','Foundation','BedroomAbvGr','Functional','Fireplaces','GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond','PoolArea','PoolQC']
target='SalePrice'
accuracy, r2, score2, score3, model = regressionmodels.RFRegressor(path,features,target)
print('Randomgorest :\t\t\t',r2)


#################################
#### classification
path = "./3. Test Data/Classification/winequality-red.csv"
features = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
target   = 'quality'
model,x_train,x_test,y_train,y_test = classificationmodels.rfc(path,features,target)
print('\n\ Random forest Classifier : ',model.score(x_test,y_test)*100)