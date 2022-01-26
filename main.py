#### ------ pour importer les fichier python en dehor de ce dossier  ------- ####
import sys 
import os
sys.path.append(os.path.abspath("SCRIPTS"))
import regressionmodels
import classificationmodels

 
#####################################################################################
#### regression
path="./TestData/Regression/House prices/House prices.csv"
features=['MSSubClass','MSZoning','LotFrontage','LotArea','Street','Neighborhood','HouseStyle','YearBuilt','RoofStyle','Foundation','BedroomAbvGr','Functional','Fireplaces','GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond','PoolArea','PoolQC']
target='SalePrice'

scores, model = regressionmodels.GBRegressor(path,features,target)
print('Random Forest Regressor:\t\t',scores[0])


#####################################################################################
#### classification
path = "./TestData/Classification/winequality-red.csv"
features = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
target   = 'quality'

scores, model = classificationmodels.RFClassifier(path,features,target)
print('\nRandom forest Classifier: \t\t',scores[0])