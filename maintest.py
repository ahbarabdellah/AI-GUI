#### ------ pour importer les fichier python en dehor de ce dossier  ------- ####
import sys 
import os
import pandas as pd
sys.path.append(os.path.abspath("./static/Scripts"))
import regressionmodels
import classificationmodels

 
#####################################################################################
#### regression
path="./static/Data/Regression/House prices/Houseprices.csv"
df=pd.read_csv(path)
features=['MSSubClass','MSZoning','LotFrontage','LotArea','Street','Neighborhood','HouseStyle','YearBuilt','RoofStyle','Foundation','BedroomAbvGr','Functional','Fireplaces','GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond','PoolArea','PoolQC']
target='SalePrice'

scores, model = regressionmodels.linearregression(df,features,target)
print('linear Regressor:\t\t',scores)
 ###################################################
 ##### saving the model with pickle ###############
# loading library
# import pickle
# # create an iterator object with write permission - model.pkl
# with open('instance/models/model_pkl', 'wb') as files:
#     pickle.dump(model, files)

#####################################################################################
#### classification
path = "./static/Data/Classification/winequality-red.csv"
df=pd.read_csv(path)
features = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
target   = 'quality'

scores, model = classificationmodels.RFClassifier(df,features,target)
print('\nRandom forest Classifier: \t\t',scores[0])

