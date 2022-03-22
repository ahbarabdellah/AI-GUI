from fileinput import filename
from fnmatch import fnmatch
import io
from flask import current_app, send_file, send_from_directory
import sys 
import os
import pickle
from static.Scripts.classificationmodels import *
from static.Scripts.regressionmodels import *
from static.Scripts.utilities import *
from static.Scripts.utilities import scores
sys.path.append(os.path.abspath("./static/Scripts"))

from flask import Flask, render_template,request 
from datetime import datetime
import pandas as pd
from werkzeug.utils import secure_filename 
app = Flask(__name__)


import seaborn as se
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np 


k=0


uploads_dir = os.path.join(app.instance_path, 'uploads')
os.makedirs(uploads_dir, exist_ok=True,)

downloads_dir = os.path.join(app.instance_path, 'models')
os.makedirs(downloads_dir, exist_ok=True,)


def plotprevstest(k,y_test,x_test, model):
    plt.plot(range(20),y_test[0:20], color = "green")
    plt.plot(range(20),model.predict(x_test[0:20]), color = "red")
    plt.legend(["Actual","prediction"]) 
    plt.title("Predicted vs True Value")
    imgurl ="static/images/"+str(k)+".png"
    plt.savefig(imgurl)
    plt.close()
    return imgurl
        
def redgreen(k,y_test,x_test, model):
    red = plt.scatter(np.arange(0,80,5),model.predict(x_test)[0:80:5],color = "red")
    green = plt.scatter(np.arange(0,80,5),y_test[0:80:5],color = "green")
    plt.title("Comparison of Regression Algorithms")
    plt.xlabel("Index of Candidate")
    plt.ylabel("target")
    plt.legend((red,green),('Model', 'REAL'))
    imgurl ="static/images/redgreen"+str(k)+".png"
    plt.savefig(imgurl)
    plt.close()
    return imgurl

def plotmatrix(df,k):
    # plot
    matrix = np.triu(df.corr())
    fig = se.heatmap(df.corr(), annot=False, linewidths=.1, mask=matrix)
    imgurl ="static/images/matrix"+str(k)+".png"
    fig = fig.get_figure()
    fig.savefig(imgurl, bbox_inches='tight')
    plt.close()
    return imgurl
    
allmodels=['logisticregression',
           'KNClassifier','DTClassifier','perceptron', 'scv', 'LinearSvc', 'RFClassifier', 'linearregression', 'DTRegressor', 'GBRegressor', 'KNRegressor', 'LassoRegressor', 'RFRegressor']


## to get current year
@app.context_processor
def inject_now():
    return {'now': datetime.utcnow()}

@app.errorhandler(413)
def too_large(e):
    return "File is too large", 413

@app.route('/download', methods=['GET', 'POST'])
def download():
    uploads = os.path.join(current_app.root_path, "instance/models")
    model_pkl='model'+str(k)+'.pkl'
    return send_from_directory(directory=downloads_dir, path=model_pkl, as_attachment=True)


@app.route("/")
def home():
    return render_template("index.html")
        

@app.route("/UnderstandML")
def UnderstandML():
    return render_template("UnderstandML.html")

@app.route("/choseparams",methods=['POST'])
def choseparams():
    global k ;
    uploaded_file = request.files['file']
    k+=1
    filename=str(k)+'.csv'
    if uploaded_file.filename =='':
        return render_template('error0.html')
    if uploaded_file.filename != '':
        uploaded_file.save(os.path.join(uploads_dir, secure_filename(filename)))
    path="instance/uploads/"+str(filename)
    df = pd.read_csv(path,sep=",")
    shape=df.shape
    myfeatures=df.columns
    fname=uploaded_file.filename
    matriximgurl = plotmatrix(df,k)
    return render_template("choseparams.html",mymodels=allmodels, mytext=shape, features=myfeatures,filename0=fname, filename=path,matriximgurl=matriximgurl,  counter=k)

@app.route("/build",methods=['POST'])
def build():
    path = request.form['filepath']
    df = pd.read_csv(path,sep=",")
    features = request.form.getlist('features')
    if len(features)==0:
        return render_template('error1.html')
    
    target=request.form['target']
    modelname=request.form['model']
    if modelname=='logisticregression':
       X, x_test, y_test, model = logisticregression(df,features,str(target))
    if modelname=='KNClassifier':
       X, x_test, y_test, model = KNClassifier(df,features,target)
    if modelname=='DTClassifier':
       X, x_test, y_test, model = DTClassifier(df,features,str(target))
    if modelname=='perceptron':
       X, x_test, y_test, model = perceptron(df,features,str(target))
    if modelname=='scv':
       X, x_test, y_test, model = scv(df,features,str(target))
    if modelname=='LinearSvc':
       X, x_test, y_test, model = LinearSvc(df,features,str(target))
    if modelname=='RFClassifier':
       X, x_test, y_test, model = RFClassifier(df,features,str(target))
    if modelname=='linearregression':
       X, x_test, y_test, model = linearregression(df,features,str(target))
    if modelname=='DTRegressor':
       X, x_test, y_test, model = DTRegressor(df,features,str(target))
    if modelname=='GBRegressor':
       X, x_test, y_test, model = GBRegressor(df,features,str(target))
    if modelname=='KNRegressor':
       X, x_test, y_test, model = KNRegressor(df,features,str(target))
    if modelname=='LassoRegressor':
       X, x_test, y_test, model = LassoRegressor(df,features,str(target))
    if modelname=='RFRegressor':
       X, x_test, y_test, model = RFRegressor(df,features,str(target))
    
    
    model_pkl='model'+str(k)+'.pkl'
    with open('instance/models/'+'model'+str(k)+'.pkl', 'wb') as files:
        pickle.dump(model, files)
        
    accuracy, r2, score2, score3 = scores(x_test, y_test, model)
    imgurl = plotprevstest(k,y_test,x_test, model)
    redgreenurl = redgreen(k,y_test,x_test, model)
    return render_template("choseparams.html", model_pkl=model_pkl, accuracy=accuracy, r2=r2, score2=score2, score3=score3,target=target,modelname=modelname,redgreen=redgreenurl,imgurl=imgurl)


@app.route("/About")
def About():
    return render_template("About.html")


if __name__ == "__main__":
    app.run(debug=True)