from fileinput import filename
from fnmatch import fnmatch
import sys 
import os
sys.path.append(os.path.abspath("./static/Scripts"))
from regressionmodels import *
from classificationmodels import *
from flask import Flask, redirect, url_for, render_template,request 
from datetime import datetime
import pandas as pd
from werkzeug.utils import secure_filename 
from time import time, sleep
app = Flask(__name__)

i=0


uploads_dir = os.path.join(app.instance_path, 'uploads')
os.makedirs(uploads_dir, exist_ok=True,)

def getfilename():
    while True:
        sleep(60 - time()%60)
        try:
            return request.files['file'].filename
            break
        except:
            return 'no file'
            continue
    
    
allmodels=['logisticregression','KNClassifier','DTClassifier','perceptron', 'scv', 'LinearSvc', 'RFClassifier', 'linearregression', 'DTRegressor', 'GBRegressor', 'KNRegressor', 'LassoRegressor', 'RFRegressor']


## to get current year
@app.context_processor
def inject_now():
    return {'now': datetime.utcnow()}

@app.errorhandler(413)
def too_large(e):
    return "File is too large", 413


@app.route("/")
def home():
    return render_template("index.html")
        


@app.route("/UnderstandML")
def UnderstandML():
    return render_template("UnderstandML.html")

@app.route("/choseparams",methods=['POST'])
def choseparams():
    global i ;
    uploaded_file = request.files['file']
    filename=str(i)+'.csv'
    if uploaded_file.filename =='':
        return render_template('error0.html')
    if uploaded_file.filename != '':
        uploaded_file.save(os.path.join(uploads_dir, secure_filename(filename)))
        i+=1
    path="instance/uploads/"+str(filename)
    df = pd.read_csv(path)
    shape=df.shape
    myfeatures=df.columns
    fname=uploaded_file.filename
    return render_template("choseparams.html",mymodels=allmodels, mytext=shape, features=myfeatures,filename0=fname, filename=path, counter=i)

@app.route("/build",methods=['POST'])
def build():
    path = request.form['filepath']
    print(path)
    df = pd.read_csv(path)
    features = request.form.getlist('features')
    if len(features)==0:
        return render_template('error1.html')
    
    target=request.form['target']
    modelname=request.form['model']
    if modelname=='logisticregression':
        scores, model = logisticregression(df,features,str(target))
    if modelname=='KNClassifier':
        scores, model = KNClassifier(df,features,target)
    if modelname=='DTClassifier':
        scores, model = DTClassifier(df,features,str(target))
    if modelname=='perceptron':
        scores, model = perceptron(df,features,str(target))
    if modelname=='scv':
        scores, model = scv(df,features,str(target))
    if modelname=='LinearSvc':
        scores, model = LinearSvc(df,features,str(target))
    if modelname=='RFClassifier':
        scores, model = RFClassifier(df,features,str(target))
    if modelname=='linearregression':
        scores, model = linearregression(df,features,str(target))
    if modelname=='DTRegressor':
        scores, model = DTRegressor(df,features,str(target))
    if modelname=='GBRegressor':
        scores, model = GBRegressor(df,features,str(target))
    if modelname=='KNRegressor':
        scores, model = KNRegressor(df,features,str(target))
    if modelname=='LassoRegressor':
        scores, model = LassoRegressor(df,features,str(target))
    if modelname=='RFRegressor':
        scores, model = RFRegressor(df,features,str(target))

    accuracy, r2, score2, score3=scores
    return render_template("choseparams.html", model=model, accuracy=accuracy, r2=r2, score2=score2, score3=score3,target=target,modelname=modelname)


@app.route("/About")
def About():
    return render_template("About.html")


if __name__ == "__main__":
    app.run(debug=True)