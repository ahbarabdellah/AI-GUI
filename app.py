from fileinput import filename
import sys 
import os
sys.path.append(os.path.abspath("./static/Scripts"))
from regressionmodels import *
import classificationmodels 
from flask import Flask, redirect, url_for, render_template,request 
from datetime import datetime
import pandas as pd
from werkzeug.utils import secure_filename 
from time import time, sleep
app = Flask(__name__)


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
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(os.path.join(uploads_dir, secure_filename(uploaded_file.filename)))
    path="instance/uploads/"+str(uploaded_file.filename)
    df = pd.read_csv(path)
    shape=df.shape
    myfeatures=df.columns
    filename=uploaded_file.filename
    return render_template("choseparams.html", mytext=shape, features=myfeatures, filename=filename)


@app.route("/build",methods=['POST'])
def build():
    uploaded_file = request.form['filename']
    path="instance/uploads/"+uploaded_file
    df = pd.read_csv(path)
    features=df.columns
    target=request.form['target']
    model=request.form['model']
    if model=='GBRegressor':
        scores, model = GBRegressor(df,features.values,str(target))
    if model=='RFClassifier':
        scores, model = classificationmodels.RFClassifier(df,features.values,target)
    accuracy, r2, score2, score3=scores
    return render_template("choseparams.html", accuracy=accuracy, r2=r2, score2=score2, score3=score3)


@app.route("/About")
def About():
    return render_template("About.html")


if __name__ == "__main__":
    app.run(debug=True)