from flask import Flask, render_template, request
import pickle
import numpy as np
classifier=pickle.load(open('classifier.pkl','rb'))
sc=pickle.load(open('sc.pkl','rb'))
app = Flask(__name__) 
@app.route('/')
def home():
    return render_template('home.html')
@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    pred = classifier.predict(sc.transform(final_features))
    return render_template('result.html', prediction=pred)
if __name__=='__main__':
       app.run(host='0.0.0.0',port=2000,debug=True)