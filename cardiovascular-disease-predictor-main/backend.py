from flask import Flask,render_template,request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
@app.route('/')
def front_page():
    return render_template('frontpage.html')

@app.route('/heart',methods=['GET','POST'])
def heart_page():
    if request.method == 'GET':
        return render_template('heart.html')
    else:
        age = request.form['age']
        sex = request.form['sex']
        chest = request.form['chest']
        trestbps = request.form['trestbps']
        chol = request.form['chol']
        fbs = request.form['fbs']
        restecg = request.form['restecg']
        thalach = request.form['thalach']
        exang = request.form['exang']
        oldpeak = request.form['oldpeak']
        slope = request.form['slope']
        ca = request.form['ca']
        thal = request.form['thal']
        heart_dataset = pd.read_csv('C:/Users/Gurakonda Konda Redd/OneDrive/Desktop/All Foldders/cardiovascular_disease_predictor/cardiovascular-disease-predictor-main/heart.csv')
        X = heart_dataset.drop(columns='target', axis=1)
        Y = heart_dataset['target']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        model = KNeighborsClassifier(n_neighbors=21, metric='manhattan')
        model.fit(X_train, Y_train)
        input_data = (age,sex,chest,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)
        input_data_as_numpy_array= np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        prediction = model.predict(input_data_reshaped)
        senddata=""
        if (prediction[0]== 0):
            senddata='According to the given details person does not have Heart Disease'
        else:
            senddata='According to the given details chances of having Heart Disease are High, So Please Consult a Doctor'
        return render_template('result.html',resultvalue=senddata)
if __name__ == '__main__':
    app.run()
