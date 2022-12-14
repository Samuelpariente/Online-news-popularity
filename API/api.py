# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 21:04:00 2022

@author: samue
"""

from flask import Flask, request  # import main Flask class and request object
import random
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



filename = 'model_api.sav'
loaded_model = pickle.load(open(filename, 'rb'))

def prediction(l):
    result = loaded_model.predict(l)
    return result 

def stringtodf(s):
    l = s.split('/')
    l = [float(z) for z in l]
    temp = pd.read_csv('news_api.csv')
    del temp['Class_shares1']
    del temp['Unnamed: 0']
    data = dict(zip(temp.columns,l))
    df = pd.DataFrame(columns = temp.columns)
    df.loc[0] = pd.Series(data)
    object = StandardScaler()
    object.fit(temp)
    df.loc[0] = object.transform(df)
    return df

app1 = Flask(__name__) 
@app1.route('/status', methods=['GET'])
def a_live():
    return "Alive!"

@app1.route('/predict', methods=['GET'])
def predict():
    args = request.args
    info = args['info']
    line = stringtodf(info)
    val = prediction(line)
    if val[0] == 1 :
        return 'high'
    if val[0] == 0 :
        return 'low'
    
app1.run(debug=True ,host='0.0.0.0', port=8080)