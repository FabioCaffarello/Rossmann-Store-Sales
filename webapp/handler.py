from flask import Flask, request, Response
import os
import pandas as pd
import pickle
from rossmann.Rossmann import Rossmann

# Loding Model
model = pickle.load(open('D:/01-DataScience/04-Projetos/00-Git/Rossmann-Store-Sales/02-Notebooks/01-FirstRoundCRISP/model/modelRossmann.pkl', 'rb'))

# Initialize API
app = Flask(__name__)

@app.route('/rossmann/predict', methods=['POST'])
def rossmanPredict():
    testJSON = request.get_json()
    
    if testJSON: #there is data
        if isinstance(testJSON, dict):
            testeRaw = pd.DataFrame(testJSON, index=[0]) #unique example
        else:
            testeRaw = pd.DataFrame(testJSON, columns=testJSON[0].keys()) #multiple examples
    
        # Instantiate
        pipeline = Rossmann()
        
        # Data Cleaning
        df1 = pipeline.dataCleaning(testeRaw)
        # Feature Engineering
        df2 = pipeline.featureEngineering(df1)
        # Data Preparation
        df3 = pipeline.dataPreparation(df2)
        # Prediction
        dfResponse = pipeline.getPrediction(model, testeRaw, df3)
        
        return dfResponse
    
    else:
        return Response('{}', status=200, mimetype='application/json')

if __name__ == '__main__':
    port = os.environ.get( 'PORT', 5000)
    app.run(host='localhost', port=port, debug=True)