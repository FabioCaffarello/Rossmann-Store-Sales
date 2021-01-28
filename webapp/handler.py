import os
import pickle
import pandas as pd
from flask import Flask, request, Response
from rossmann.Rossmann import Rossmann



# Loding Model
model = pickle.load(open('model/1_flc_xgb_tuned.pkl', 'rb'))

# Initialize API
app = Flask(__name__)

@app.route('/rossmann/predict', methods=['POST'])
def rossmanPredict():
    test_JSON = request.get_json()
    
    if test_JSON: #there is data
        if isinstance(test_JSON, dict):
            teste_raw = pd.DataFrame(test_JSON, index=[0]) #unique example
        else:
            teste_raw = pd.DataFrame(test_JSON, columns=test_JSON[0].keys()) #multiple examples
    
        # Instantiate
        pipeline = Rossmann()
        
        # Data Cleaning
        df1 = pipeline.data_cleaning(teste_raw)
        # Feature Engineering
        df2 = pipeline.feature_engineering(df1)
        # Data Preparation
        df3 = pipeline.data_preparation(df2)
        # Prediction
        df_response = pipeline.get_prediction(model, teste_raw, df3)
        
        return df_response
    
    else:
        return Response('{}', status=200, mimetype='application/json')

if __name__ == '__main__':
    port = os.environ.get( 'PORT', 5000)
    app.run(host='0.0.0.0', port=port)