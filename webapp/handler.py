import pickle
import pandas as pd
from flask             import Flask, request, Response
import sys
from api.empresa.empresa import PredictEmprestimo
import os

# loading model
model = pickle.load( open( 'model/final_model.pkl', 'rb') )

# initialize API
app = Flask( __name__ )

@app.route( '/empresa/predict', methods=['POST'] )
def rossmann_predict():
    test_json = request.get_json()
   
    if test_json: # there is data
        if isinstance( test_json, dict ): # unique example
            test_raw = pd.DataFrame( test_json, index=[0] )
            
        else: # multiple example
            test_raw = pd.DataFrame( test_json, columns=test_json[0].keys() )
            
        # Instantiate Rossmann class
        pipeline = PredictEmprestimo()
        
        df_cleaning = pipeline.data_cleaning(test_raw)

        df_feature = pipeline.feature_engineering(df_cleaning)

        df_preparation = pipeline.data_preparation(df_feature)

        df_predict = pipeline.get_predictions(model, df_preparation, test_raw)
        
        return df_predict
        
        
    else:
        return Response( '{}', status=200, mimetype='application/json' )

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run( '0.0.0.0', port=port )