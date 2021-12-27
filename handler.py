from flask import Flask, request
import pickle
import pandas as pd
from wine_processing import WineProcessing
Model = pickle.load(open('./Model.pkl','rb'))

app = Flask(__name__)
WineProcessing = WineProcessing()


@app.route('/')
def hello():
    return 'Ola'

@app.route("/predict", methods=['POST'])
def predict():
    test_json = request.get_json()
    
    #Coletar dados
    if test_json:
        if isinstance(test_json, dict): #Uma linha apenas
            df_raw = pd.DataFrame(test_json,index=[0])
        else:
            df_raw = pd.DataFrame(test_json, columns = test_json[0].keys())
    
    #Instanciando
    
    #Preparação
    df_raw = WineProcessing.pre_processing(df_raw)
    
    
    #Predição
    pred = Model.predict(df_raw)
    df_raw['prediction'] = pred
    
    return df_raw.to_json(orient='records')

if __name__ == '__main__':
    app.run(
        #host='0.0.0.0',port='5000'
        )