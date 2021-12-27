import pickle as pkl
import numpy as np
class WineProcessing(object ):
    def __init__(self):
        self.Imputer = pkl.load(open("./Imputer.pkl","rb"))
        self.MinMaxScaler = pkl.load(open("./MinMaxScaler.pkl","rb"))
    
    def pre_processing(self,df):
        columns = df.columns
        df[columns] = self.Imputer.transform(df[columns])
        df[columns] = self.MinMaxScaler.transform(df[columns])
        df = np.log(df+1)
        
        return df 
