import os
import pandas as pd 
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn import ensemble 
from . import dispatcher
#to save
import joblib
#we get those values from the environment variable
TESTDATA = os.environ.get("TESTDATA")
MODEL = os.environ.get("MODEL")
FOLD = int(os.environ.get("FOLD")) 

def predict():
    df = pd.read_csv(TESTDATA)
    test_idx = df["id"].values
    predictions = None        
    for FOLD in range(5):
        df = pd.read_csv(TESTDATA)
        cols = joblib.load(os.path.join("models",f"{MODEL}_{FOLD}_columns.pkl"))    
        encoders = joblib.load(os.path.join("models",f"{MODEL}_{FOLD}_label_encoder.pkl"))
        for c in cols:
            lbl = encoders[c]                        
            df.loc[: , c] = lbl.transform(df[c].values.tolist())
        #predict           
        clf = joblib.load(os.path.join("models",f"{MODEL}_{FOLD}.pkl"))
        df =  df[cols]
        preds = clf.predict_proba(df)[:, 1]
        if FOLD == 0:
            predictions = preds
        else:
            predictions = predictions + preds
    predictions /= 5
    
    sub = pd.DataFrame(np.column_stack((test_idx,predictions)),columns=["id", "target"])
    return sub 

if __name__ =="__main__":
    submissions = predict()
    submissions.id= submissions.id.astype(int)
    submissions.to_csv(f"models/{MODEL}.csv",index=False)




    
        