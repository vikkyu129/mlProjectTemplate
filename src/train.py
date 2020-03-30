import os
import pandas as pd 
from sklearn import preprocessing
from sklearn import metrics
from sklearn import ensemble 
from . import dispatcher
#to save
import joblib
#we get those values from the environment variable
TRAININGDATA = os.environ.get("TRAININGDATA")
TESTDATA = os.environ.get("TESTDATA")

FOLD = int(os.environ.get("FOLD")) 
MODEL = os.environ.get("MODEL")

FOLD_MAPPING={
    0 : [1,2,3,4],
    1 : [0,2,3,4],
    2 : [0,1,3,4],
    3 : [0,1,2,4],
    4 : [0,1,2,3]
}

if __name__=="__main__":
    df = pd.read_csv(TRAININGDATA)
    test_df = pd.read_csv(TESTDATA)
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]  #here getting the data of the other folds
    validation_df = df[df.kfold==FOLD]

    ytrain = train_df.target.values
    yvalidation = validation_df.target.values
    #dropping the unnecessary values
    train_df = train_df.drop(["id", "target", "kfold"],axis = 1)
    validation_df = validation_df.drop(["id", "target", "kfold"],axis = 1)
    #to make sure the order of variables is the same
    validation_df = validation_df[train_df.columns]
    #label encoding 
    label_encoders = {}
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[c].values.tolist() + validation_df[c].values.tolist() + test_df[c].values.tolist())
        #for all the rows for the column "c"
        train_df.loc[: , c] = lbl.transform(train_df[c].values.tolist())
        validation_df.loc[: , c] = lbl.transform(validation_df[c].values.tolist())
        label_encoders[c] = lbl

    #training the data
    clf = dispatcher.MODELS[MODEL] 
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(validation_df)[:,1]    
    print(preds)
    print(metrics.roc_auc_score(yvalidation, preds))
    joblib.dump(label_encoders, f"models/{MODEL}_{FOLD}_label_encoder.pkl")
    joblib.dump(clf,f"models/{MODEL}_{FOLD}.pkl")
    joblib.dump(train_df.columns,f"models/{MODEL}_{FOLD}_columns.pkl")

    