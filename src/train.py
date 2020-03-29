import os
import pandas as pd 
from sklearn import preprocessing
from sklearn import ensemble 
#we get those values from the environment variable
TRAININGDATA = os.environ.get("TRAININGDATA")
FOLD = int(os.environ.get("FOLD")) 

FOLD_MAPPING={
    0 : [1,2,3,4],
    1 : [0,1,3,4],
    2 : [0,1,2,4],
    3 : [0,1,2,4],
    4 : [0,1,2,3]
}

if __name__=="__main__":
    df = pd.read_csv(TRAININGDATA)
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
    label_encoders = []
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[c].values.to_list() + validation_df[c].values.to_list())
        #for all the rows for the column "c"
        train_df.loc[: , c] = lbl.transform(train_df[c].values.to_list())
        validation_df.loc[: , c] = lbl.transform(validation_df[c].values.to_list())
        label_encoders.append((c,lbl))

    #training the data
    clf = ensemble.RandomForestClassifer(n_jobs=-1, verbose = 2)
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(validation_df)[:, 1]
    print("The validation data shape: {}".format(validation_df.shape))
    print("The preds shape: {}".format(preds.shape))   
    print(preds) 