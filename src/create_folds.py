import pandas as pd 
from sklearn import model_selection 


if __name__=="__main__":
    #reading the dataframe
    df = pd.read_csv("input/train.csv")
    #making a fake column k-fold to assign the fold number to it
    df['kfold'] = -1
    #shuffling the data and dropping the index
    df = df.sample(frac=1).reset_index(drop=True)
    #creating the kfolds
    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False, random_state=42)
    for fold , (train_idx, validation_idx) in enumerate(kf.split(X=df,y=df.target.values)):
        #set of fold and the idx in the train and validation in that fold
        print(len(train_idx), len(validation_idx))
        df.loc[validation_idx,'kfold'] = fold #setting the fold value to the fake column we created

    #saving the file
    df.to_csv("input/train_folds.csv",index=False)
    