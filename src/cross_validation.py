import pandas as pd 
from sklearn import model_selection
"""
Hold-out based validation is dont when :
1. It is time series Data
2. The number of datapoints is in millions and we want to perfom Hold-Out Validaiton on a few samples
"""
class CrossValidation:
    def __init__(
        self,
        df,
        target_cols,
        problem_type = "binary_classification",
        multilabel_delimiter = ",",
        num_folds = 5,
        shuffle, 
        random_state = 42
        ):
        self.dataframe = df 
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.problem_type = problem_type
        self.num_folds = num_folds
        self.shuffle = shuffle
        self.random_state = random_state
        self.multilabel_delimiter = multilabel_delimiter

        if self.shuffle is True:
            self.dataframe = self.dataframe.sample(frac =1).reset_index(drop = True)

        self.dataframe["kfold"] = -1
    #look at the target columns and split the data accordingly 
    def split(self):
        if self.num_targets!=1:
            raise Exception("More than 1 target column!")
        if self.problem_type in ["binary_classification","multi_class_classification"]:
            unique = self.dataframe[self.target_cols[0]].nunique()
            if unique ==1:
                raise Exception("Only one unique target is present!")
            elif unique>1:
                target = self.target_cols[0]
                kf = model_selection.StratifiedKFold(n_splits = self.num_folds, shuffle = False)
                for fold, (train_idx, validation_idx) in enumerate(kf.split(X=self.dataframe,y=self.dataframe[target])):
                    print(len(train_idx))
                    print(len(validation_idx))
                    self.dataframe.loc[validation_idx, "kfold"] = fold
        elif self.problem_type in ["single_col_regression","multi_col_regression"]:
            if self.num_targets!=1 and self.problem_type=="single_col_regression":
                raise Exception("More than 1 target column!")
            if self.num_targets<2 and self.problem_type=="multi_col_regression":
                raise Exception("Less than 1  target column!")
            kf = model_selection.KFold(n_splits = self.num_folds)
            for fold, (train_idx, validation_idx) in enumerate(kf.split(X=self.dataframe)):
                self.dataframe.loc[validation_idx, "kfold"] = fold
        elif self.problem_type.startswith("holdout_"):
            holdout_percentage = int(self.problem_type.split("_")[1])
            num_holdout_samples = int(len(self.dataframe)*holdout_percentage/100)
            self.dataframe.loc[num_holdout_samples:] = 0 
            self.dataframe.loc[:num_holdout_samples] = 1 
        elif self.problem_type == "multi_label":
            if num_targets != 1:
                raise Exception("Invalid number of Targets!")
            #creating folds based on the counts of number of classes    
            target = self.dataframe[self.target_cols[0]].apply(lambda x :len(str(x).split(self.multilabel_delimiter)))
            kf = model_selection.StratifiedKFold(n_splits = num_folds)
            for fold , (train_idx, validation_idx) in enumerate(kf.split(X= self.dataframe, y= target)):
                self.dataframe.loc[validation_idx,"kfold"] = fold
        else:
            raise Exception("Problem Type not understood!")
        return self.dataframe

if __name__=="__main__":
    df= pd.read_csv("../input/train_reg.csv")
    cv = CrossValidation(df, target_cols = ["SalePrice"], problem_type = "holdout_10")
    df_split = cv.split()
    print(df_split.head())
    print(df_split.kfold.value_counts())


