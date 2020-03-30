#The dispatcher is to serve different models to the data 
from sklearn import ensemble
MODELS = {
    "randomforest" : ensemble.RandomForestClassifier(n_estimators=200,n_jobs=-1, verbose = 2),
    "randomforestDepth" : ensemble.RandomForestClassifier(n_estimators=500,n_jobs=-1, verbose = 2),
    "extratrees" : ensemble.ExtraTreesClassifier(n_estimators=200,n_jobs=-1, verbose = 2)
}