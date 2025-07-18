import gc
from datetime import datetime
import pickle
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
#import nxviz as nv
import missingno as msno
from itertools import combinations
from tabulate import tabulate
import os

from sklearn.pipeline import Pipeline,make_pipeline
from scipy.stats import iqr, shapiro,kstest
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score,roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression,Lasso, Ridge
from sklearn.neural_network import MLPRegressor,MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier,AdaBoostRegressor,GradientBoostingRegressor,GradientBoostingClassifier,HistGradientBoostingClassifier,HistGradientBoostingRegressor,RandomForestRegressor,RandomForestClassifier
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.metrics import confusion_matrix, recall_score,f1_score,accuracy_score,precision_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer

import xgboost as xgb
import lightgbm as lgb
#from catboost import CatBoostClassifier


import seaborn as sns
import statsmodels.api as sm
from statsmodels.api import qqplot
import numpy as np
import joblib



def generate_list_of_models(model_type:str):
    baseEstimatorClassifier=DecisionTreeClassifier()
    baseEstimatorRegressor=DecisionTreeRegressor()
    if model_type=="classification":
        modeles={
            "lightgbm":lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1),
            "xgboost":xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3),
            "adaboost":AdaBoostClassifier(estimator=baseEstimatorClassifier,n_estimators=100),
            "gradientboost":GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3),
            "histgboost":HistGradientBoostingClassifier(),
            "xgboost":xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3),
            "randomForest":RandomForestClassifier(n_estimators=100),

            "logreg":LogisticRegression(solver='lbfgs', max_iter=100),
            "knn":KNeighborsClassifier(n_neighbors=5),
            "tree":DecisionTreeClassifier(random_state=0),
            "mlp":MLPClassifier(hidden_layer_sizes=(100,), # Taille de la couche cachée
                                activation='relu',         # Fonction d'activation
                                solver='adam',             # Algorithme d'optimisation
                                max_iter=300,              # Nombre max d'itérations
                                random_state=42)

        }

        modeles_grid={
            "logreg":{
                'C': [0.1, 1, 10, 100],  # Force de régularisation (inverse de λ)
                'penalty': ['l1', 'l2', 'elasticnet', None],  # Types de régularisation
                'solver': ['lbfgs'],  # Algorithmes d'optimisation
                'l1_ratio': [0, 0.5, 1]  # Utilisé uniquement avec elasticnet},
            },
            "knn":{
                'n_neighbors': [3, 5, 7, 9],          # Nombre de voisins
                'weights': ['uniform', 'distance'],   # Poids uniformes ou inversement proportionnels à la distance
                'metric': ['euclidean', 'manhattan', 'minkowski'],  # Type de distance
                'p': [1, 2]  # Valeur de 'p' pour la distance Minkowski (1 = Manhattan, 2 = Euclidienne)},
            },
            "tree":{
                'criterion': ['gini', 'entropy'],       # Fonction pour mesurer la qualité du split
                'splitter': ['best', 'random'],         # Stratégie pour choisir la variable de split
                'max_depth': [3, 5, 10, None],          # Profondeur maximale de l'arbre
                'min_samples_split': [2, 5, 10],        # Nombre minimum d'échantillons pour diviser un nœud
                'min_samples_leaf': [1, 2, 4],          # Nombre minimum d'échantillons dans une feuille
                'max_features': [None, 'auto', 'sqrt', 'log2']  # Nombre maximum de caractéristiques pour les splits}
            },
            "mlp":{
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'solver': ['adam', 'lbfgs'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'max_iter': [200, 400, 600]}
        }
    else:
        modeles={
            #"linreg":LinearRegression(),
            "gradient_boost":GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3),
            "xgboost":xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3),
            "lgb_boost":lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1),
            "adaboost":AdaBoostRegressor(estimator=baseEstimatorRegressor,n_estimators=100),
            "histgboost":HistGradientBoostingRegressor(),
            "randomForest":RandomForestRegressor(n_estimators=100),
            "lasso":Lasso(alpha=0.1),
            "Ridge":Ridge(alpha=0.1),
            "knn":KNeighborsRegressor(n_neighbors=5),
            "tree":DecisionTreeRegressor(max_depth=5),
            "mlp":MLPRegressor(hidden_layer_sizes=(100,50), # Taille de la couche cachée
                               learning_rate='adaptive',         # Fonction d'activation
                               solver='adam',             # Algorithme d'optimisation
                               max_iter=300)


        }

    return modeles

def check_model_coherences(dataframe : pd.DataFrame, target : str, model_type: str):
    if (dataframe[target].nunique() == 2) & (model_type != "classification"):
        raise Exception ("Vérifier le type de modèle")
    else:
        print("Check Model OK")