
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline,make_pipeline
from Projects.Functions.Data_cleaning.Data_cleaning import *
import joblib
from Projects.Functions.Read_config.read_config import *
from datetime import datetime

def build_the_final_pipeline(pipeline_best_model,features, targets, cible,pca=False, pca_number_of_composantes=5):

    liste_cat=dataset_categorical_values(features)
    liste_float_int=dataset_numerical_values(features)

    liste_cat = [val for val in liste_cat if val != cible]
    liste_float_int = [val for val in liste_float_int if val != cible]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False), liste_cat),
            ('num', StandardScaler(), liste_float_int)
        ]
    )

    if pca:
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),    # Prétraitement
            ('pca', PCA(n_components=pca_number_of_composantes)),      # Réduction de dimensionnalité
            ('classifier', pipeline_best_model)  # Modèle de classification
        ])

    else:
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),    # Prétraitement
            ('classifier', pipeline_best_model)  # Modèle de classification
        ])

    pipeline.fit(features,targets)

    return pipeline

def update_config(config_path,config_api,model_file,path_model):
    update_config(config_path,"model_file",model_file)
    update_config(config_api,"model_file",model_file)
    update_config(config_path,"api_model_path",path_model)
    update_config(config_api,"api_model_path","./"+path_model)
    update_config(config_path,"time_stamp_model",datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    update_config(config_api,"time_stamp_model",datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def saving_of_best_model_for_production(model_path,model_path_api, pipeline,best):
    model_file = best + "_pipeline_model.joblib"
    path_model = "Models/" + model_file

    joblib.dump(pipeline, model_path + model_file)
    joblib.dump(pipeline, model_path_api + model_file)
    print(model_path + model_file)
    print(model_path_api + model_file)

    #update_config(config_path,config_api,model_file,path_model)