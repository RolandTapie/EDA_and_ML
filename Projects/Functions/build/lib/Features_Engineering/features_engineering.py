import pandas as pd
from Data_cleaning.Data_cleaning import List_of_category_columns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import json
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


def encoding_categorical_features(dataframe: pd.DataFrame,target:str):
    features=dataframe.drop(target, axis=1)
    liste_cat = List_of_category_columns(dataframe)
    df_cat=features[liste_cat]
    encoder = OneHotEncoder(sparse_output=False)
    encoded_data = encoder.fit_transform(df_cat)
    encoded_df=pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(df_cat.columns.tolist()))
    features = pd.concat([features.drop(liste_cat, axis=1), encoded_df], axis=1)
    return features, dataframe[target]


def keep_original_data_for_pipeline(dataframe: pd.DataFrame, fraction:float, cible,seed):

    if fraction==0:
        X_train_modele=dataframe.drop(cible, axis=1)
        X_test_modele=dataframe.drop(cible, axis=1)
        y_train_modele=dataframe[cible]
        y_test_modele=dataframe[cible]
    else:
        features_modele=dataframe.drop(cible, axis=1)
        targets_modele=dataframe[cible]
        X_train_modele, X_test_modele, y_train_modele, y_test_modele = train_test_split(features_modele, targets_modele, test_size=fraction, random_state=seed)

    return X_train_modele, X_test_modele, y_train_modele, y_test_modele


def generate_data_contract_for_frontEnd(X_train_modele : pd.DataFrame, contract_path:str):

    format_features= {}
    for col in X_train_modele.columns.to_list():
        format_features[col] = str(X_train_modele[col].dtype)
        if X_train_modele[col].dtype == "category":
            val = list(pd.unique(X_train_modele[col]))
            format_features[col]= str(X_train_modele[col].dtype) +"  : values = "+str((val))

        if (X_train_modele[col].dtype == "float64")|(X_train_modele[col].dtype == "int64"):
            val = list(pd.unique(X_train_modele[col]))
            format_features[col]= str(X_train_modele[col].dtype) +"  : values = [ "+str((X_train_modele[col].min())) +" to "+str((X_train_modele[col].max()))+" ]"

    with open(contract_path,"w") as f:
        json.dump(format_features, f, indent=4)

    print(format_features)


def scale_features(dataframe_train: pd.DataFrame, dataframe_test: pd.DataFrame):
    scaler = StandardScaler()
    features = scaler.fit_transform(dataframe_train)
    X_train = scaler.fit_transform(dataframe_train)
    X_test=scaler.transform(dataframe_test)

    return features, X_train, X_test

def split_dataset(features: pd.DataFrame, targets : pd.DataFrame, fraction:float,seed):

    if fraction==0:
        X_train=features
        X_test=features
        y_train=targets
        y_test=targets
    else:
        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=fraction, random_state=seed)

    return X_train, X_test, y_train, y_test



def TSNE_dimension_reduction(features: pd.DataFrame, y_train, save_path):
    model_tsne = TSNE(learning_rate=100)
    features_tsne = model_tsne.fit_transform(features)
    xs = features_tsne[:,0]
    ys = features_tsne[:,1]
    plt.scatter(xs, ys, c=y_train)
    plt.savefig(save_path+"tsne.png")
    plt.show()



def PCA_dimension_reduction(features:pd.DataFrame,X_train:pd.DataFrame,X_test:pd.DataFrame, var_pca:float,dim_pca):

    if dim_pca > 0:
        model_PCA=PCA(dim_pca)
    else:
        model_PCA=PCA()

    features_PCA=model_PCA.fit_transform(features)
    x_features=range(model_PCA.n_components_)
    variance_cumulee = np.cumsum(model_PCA.explained_variance_ratio_)
    # Trouver le nombre de dimensions pour expliquer au moins var_pca % de la variance
    n_composantes = np.argmax(variance_cumulee >= var_pca) + 1
    print(f"Nombre de dimensions intrinsèques (pour {var_pca} de variance): {n_composantes}")

    plt.bar(x_features,model_PCA.explained_variance_)
    plt.show()
    print(model_PCA.explained_variance_)
    print(model_PCA.n_components_)

    model_PCA=PCA(n_composantes)
    features_train_pca=model_PCA.fit_transform(features)
    features_test_pca=model_PCA.fit_transform(X_test)
    X_train=model_PCA.transform(X_train)
    X_test=model_PCA.transform(X_test)

    return X_train, X_test, features_train_pca,features_test_pca,n_composantes
