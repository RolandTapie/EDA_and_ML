import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno


def drop_unused_columns(dataframe: pd.DataFrame, del_cols:list):
    for col in del_cols:
        if col in dataframe.columns.tolist():
            dataframe=dataframe.drop(col,axis=1)
    colonnes=dataframe.columns.tolist()
    return dataframe, colonnes


def Change_ojectColumns_to_categoryColumns(dataframe: pd.DataFrame):

    compteur=0
    obj=[]
    for col in dataframe.columns.tolist():
        if dataframe[col].dtype=="object":
            obj.append(col)
            dataframe[col]=dataframe[col].astype("category")
            compteur+=1
            print(f'les données du champ {col} de type {dataframe[col].dtype} ont été changées en type category')
            print('\n')
            print('Avec les valeurs ci-dessous:')
            print('\n')
            print(dataframe[col].unique())
            print('\n')

    return dataframe, obj, compteur


def List_of_category_columns(dataframe: pd.DataFrame):
    return [col for col in dataframe.columns if dataframe[col].dtype=="category"]


def List_of_float_and_int_columns(dataframe: pd.DataFrame):
    return [col for col in dataframe.columns if ((dataframe[col].dtype=="int64")|(dataframe[col].dtype=="float64"))]


def fill_na_value(dataframe:pd.DataFrame, target:str, int_list:[], cat_list:[]):
    for col in int_list:
        if (col!=target):
            dataframe[col]=dataframe[col].fillna(dataframe[col].mean())

    for col in cat_list:
        if (col!=target):
            dataframe[col]=dataframe[col].fillna(dataframe[col].mode()[0])

    return dataframe


def generate_missing_values_matrix(dataframe: pd.DataFrame, figure_path:str):

    plt.figure(figsize=(1, 1))
    msno.matrix(dataframe)
    img=figure_path+"missing_values_matrix.png"
    plt.savefig(img)
    plt.show()
    return img