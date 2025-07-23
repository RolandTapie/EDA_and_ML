import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno


def dataset_reading(file_path, sep=";"):
    return pd.read_csv(file_path)

def dataset_target_representation(df,cible):
    classes = pd.DataFrame(df[cible].value_counts())
    classes['class']=classes.index
    classes['class']=classes['class'].astype("category")
    x = classes["class"]
    y = classes["count"]
    plt.bar(x, y)
    for i in range(len(x)):
        plt.text(x=x[i],               # Position X
                 y=y[i] + 0.5,             # Position Y (juste au-dessus de la barre)
                 s=f"{x[i]}: {y[i]}", # Texte à afficher
                 ha='center',                   # Alignement horizontal
                 va='bottom')                   # Alignement vertical
    plt.show()

def dataset_drop_unused_columns(dataframe: pd.DataFrame, del_cols:list):
    for col in del_cols:
        if col in dataframe.columns.tolist():
            dataframe=dataframe.drop(col,axis=1)

    return dataframe


def Change_ojectColumns_to_categoryColumns(dataframe: pd.DataFrame):

    compteur=0
    obj=[]
    for col in dataframe.columns.tolist():
        if dataframe[col].dtype=="object":
            obj.append(col)
            dataframe[col]=dataframe[col].astype("category")
            compteur+=1

    return dataframe, obj, compteur




def generate_missing_values_matrix(dataframe: pd.DataFrame, figure_path:str):

    plt.figure(figsize=(1, 1))
    msno.matrix(dataframe)
    img=figure_path+"missing_values_matrix.png"
    plt.savefig(img)
    plt.show()
    return img

def dataset_dimensions(dataframe: pd.DataFrame):
    return (dataframe.size,dataframe.shape)

def dataset_data_types(dataframe: pd.DataFrame):
    return dataframe.dtypes.to_frame()

def dataset_stats(dataframe: pd.DataFrame):
    return dataframe.describe()

def dataset_duplicates(dataframe: pd.DataFrame):
    return dataframe.duplicated().sum()

def dataset_remove_duplicates(dataframe: pd.DataFrame):
    return dataframe.drop_duplicates()

def dataset_object_to_categorical(dataframe: pd.DataFrame):
    for col in dataframe.columns:
        if dataframe[col].dtype == "object":
            dataframe[col]=dataframe[col].astype("category")

    return dataframe

def dataset_categorical_values(dataframe: pd.DataFrame):
    df=dataframe
    return [col for col in df.columns if ((df[col].dtype=="category")|(df[col].dtype=="object"))]

def dataset_numerical_values(dataframe: pd.DataFrame):
    df=dataframe
    return [col for col in df.columns if ((df[col].dtype=="int64")|(df[col].dtype=="float64"))]

def dataset_na_values(dataframe: pd.DataFrame):
    return dataframe.isna().any().sum()

def dataset_na_fill(dataframe: pd.DataFrame, cible: str):
    df=dataframe
    for col in dataset_numerical_values(df):
        if (col!=cible):
            df[col]=df[col].fillna(df[col].mean())


    for col in dataset_categorical_values(df):
        if (col!=cible):
            df[col]=df[col].fillna(df[col].mode()[0])

    return df

def dataset_na_drop(dataframe: pd.DataFrame):
    dataframe.dropna(inplace=True)
    return dataframe

def dataset_missing_values(dataframe: pd.DataFrame):
    plt.figure(figsize=(1, 1))
    msno.matrix(dataframe)
    plt.show()