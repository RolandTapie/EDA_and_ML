import pandas as pd
from Data_cleaning.Data_cleaning import List_of_category_columns
from Data_cleaning.Data_cleaning import List_of_float_and_int_columns
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import iqr, shapiro,kstest
from statsmodels.api import qqplot
from itertools import combinations
import numpy as np


def category_data_analysis(dataframe: pd.DataFrame, target: str):
    cats = [col for col in List_of_category_columns(dataframe)]
    df_cats = dataframe[cats]
    analyse = ""
    for cat in cats:
        print(df_cats[cat].value_counts(normalize=True).round(2))
        print(df_cats.groupby([cat, target]).size().reset_index(name='count'))
        analyse = analyse + cat + '\n'
        analyse = analyse+("*"*10) + '\n'
        analyse = analyse+str(df_cats[cat].value_counts(normalize=True).round(2)) + '\n'

    return analyse


def distribution_analysis_for_int_and_float_features(dataframe: pd.DataFrame, save_path):
    liste_float_int=List_of_float_and_int_columns(dataframe)

    size=len(liste_float_int)
    fig, axes = plt.subplots((size//4)+1, 4, figsize=(10, 5))
    axes=axes.flatten()

    for col in liste_float_int:
        sns.histplot(dataframe[col],ax=axes[liste_float_int.index(col)], kde=True)
        axes[liste_float_int.index(col)].set_title(col)

    img=save_path+"hist_distribution.png"
    plt.savefig(img)
    plt.show()
    return img

def checking_normality_of_features(dataframe: pd.DataFrame, alpha:float, test):

    liste_float_int=List_of_float_and_int_columns(dataframe)
    liste_norm=""
    liste=[["champ","stats","p_value"]]
    print(str(test))
    for col in liste_float_int:

        if test==kstest:
            stats, p_value = test(dataframe[col],'norm')
        else:
            stats, p_value = test(dataframe[col])
        stats=round(stats,3)
        p_value=round(p_value,3)
        if p_value>alpha:
            liste_norm=liste_norm+col+f" : p_value ({p_value} > alpha ({alpha})) Non rejet de H0 > les données suivent une distribution normale"+'\n'
        else:
            liste.append([col,stats,p_value])
            liste_norm=liste_norm+col+f" : p_value ({p_value} < alpha ({alpha})) Rejet de H0 > les données ne suivent pas une distribution normale"+'\n'
        print(col,stats, p_value)
    texte=f" Distribution normale des données (Test de {test}) : Mise en place des hypothèses (alpha) = {(alpha)}:" +"\n" \
          + "H0 > (les données suivent une distribution normale)" +"\n"+ "H1 > (les données ne suivent pas une distribution normale) \n" \
          +"\n" + f"Ci-dessous les champs du dataset qui ne semblent pas suivre une distribution suivant le test de {test}"
    infos=liste

    return texte, infos, liste_norm, str(test)

def checking_normality_of_features_QQPLOT(dataframe: pd.DataFrame, save_path):
    liste_float_int=List_of_float_and_int_columns(dataframe)
    size=len(liste_float_int)
    fig, axes = plt.subplots((size//4)+1, 4, figsize=(10, 10))
    axes=axes.flatten()

    for col in liste_float_int:
        qqplot(dataframe[col], fit=True, line="45",ax=axes[liste_float_int.index(col)])
        axes[liste_float_int.index(col)].set_title(col)

    plt.tight_layout()
    img=save_path+"qqplot.png"
    plt.savefig(img)
    plt.show()
    return img

def generate_correlation_matrix(dataframe: pd.DataFrame,corr_limit, save_path):
    liste_float_int=List_of_float_and_int_columns(dataframe)
    df_corr=dataframe[liste_float_int].corr()
    sns.heatmap(df_corr,annot=True)
    plt.savefig(save_path+"correlation_matrix.png")
    plt.show()
    liste=[["champ_x","champ_y","Coef_corr"]]
    corr_check=[]
    liste_corr=""
    for i in range(len(df_corr)):
        for j in range(len(df_corr)):
            if (abs(df_corr.iloc[i,j])>=corr_limit)&(i<j):
                liste.append([str(df_corr.columns[i]), str(df_corr.columns[j]),str(round(df_corr.iloc[i,j],2))])
                corr_check.append((str(df_corr.columns[i])+ " > " + str(df_corr.columns[j])+ " : " + str(round(df_corr.iloc[i,j],3))))
                liste_corr=liste_corr+str(df_corr.columns[i])+ " > " + str(df_corr.columns[j])+ " : " + str(round(df_corr.iloc[i,j],3))+'\n'
    print("*"*100)
    print("Liste des corrélations à analyser")
    print("*"*100)
    for cor in corr_check:
        print(cor)

    return liste_corr,liste,df_corr

def generate_box_plots(dataframe: pd.DataFrame,cible,save_path, model_type:str="Regression"):
    liste_float_int=List_of_float_and_int_columns(dataframe)
    size=len(liste_float_int)
    fig, axes = plt.subplots((size//4)+1, 4, figsize=(10, 5))
    axes=axes.flatten()
    for i, cols in enumerate(liste_float_int):
        if model_type=="classification":
            sns.boxenplot(data=dataframe,y=cols,hue=cible,ax=axes[i],legend=False)
        else:
            sns.boxenplot(data=dataframe,y=cols,ax=axes[i])

    plt.tight_layout()
    plt.savefig(save_path+"Boxen_plot.png")
    plt.show()

def generate_scatter_plots(dataframe: pd.DataFrame, save_path,cible, model_type:str="Regression"):
    liste_float_int=List_of_float_and_int_columns(dataframe)
    size=len(list(combinations(liste_float_int,2)))
    print(size)
    fig, axes = plt.subplots(max(size//4,1)+1, 4, figsize=(10, 10))
    axes=axes.flatten()
    for i, cols in enumerate(combinations(liste_float_int,2)):
        if model_type=="classification":
            sns.scatterplot(data=dataframe,x=cols[0],y=cols[1],hue=cible,ax=axes[i])
        else:
            sns.scatterplot(data=dataframe,x=cols[0],y=cols[1],ax=axes[i])

    plt.tight_layout()
    plt.savefig(save_path+"Scatter_plot.png")
    plt.show()

def checking_outliers (dataframe: pd.DataFrame):
    liste_float_int=List_of_float_and_int_columns(dataframe)
    outlierss=""
    for col in liste_float_int:
        if dataframe[col].nunique()>=5:
            inf=dataframe[col].quantile(0.25)-1.5*iqr(dataframe[col])
            sup=dataframe[col].quantile(0.75)+1.5*iqr(dataframe[col])
            df_outliers=dataframe[(dataframe[col]<inf) | (dataframe[col]>sup)][col]
            outlierss=outlierss + "la colonne " + col + " contient "+ str(len(df_outliers))+ " outliers pour un iqr de " + str(iqr(dataframe[col])) +'\n'
            dataframe[col]=np.where(dataframe[col]>sup,dataframe[col].mean(),dataframe[col])
            dataframe[col]=np.where(dataframe[col]<inf,dataframe[col].mean(),dataframe[col])

    print("Outliers avant cleaning")
    print(df_outliers)
    print("*"*100)

    outliers=""
    for col in liste_float_int:
        if dataframe[col].nunique()>=5:
            inf=dataframe[col].quantile(0.25)-1.5*iqr(dataframe[col])
            sup=dataframe[col].quantile(0.75)+1.5*iqr(dataframe[col])
            df_outliers=dataframe[(dataframe[col]<inf) | (dataframe[col]>sup)][col]

    print("Outliers après cleaning")
    print(df_outliers)
    print("*"*100)
    return dataframe,outlierss