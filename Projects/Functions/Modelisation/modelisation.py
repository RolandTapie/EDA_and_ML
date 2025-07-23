from sklearn.metrics import confusion_matrix, recall_score,f1_score,accuracy_score,precision_score
from sklearn.metrics import mean_squared_error, r2_score,roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from statsmodels.graphics.gofplots import qqplot


def training_machine_learning_models(model_type: str, modeles: dict, validation_croisee, X_train, X_test, y_train, y_test):

    predictions=[]
    recall={}
    accuracy={}
    f1score={}
    precision={}
    mse=[]
    r2=[]

    evaluations=""
    pipeline_model={}
    dict_modeles={}

    if model_type=="classification":
        tab_modele=[["Modele","Recall","F1-Score","Accuracy","Precision"]]

    else:
        tab_modele=[["Modele","MSE","RMSE","R2"]]

    for nom, modele in modeles.items():
        if not validation_croisee:

            pipeline_model[nom]=modele
            modele.fit(X_train,y_train)
            y_pred=modele.predict(X_test)

            if hasattr(modele, 'loss_curve_'):
                # Tracé de l'évolution de la fonction de perte
                plt.plot(modele.loss_curve_)
                plt.title("Évolution de la fonction de perte " + str(modele))
                plt.xlabel("Itérations")
                plt.ylabel("Fonction de perte (loss)")
                plt.grid(True)
                plt.show()
            else:

                print(f"{str(modele)} - Ce modèle ne dispose pas d'un attribut `loss_curve_`.")
            dict_modeles[nom]=modele

            if model_type == "classification":
                y_pred_proba=modele.predict_proba(X_test)[:,1]
                predictions.append((nom,y_pred,y_pred_proba))
            else:
                predictions.append((nom,y_pred,y_test,y_test-y_pred))
        else:
            print ("pas de valisation croisée")

        if model_type == "classification":
            recall[nom]=recall_score(y_test,y_pred)
            precision[nom]=precision_score(y_test,y_pred)
            accuracy[nom]=accuracy_score(y_test,y_pred)
            f1score[nom]=f1_score(y_test,y_pred)

            liste=[]
            liste.append(nom)
            liste.append(str(round(recall[nom],3)))
            liste.append(str(round(f1score[nom],3)))
            liste.append(str(round(accuracy[nom],3)))
            liste.append(str(round(precision[nom],3)))

            evaluations=evaluations+ str(nom) +"  Recall "+str(round(recall[nom],3))+"\n"
            evaluations=evaluations+ str(nom) +"  F1-Score "+str(round(f1score[nom],3))+"\n"
            evaluations=evaluations+ str(nom) +"  Accuracy "+str(round(accuracy[nom],3))+"\n"
            evaluations=evaluations+ str(nom) +"  Precision "+str(round(precision[nom],3))+"\n"

            tab_modele.append(liste)
        else:
            mse.append((nom,mean_squared_error(y_test,y_pred)))
            r2.append((nom,r2_score(y_test,y_pred)))

            liste=[]
            liste.append(nom)
            ms=(mean_squared_error(y_test,y_pred))
            liste.append((round(ms,3)))
            liste.append((round(ms**(0.5),3)))
            r_2=r2_score(y_test,y_pred)
            liste.append((round(r_2,3)))

            tab_modele.append(liste)
    listes = [pipeline_model, predictions, recall,accuracy,f1score,precision,mse, r2, tab_modele, dict_modeles]
    return (pipeline_model, predictions, recall,accuracy,f1score,precision,mse, r2, tab_modele, dict_modeles)


def evaluation_of_trained_models(predictions , dict_modeles,pipeline_model, model_type, y_test, recall,accuracy,f1score,precision,mse, r2, tab_modele, save_path):
    best=""
    scor=0

    if model_type=="classification":
        for val in predictions:
            cm = confusion_matrix(y_test, val[1])
            # Afficher la matrice de confusion avec seaborn
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix : '+ val[0])
            img=save_path+"classification_confusion_matrix_"+str(val[0]+".png")

            #rapport=Report.add_image(rapport,img)
            plt.savefig(img)
            plt.show()

            #resume= utils.summary(resume,"Evaluation","Matrice de confusion : " + val[0],cm)

            texte= "Matrice de confusion : " + str(val[0])
            #rapport=Report.add_text(rapport, texte)

            table_data = [[str(val[0]), "Prédiction : Classe 0", "Prédiction : Classe 1"]]
            table_data.append(["Classe réelle 0", cm[0][0], cm[0][1]])
            table_data.append(["Classe réelle 1", cm[1][0], cm[1][1]])

        tab_modele=[["Modele","Score ROC_AUC"]]
        for val in predictions:
            fpr, tpr, thresholds = roc_curve(y_test, val[2])
            plt.plot(fpr, tpr)
            plt.plot(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1))
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve : ' + val[0])
            img=save_path+"classification_Roc_auc_"+ str(val[0])+".png"
            plt.savefig(img)
            plt.show()
            roc_auc = auc(fpr, tpr)
            liste=[]
            liste.append(str(val[0]))
            liste.append(str(round(roc_auc,3)))
            tab_modele.append(liste)
            if roc_auc > scor:
                scor=(round(roc_auc,3))
                best=val[0]

        best_modele = dict_modeles[best]
        pipeline_best_model=pipeline_model[best]

    else:

        for prediction in predictions:
            name=prediction[0]
            predicted_values=prediction[1]
            df_residuals = pd.DataFrame({"real": y_test , "predicted":predicted_values})
            df_residuals['residuals']=df_residuals['real']-df_residuals['predicted']
            plt.scatter(x=df_residuals['real'],y=df_residuals['predicted'])
            plt.title('real-predicted scatter : ' + name)
            img=save_path+"regression_real_predicted_scatter_"+ name+".png"
            plt.savefig(img)
            plt.show()

            plt.scatter(x=df_residuals.index,y=df_residuals['residuals'], c="red")
            plt.title('residuals scatter : ' + name)
            img=save_path+"regression_residuals_scatter_"+ name+".png"
            plt.savefig(img)
            plt.show()

            sns.histplot(df_residuals['residuals'], kde=True, bins=30)
            plt.title('normality check of residuals : ' + name)
            img=save_path+"regression_residuals_normality_hist_"+ name+".png"
            plt.savefig(img)
            plt.show()

            qqplot(df_residuals['residuals'], line='s')
            plt.title('normality check of residuals : ' + name)
            img=save_path+"regression_residuals_normality_qqplot_"+ name+".png"
            plt.savefig(img)
            plt.show()
        scor=mse[0][1]
        best=mse[0][0]
        tab_modele=[["Modele","RMSE"]]
        for val in mse:
            if val[1]<scor:
                scor=val[1]
                best=val[0]

        best_modele = dict_modeles[best]
        pipeline_best_model=pipeline_model[best]

    return best_modele, pipeline_best_model, best

def generate_results(list_results, model_type):
    results = pd.DataFrame(list_results[1:], columns=list_results[0])
    if model_type == "classification":
        results["Precision"]=results["Precision"].astype("float64")
        results["Accuracy"]=results["Accuracy"].astype("float64")
        results["F1-Score"]=results["F1-Score"].astype("float64")
        results["Recall"]=results["Recall"].astype("float64")
        results = results.sort_values(by = "Precision", ascending=False)
    elif model_type == "regression":
        "MSE","RMSE","R2"
        results["MSE"]=results["MSE"].astype("float64")
        results["RMSE"]=results["RMSE"].astype("float64")
        results["R2"]=results["R2"].astype("float64")
        results = results.sort_values(by = "R2", ascending=False)
    return results