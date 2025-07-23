from Projects.Functions.Config_Loader.config_loader import *
from Projects.Functions.Features_Engineering.features_engineering import *
from Projects.Functions.Modelisation.modelisation import *
from Projects.Functions.Models_generation.models_generation import *
from Projects.Functions.Report.report import *
from Projects.Functions.Pipeline_Modelisation.pipeline_modelisation import *

def execute_pipeline(configuration_path):

    config=ConfigLoader(configuration_path)

    #read configuraion
    print("⚙️-Lecture de la configuration")
    dataset=config.get_dataset_path()
    cible=config.get_project_target()
    modele_type=config.get_model_type()
    test_fraction=config.get_test_fraction()
    model_path=config.get_model_path()
    alpha = config.get_alpha()
    correlation_limit = config.get_corr_limit()
    cross_validation =config.get_cross_validation()
    file_name =config.get_dataset_filename()
    columns_to_delete = config.get_deleted_columns()
    features_path = config.get_features_json_path()
    figure_path = config.get_figure_path()
    model_api_path=config.get_model_api_path()
    k_fold = config.get_fold_count()
    pca_dim = config.get_pca_dim()
    pca_variance = config.get_pca_variance()
    pca=config.use_pca()
    delete_columns = config.get_deleted_columns()

    #project Name
    print("📋-Nom du projet")
    project_name="cancer"
    project_name = project_name + " version_"+ str(datetime.now().year) +"_"+ str(datetime.now().month) +"_"+ str(datetime.now().day) +"_"+datetime.now().strftime("%H%M%S")
    report=Report(project_name)

    #Read the dataset
    print("📜-Lecture du dataset")
    df= dataset_reading(dataset)
    df=dataset_drop_unused_columns(df,delete_columns)
    cible=cible
    raw_features=df.drop(cible, axis=1)
    raw_features= dataset_object_to_categorical(raw_features)
    raw_features=dataset_na_fill(raw_features, cible)
    raw_targets=df[cible]

    #target representation
    if modele_type=="classification":
        dataset_target_representation(df,cible)

    #object_type to categorical_type
    print("🔁-Transition du object vers category")
    df = dataset_object_to_categorical(df)
    df=dataset_na_fill(df,cible)

    #Encoding categorical_type
    print("🔂-Encodage numérique des valeurs catégorielles")
    features, target = encoding_categorical_features(df,cible)

    #Split the dataset
    print("✂️-Division du dataset")
    features_train, features_test, target_train, target_test = split_dataset(features,target,test_fraction,42)

    #scaled numericals features
    print("🧬-Mise à l'échelle des données numériques")
    features, features_train_scaled, features_test_scaled = scale_features(features_train,features_test)

    #run the pca for dimension reduction
    print("🔻-Réduction des dimensions")
    if pca==True:
        features_train, features_test, n_composantes = PCA_dimension_reduction(features_train_scaled,features_test_scaled,pca_variance,pca_dim)

    #generate the lists of models
    print("🗃️-Génération de la liste des modèles")
    modeles=generate_list_of_models(modele_type)

    #Trained the generated models
    print("🤖-Entrainement des modèles")
    pipeline_model, predictions, recall,accuracy,f1score,precision,mse, r2, tab_modele, dict_modeles = training_machine_learning_models(modele_type,modeles,False, features_train_scaled,features_test_scaled,target_train,target_test)

    #Evaluate the models results
    print("📝-Evaluation des modèles")
    best_modele, pipeline_best_model, best = evaluation_of_trained_models(predictions , dict_modeles,pipeline_model, modele_type, target_test, recall,accuracy,f1score,precision,mse, r2, tab_modele, figure_path)

    #The resume of results
    results = generate_results(tab_modele,modele_type)

    #Build pipeline of the production model
    print("🧱-Construction du modèle final")
    pipeline = build_the_final_pipeline(pipeline_best_model,raw_features,raw_targets,cible,pca,pca_dim)

    #Generate the production model
    print("📦-Packaging du modèle de production")
    saving_of_best_model_for_production(model_path,model_api_path,pipeline,best,project_name)

    #Generate the list of features for API calls
    print("🤝-Features API Contract")
    generate_data_contract_for_frontEnd(raw_features,features_path)

    print("\n")

    print("🎉🎉🎉 Modelisation terminated. 🎉🎉🎉")


execute_pipeline(r"C:\Users\tallar\Documents\PROJETS\EDA_and_ML\Projects\Configs\Config.json")