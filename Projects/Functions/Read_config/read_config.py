import json

def get_config(config_path: str):
    with open(config_path, 'r') as fichier:
        config = json.load(fichier)

    cible=config["target"]
    validation_croisee=config["cross-validation"]
    pca=config["pca"]
    dim_pca=config["dim_pca"]
    var_pca=config["var_pca"]
    alpha=config["alpha"]
    fraction=config["fraction_test"]
    seed=config["seed"]
    corr_limit=config["corr_limit"]
    fold=config["fold"]
    del_cols=config["delete_col"]
    file_path=config["project_folder"] + config["data_path"]
    dataset=config["data_path"]
    model_path=config["project_folder"] + config["model_path"]
    model_path_api=config["project_folder"] + config["model_path_api"]
    features_json=config["project_folder"] + config["features_types"]
    report_file=config["project_folder"] +config["reports_path"]
    figure_path= config["project_folder"] +config["figures_path"]
    model_type=config["model_type"]

    if config["kaggle"]==True:
        file_path=config["project_folder"] + config["kaggle_path"]
        fraction=0
        cible=config["kaggle_target"]
        report_kaggle=config["project_folder"] +config["submission_kaggle"]
        test_kaggle=config["project_folder"] + config["kaggle_test_path"]

    return cible,validation_croisee, pca, dim_pca,var_pca,alpha,fraction,seed,corr_limit,fold,del_cols,file_path,dataset,model_path,model_path_api,features_json,report_file,figure_path,model_type

def update_config(config_path, cle, valeur):
    with open(config_path, "r") as f:
        data = json.load(f)

    data[cle] = valeur

    with open(config_path, "w") as f:
        json.dump(data, f, indent=4)