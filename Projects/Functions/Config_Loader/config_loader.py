import json

class ConfigLoader:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)


    def get_project_folder(self):
        return self.config["project_folder"]

    # 📌 1. Cible et validation
    def get_project_target(self):
        return self.config["target"]

    def get_cross_validation(self):
        return self.config["cross-validation"]

    def get_fold_count(self):
        return self.config["fold"]

    # 📌 2. PCA
    def use_pca(self):
        return self.config["pca"]

    def get_pca_dim(self):
        return self.config["dim_pca"]

    def get_pca_variance(self):
        return self.config["var_pca"]

    # 📌 3. Données
    def get_dataset_path(self):
        return  self.config["data_path"]

    def get_dataset_filename(self):
        return self.config["data_path"]

    def get_deleted_columns(self):
        return self.config["delete_col"]

    # 📌 4. Modèle
    def get_model_type(self):
        return self.config["model_type"]

    def get_model_path(self):
        return  self.config["model_path"]

    def get_model_api_path(self):
        return  self.config["model_path_api"]

    # 📌 5. Hyperparamètres
    def get_alpha(self):
        return self.config["alpha"]

    def get_test_fraction(self):
        return self.config["fraction_test"]

    def get_random_seed(self):
        return self.config["seed"]

    def get_corr_limit(self):
        return self.config["corr_limit"]

    # 📌 6. Répertoires
    def get_features_json_path(self):
        return self.config["features_types"]

    def get_report_path(self):
        return  self.config["reports_path"]

    def get_figure_path(self):
        return self.config["figures_path"]
