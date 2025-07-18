
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def summary(liste, section, libelle, valeur):
    liste.append([len(liste) + 1, section, libelle, valeur])
    return (liste)


def print_summary(data):
    headers = data[0]
    table_data = data[1:]
    titre = "RESULTATS: analyses et Modélisation"
    titre = titre.center(100, "=")
    print("*" * len(titre))
    print(titre)
    print("*" * len(titre))
    #print(tabulate(table_data, headers, tablefmt="grid"))

    #table_str = titre + "\n" +tabulate(table_data, headers, tablefmt="grid")


def update_config(config_path, cle, valeur):
    with open(config_path, "r") as f:
        data = json.load(f)

    data[cle] = valeur

    with open(config_path, "w") as f:
        json.dump(data, f, indent=4)


def compte(start=0, end=None):
    current = start
    while end is None or current < end:
        yield current
        current += 1

def courbe_entrainement(model, features, targets):


    # Exemple : on utilise un modèle de régression linéaire ici
    model = model
    X=features
    y=targets
    # Génération de la courbe d'apprentissage
    train_sizes, train_scores, validation_scores = learning_curve(
        model, X, y, cv=5, scoring='neg_mean_squared_error',
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )

    # Calcul de la moyenne et de l'écart-type des scores d'entraînement et de validation
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    validation_mean = np.mean(validation_scores, axis=1)
    validation_std = np.std(validation_scores, axis=1)

    # Tracé de la courbe d'apprentissage
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Score d’entraînement')
    plt.plot(train_sizes, validation_mean, 'o-', color='green', label='Score de validation')

    # Affichage de l’intervalle de confiance
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.2)
    plt.fill_between(train_sizes, validation_mean - validation_std, validation_mean + validation_std, color='green', alpha=0.2)

    # Légendes et titres
    plt.title('Courbe d’apprentissage')
    plt.xlabel("Taille de l’échantillon d’entraînement")
    plt.ylabel("Erreur quadratique moyenne négative")
    plt.legend(loc="best")
    plt.grid()
    plt.show()