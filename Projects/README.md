📌 1. Compréhension générale du dataset
Dimensions du dataset (nombre de lignes et de colonnes)

Aperçu des premières lignes (head(), tail())

Types de données (dtypes)

Description statistique (describe())

Vérification des doublons

Lecture de la documentation / dictionnaire de données s’il existe

📌 2. Qualité des données
🔍 Valeurs manquantes
Nombre et pourcentage de valeurs manquantes par colonne

Répartition des valeurs manquantes (ex : heatmap)

🔁 Doublons
Détection des lignes dupliquées

Suppression si pertinent

❌ Valeurs aberrantes (outliers)
Détection via :

Boxplots

Z-score ou IQR

Méthodes robustes (Isolation Forest, etc.)

Traitement (suppression ou remplacement)

❓ Incohérences
Incohérences logiques (ex : date de fin < date de début)

Incohérences de formats (ex : majuscules vs minuscules, types de chaînes)

Données hors domaine (ex : âges négatifs)

📌 3. Analyse des variables
🔣 Variables qualitatives (catégorielles)
Nombre de modalités

Fréquence des catégories (value_counts())

Catégories rares ou déséquilibrées

Cohérence des libellés

🔢 Variables quantitatives
Statistiques descriptives : moyenne, médiane, min, max, écart-type

Distribution (histogrammes, densité)

Symétrie (skewness)

Normalité (test de Shapiro-Wilk, QQ plot)

📌 4. Relations entre variables
🧮 Corrélations
Matrice de corrélation pour variables numériques

Heatmap

Corrélation de Spearman/Pearson/Kendall selon le cas

📊 Visualisations croisées
Catégorielle vs numérique : boxplot, violinplot

Numérique vs numérique : scatter plot, regression plot

Catégorielle vs catégorielle : crosstab, barplot

📌 5. Analyse temporelle (si données temporelles)
Format de date cohérent

Distribution des dates

Séries chronologiques : tendance, saisonnalité

Valeurs manquantes ou irrégularités temporelles

📌 6. Préparation pour la suite
Codage des variables catégorielles (Label Encoding, One-hot)

Création de variables dérivées utiles

Transformation éventuelle (log, standardisation, normalisation)

Séparation jeu d'entraînement / test si besoin