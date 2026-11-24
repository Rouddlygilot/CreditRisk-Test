import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# 1. Définition des chemins
# ============================================================

DOSSIER = Path(__file__).resolve().parent

FICHIER_MODELE = DOSSIER / "modele_v1.joblib"

FICHIER_TEST = DOSSIER / "jeu_test_commun.csv"


# ============================================================
# 2. Chargement du modèle et du jeu de test
# ============================================================

modele = joblib.load(FICHIER_MODELE)

jeu_test = pd.read_csv(FICHIER_TEST)

colonnes = [
    "age",
    "revenu_mensuel",
    "montant_credit",
    "duree_mois",
    "anciennete_activite"
]

X_test = jeu_test[colonnes]

y_test = jeu_test["valeur_reelle"]


# ============================================================
# 3. Réalisation des prédictions
# ============================================================

predictions = modele.predict(X_test)

probabilites = modele.predict_proba(X_test)[:, 1]


# ============================================================
# 4. Calcul des métriques
# ============================================================

metriques = {

    "version": "V1",

    "type_evaluation": "Jeu de test indépendant",

    "nombre_observations": int(len(X_test)),

    "accuracy": float(
        accuracy_score(y_test, predictions)
    ),

    "precision": float(
        precision_score(y_test, predictions, zero_division=0)
    ),

    "recall": float(
        recall_score(y_test, predictions, zero_division=0)
    ),

    "f1_score": float(
        f1_score(y_test, predictions, zero_division=0)
    ),

    "auc": float(
        roc_auc_score(y_test, probabilites)
    )

}


# ============================================================
# 5. Matrice de confusion
# ============================================================

matrice = confusion_matrix(y_test, predictions)


# ============================================================
# 6. Sauvegarde des métriques
# ============================================================

with open(
    DOSSIER / "metriques_v1_test_independant.json",
    "w",
    encoding="utf-8"
) as fichier:

    json.dump(
        metriques,
        fichier,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# 7. Sauvegarde des prédictions
# ============================================================

resultats = X_test.copy()

resultats["valeur_reelle"] = y_test

resultats["prediction"] = predictions

resultats["probabilite_risque"] = probabilites

resultats.to_csv(
    DOSSIER / "predictions_v1_test_independant.csv",
    index=False,
    encoding="utf-8"
)


# ============================================================
# 8. Affichage des résultats
# ============================================================

print("\n==============================================")
print("ÉVALUATION DE LA VERSION V1")
print("JEU DE TEST INDÉPENDANT")
print("==============================================")

print("\nNombre d'observations :", len(X_test))

print("\nMÉTRIQUES DE PERFORMANCE")

for nom, valeur in metriques.items():

    print(f"{nom} : {valeur}")


print("\nMATRICE DE CONFUSION")

print(matrice)


print("\nFichiers enregistrés :")

print("metriques_v1_test_independant.json")

print("predictions_v1_test_independant.csv")

print("\nÉvaluation de la version V1 terminée.")