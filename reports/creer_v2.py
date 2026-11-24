import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

RACINE = Path(__file__).resolve().parent

DOSSIER_REFERENCE = RACINE / "reference_v1"

DOSSIER_REFERENCE.mkdir(parents=True, exist_ok=True)

# Paramètre de régularisation de V2
PARAMETRE_C = 0.8

# Colonnes utilisées par le modèle
COLONNES = [
    "age",
    "revenu_mensuel",
    "montant_credit",
    "duree_mois",
    "anciennete_activite"
]


# ============================================================
# 2. REPRODUCTION DES DONNÉES D'ENTRAÎNEMENT DE V1
# ============================================================

rng = np.random.default_rng(42)

nombre_observations = 1000

age = rng.integers(18, 75, nombre_observations)

revenu = rng.uniform(50000, 1000000, nombre_observations)

montant = rng.uniform(100000, 5000000, nombre_observations)

duree = rng.integers(3, 61, nombre_observations)

anciennete = rng.uniform(0, 30, nombre_observations)


# Construction du DataFrame d'entraînement

X_train = pd.DataFrame({

    "age": age,

    "revenu_mensuel": revenu,

    "montant_credit": montant,

    "duree_mois": duree,

    "anciennete_activite": anciennete

})


# Génération de la cible

risque = (
    0.8 * montant / revenu
    + 0.015 * duree
    - 0.04 * anciennete
    - 0.01 * age
)

probabilite_cible = 1 / (1 + np.exp(-risque))

y_train = (
    rng.random(nombre_observations) < probabilite_cible
).astype(int)


# ============================================================
# 3. CRÉATION DU MODÈLE V2
# ============================================================

modele_v2 = Pipeline([

    ("scaler", StandardScaler()),

    (
        "classifier",
        LogisticRegression(
            C=PARAMETRE_C,
            random_state=42
        )
    )

])


# ============================================================
# 4. ENTRAÎNEMENT DU MODÈLE
# ============================================================

modele_v2.fit(X_train, y_train)


print("\n======================================")
print("ENTRAÎNEMENT DU MODÈLE V2")
print("======================================")

print("Observations d'entraînement :", len(X_train))

print("Variables explicatives :", X_train.shape[1])

print("Paramètre C :", PARAMETRE_C)


# ============================================================
# 5. CHARGEMENT DU JEU DE TEST INDÉPENDANT
# ============================================================

FICHIER_TEST = DOSSIER_REFERENCE / "jeu_test_commun.csv"

if not FICHIER_TEST.exists():

    raise FileNotFoundError(
        f"Jeu de test introuvable : {FICHIER_TEST}"
    )


jeu_test = pd.read_csv(FICHIER_TEST)

# Vérification des colonnes nécessaires

colonnes_manquantes = [
    colonne
    for colonne in COLONNES + ["valeur_reelle"]
    if colonne not in jeu_test.columns
]

if colonnes_manquantes:

    raise ValueError(
        f"Colonnes manquantes : {colonnes_manquantes}"
    )


X_test = jeu_test[COLONNES]

y_test = jeu_test["valeur_reelle"]


# ============================================================
# 6. RÉALISATION DES PRÉDICTIONS
# ============================================================

predictions = modele_v2.predict(X_test)

probabilites = modele_v2.predict_proba(X_test)[:, 1]


# ============================================================
# 7. CALCUL DES MÉTRIQUES
# ============================================================

metriques = {

    "version": "V2",

    "parametre_C": PARAMETRE_C,

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
# 8. MATRICE DE CONFUSION
# ============================================================

matrice = confusion_matrix(
    y_test,
    predictions
)


# ============================================================
# 9. SAUVEGARDE DU MODÈLE V2
# ============================================================

fichier_modele = (
    DOSSIER_REFERENCE / "modele_v2.joblib"
)

joblib.dump(
    modele_v2,
    fichier_modele
)


# ============================================================
# 10. SAUVEGARDE DES MÉTRIQUES
# ============================================================

fichier_metriques = (
    DOSSIER_REFERENCE / "metriques_v2_test_independant.json"
)

with open(
    fichier_metriques,
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
# 11. SAUVEGARDE DES PRÉDICTIONS
# ============================================================

resultats = X_test.copy()

resultats["valeur_reelle"] = y_test

resultats["prediction"] = predictions

resultats["probabilite_risque"] = probabilites

fichier_predictions = (
    DOSSIER_REFERENCE / "predictions_v2_test_independant.csv"
)

resultats.to_csv(
    fichier_predictions,
    index=False,
    encoding="utf-8"
)


# ============================================================
# 12. AFFICHAGE DES RÉSULTATS
# ============================================================

print("\n======================================")
print("ÉVALUATION DU MODÈLE V2")
print("======================================")

for nom, valeur in metriques.items():

    print(f"{nom} : {valeur}")


print("\nMATRICE DE CONFUSION")

print(matrice)


# ============================================================
# 13. VÉRIFICATION DE SAUVEGARDE ET RECHARGEMENT
# ============================================================

modele_recharge = joblib.load(fichier_modele)

probabilites_rechargees = (
    modele_recharge.predict_proba(X_test)[:, 1]
)

predictions_rechargees = (
    modele_recharge.predict(X_test)
)

ecart_maximal = float(
    np.max(
        np.abs(
            probabilites - probabilites_rechargees
        )
    )
)

probabilites_identiques = np.allclose(
    probabilites,
    probabilites_rechargees,
    rtol=0,
    atol=1e-12
)

predictions_identiques = np.array_equal(
    predictions,
    predictions_rechargees
)


# ============================================================
# 14. AFFICHAGE DE LA VÉRIFICATION
# ============================================================

print("\n======================================")
print("VÉRIFICATION DU RECHARGEMENT")
print("======================================")

print(
    "Prédictions identiques :",
    predictions_identiques
)

print(
    "Probabilités identiques :",
    probabilites_identiques
)

print("Écart maximal :", ecart_maximal)


if not probabilites_identiques or not predictions_identiques:

    raise AssertionError(
        "Échec de la vérification après rechargement."
    )


# ============================================================
# 15. AFFICHAGE DES FICHIERS
# ============================================================

print("\nFICHIERS ENREGISTRÉS")

print("Modèle :", fichier_modele)

print("Métriques :", fichier_metriques)

print("Prédictions :", fichier_predictions)

print("\n======================================")
print("ENTRAÎNEMENT ET ÉVALUATION V2 TERMINÉS")
print("======================================")