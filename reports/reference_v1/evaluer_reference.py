import json
import joblib
import numpy as np
import pandas as pd
import sys

from pathlib import Path


# ============================================================
# Configuration du chemin du projet
# ============================================================

RACINE_PROJET = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(RACINE_PROJET))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from app.main import model


# ============================================================
# 1. Définition des chemins
# ============================================================

DOSSIER = Path("reports/reference_v1")

DOSSIER.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Reproduction des données de référence
# ============================================================

rng = np.random.default_rng(42)

nombre_observations = 1000

age = rng.integers(18, 75, nombre_observations)

revenu = rng.uniform(
    50000, 1000000, nombre_observations
)

montant = rng.uniform(
    100000, 5000000, nombre_observations
)

duree = rng.integers(
    3, 61, nombre_observations
)

anciennete = rng.uniform(
    0, 30, nombre_observations
)

X = np.column_stack([
    age,
    revenu,
    montant,
    duree,
    anciennete
])


# Reproduction de la cible synthétique

risque = (
    0.8 * montant / revenu
    + 0.015 * duree
    - 0.04 * anciennete
    - 0.01 * age
)

probabilite_cible = 1 / (1 + np.exp(-risque))

y = (
    rng.random(nombre_observations) < probabilite_cible
).astype(int)


# ============================================================
# 3. Vérification du modèle
# ============================================================

if model is None:
    raise RuntimeError("Le modèle V1 n'est pas disponible.")

print("\n===== ÉVALUATION DE LA VERSION V1 =====")

print("Nombre d'observations :", len(X))
print("Nombre de variables :", X.shape[1])


# ============================================================
# 4. Calcul des prédictions
# ============================================================

probabilites = model.predict_proba(X)[:, 1]

predictions = model.predict(X)


# ============================================================
# 5. Calcul des métriques
# ============================================================

metriques = {

    "version": "V1",

    "nombre_observations": int(len(X)),

    "accuracy": float(
        accuracy_score(y, predictions)
    ),

    "precision": float(
        precision_score(y, predictions, zero_division=0)
    ),

    "recall": float(
        recall_score(y, predictions, zero_division=0)
    ),

    "f1_score": float(
        f1_score(y, predictions, zero_division=0)
    ),

    "auc": float(
        roc_auc_score(y, probabilites)
    )
}


# ============================================================
# 6. Affichage des résultats
# ============================================================

print("\n===== MÉTRIQUES DE RÉFÉRENCE =====")

for nom, valeur in metriques.items():

    print(f"{nom} : {valeur}")


print("\n===== MATRICE DE CONFUSION =====")

print(confusion_matrix(y, predictions))


# ============================================================
# 7. Sauvegarde des métriques
# ============================================================

fichier_metriques = DOSSIER / "metriques_v1.json"

with open(fichier_metriques, "w", encoding="utf-8") as fichier:

    json.dump(
        metriques,
        fichier,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# 8. Sauvegarde des prédictions
# ============================================================

resultats = pd.DataFrame({

    "valeur_reelle": y,

    "prediction": predictions,

    "probabilite_defaut": probabilites

})

fichier_predictions = DOSSIER / "predictions_v1.csv"

resultats.to_csv(
    fichier_predictions,
    index=False,
    encoding="utf-8"
)


# ============================================================
# 9. Sauvegarde du modèle
# ============================================================

fichier_modele = DOSSIER / "modele_v1.joblib"

joblib.dump(model, fichier_modele)


# ============================================================
# 10. Rechargement du modèle
# ============================================================

modele_recharge = joblib.load(fichier_modele)

nouvelles_probabilites = modele_recharge.predict_proba(X)[:, 1]

nouvelles_predictions = modele_recharge.predict(X)


# ============================================================
# 11. Vérification de la cohérence
# ============================================================

predictions_identiques = np.array_equal(
    predictions,
    nouvelles_predictions
)

probabilites_identiques = np.allclose(
    probabilites,
    nouvelles_probabilites,
    rtol=0,
    atol=1e-12
)

ecart_maximal = float(
    np.max(np.abs(probabilites - nouvelles_probabilites))
)

print("\n===== TEST DE SAUVEGARDE ET RECHARGEMENT =====")

print("Prédictions identiques :", predictions_identiques)

print("Probabilités identiques :", probabilites_identiques)

print("Écart maximal :", ecart_maximal)


# ============================================================
# 12. Vérification finale
# ============================================================

if not predictions_identiques or not probabilites_identiques:

    raise AssertionError(
        "Les prédictions ont changé après rechargement."
    )

print("\n===== SAUVEGARDE TERMINÉE =====")

print("Métriques :", fichier_metriques)

print("Prédictions :", fichier_predictions)

print("Modèle :", fichier_modele)