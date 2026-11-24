import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path


# ============================================================
# 1. Définition des chemins
# ============================================================

RACINE = Path(__file__).resolve().parent

DOSSIER_REFERENCE = RACINE / "reference_v1"

FICHIER_CRITERES = RACINE / "criteres_non_regression.json"

FICHIER_METRIQUES_V1 = DOSSIER_REFERENCE / "metriques_v1_test_independant.json"

FICHIER_METRIQUES_V2 = DOSSIER_REFERENCE / "metriques_v2_test_independant.json"

FICHIER_TEST = DOSSIER_REFERENCE / "jeu_test_commun.csv"


# ============================================================
# 2. Chargement des critères
# ============================================================

with open(FICHIER_CRITERES, "r", encoding="utf-8") as fichier:

    criteres = json.load(fichier)

seuils = criteres["seuils"]


# ============================================================
# 3. Chargement des métriques
# ============================================================

with open(FICHIER_METRIQUES_V1, "r", encoding="utf-8") as fichier:

    metriques_v1 = json.load(fichier)


with open(FICHIER_METRIQUES_V2, "r", encoding="utf-8") as fichier:

    metriques_v2 = json.load(fichier)


# ============================================================
# 4. Comparaison des métriques
# ============================================================

comparaison = {}

for metrique in ["accuracy", "precision", "recall", "f1_score", "auc"]:

    valeur_v1 = metriques_v1[metrique]

    valeur_v2 = metriques_v2[metrique]

    difference = valeur_v2 - valeur_v1

    comparaison[metrique] = {

        "V1": valeur_v1,

        "V2": valeur_v2,

        "difference": difference

    }


# ============================================================
# 5. Vérification des probabilités
# ============================================================

modele_v1 = joblib.load(DOSSIER_REFERENCE / "modele_v1.joblib")

modele_v2 = joblib.load(DOSSIER_REFERENCE / "modele_v2.joblib")

jeu_test = pd.read_csv(FICHIER_TEST)

colonnes = [

    "age",
    "revenu_mensuel",
    "montant_credit",
    "duree_mois",
    "anciennete_activite"

]

X_test = jeu_test[colonnes]

probabilites_v1 = modele_v1.predict_proba(X_test)[:, 1]

probabilites_v2 = modele_v2.predict_proba(X_test)[:, 1]

variation_maximale = float(
    np.max(np.abs(probabilites_v1 - probabilites_v2))
)


# ============================================================
# 6. Vérification du rechargement de V2
# ============================================================

modele_v2_recharge = joblib.load(
    DOSSIER_REFERENCE / "modele_v2.joblib"
)

probabilites_rechargees = modele_v2_recharge.predict_proba(X_test)[:, 1]

ecart_rechargement = float(
    np.max(np.abs(probabilites_v2 - probabilites_rechargees))
)


# ============================================================
# 7. Application des critères d'acceptation
# ============================================================

controle_probabilites = (
    variation_maximale <= seuils["variation_probabilite_max"]
)

baisse_recall = (
    metriques_v1["recall"] - metriques_v2["recall"]
)

controle_recall = (
    baisse_recall <= seuils["baisse_recall_max"]
)

baisse_f1 = (
    metriques_v1["f1_score"] - metriques_v2["f1_score"]
)

controle_f1 = (
    baisse_f1 <= seuils["baisse_f1_max"]
)

controle_rechargement = (
    ecart_rechargement <= seuils["ecart_rechargement_max"]
)


# ============================================================
# 8. Résultat global
# ============================================================

controles = {

    "variation_probabilites": bool(controle_probabilites),

    "controle_recall": bool(controle_recall),

    "controle_f1_score": bool(controle_f1),

    "sauvegarde_rechargement": bool(controle_rechargement)

}

non_regression_reussie = all(controles.values())


# ============================================================
# 9. Construction du rapport
# ============================================================

rapport = {

    "version_reference": "V1",

    "version_cible": "V2",

    "comparaison_metriques": comparaison,

    "variation_probabilite_maximale": variation_maximale,

    "baisse_recall": float(baisse_recall),

    "baisse_f1_score": float(baisse_f1),

    "ecart_rechargement": ecart_rechargement,

    "controles": controles,

    "non_regression_reussie": bool(non_regression_reussie)

}


# ============================================================
# 10. Sauvegarde du rapport
# ============================================================

FICHIER_RAPPORT = RACINE / "rapport_non_regression_v1_v2.json"

with open(FICHIER_RAPPORT, "w", encoding="utf-8") as fichier:

    json.dump(
        rapport,
        fichier,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# 11. Affichage des résultats
# ============================================================

print("\n============================================")

print("COMPARAISON DES VERSIONS V1 ET V2")

print("============================================")


print("\nCOMPARAISON DES MÉTRIQUES")

for nom, valeurs in comparaison.items():

    print(f"\n{nom.upper()}")

    print("V1 :", valeurs["V1"])

    print("V2 :", valeurs["V2"])

    print("Différence :", valeurs["difference"])


print("\n============================================")

print("CONTRÔLE DES CRITÈRES")

print("============================================")


print("\nVariation maximale des probabilités :", variation_maximale)

print("Contrôle probabilités :", controle_probabilites)

print("Baisse du Recall :", baisse_recall)

print("Contrôle Recall :", controle_recall)

print("Baisse du F1-score :", baisse_f1)

print("Contrôle F1-score :", controle_f1)

print("Écart après rechargement :", ecart_rechargement)

print("Contrôle rechargement :", controle_rechargement)


print("\n============================================")

if non_regression_reussie:

    print("RÉSULTAT : NON-RÉGRESSION VALIDÉE")

else:

    print("RÉSULTAT : NON-RÉGRESSION NON VALIDÉE")


print("============================================")

print("\nRapport enregistré :", FICHIER_RAPPORT)