import json
from pathlib import Path

import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# --------------------------------------------------
# 1. Chemins des modèles
# --------------------------------------------------

DOSSIER = Path(__file__).resolve().parent
DOSSIER_REFERENCE = DOSSIER / "reference_v1"

MODELES = {
    "V1": DOSSIER_REFERENCE / "modele_v1.joblib",
    "V2": DOSSIER_REFERENCE / "modele_v2_C08.joblib"
}

resultats = {}


# --------------------------------------------------
# 2. Vérification du contrat technique
# --------------------------------------------------

def verifier_modele(nom, chemin):

    print("\n======================================")
    print(f"VÉRIFICATION DU MODÈLE {nom}")
    print("======================================")

    modele = joblib.load(chemin)

    controles = {}

    # Vérifier que le modèle est un Pipeline
    controles["pipeline"] = isinstance(modele, Pipeline)

    # Vérifier les étapes du pipeline
    controles["etapes"] = (
        "scaler" in modele.named_steps
        and "classifier" in modele.named_steps
    )

    # Vérifier le StandardScaler
    controles["standard_scaler"] = isinstance(
        modele.named_steps["scaler"],
        StandardScaler
    )

    # Vérifier la régression logistique
    controles["logistic_regression"] = isinstance(
        modele.named_steps["classifier"],
        LogisticRegression
    )

    # Vérifier le nombre de variables
    controles["cinq_variables"] = (
        modele.n_features_in_ == 5
    )

    # Vérifier le paramètre C
    valeur_C = modele.named_steps["classifier"].C

    if nom == "V1":
        controles["parametre_C"] = valeur_C == 1.0
    else:
        controles["parametre_C"] = valeur_C == 0.8

    # Afficher les résultats
    for controle, resultat in controles.items():
        statut = "OK" if resultat else "ÉCHEC"
        print(f"{controle} : {statut}")

    print(f"Nombre de variables : {modele.n_features_in_}")
    print(f"Paramètre C : {valeur_C}")

    valide = all(controles.values())

    print(
        "\nCONTRAT TECHNIQUE :",
        "VALIDÉ" if valide else "NON VALIDÉ"
    )

    return {
        "version": nom,
        "controles": controles,
        "nombre_variables": modele.n_features_in_,
        "parametre_C": valeur_C,
        "contrat_valide": valide
    }


# --------------------------------------------------
# 3. Vérifier les deux modèles
# --------------------------------------------------

for nom, chemin in MODELES.items():

    if not chemin.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {chemin}"
        )

    resultats[nom] = verifier_modele(nom, chemin)


# --------------------------------------------------
# 4. Bilan général
# --------------------------------------------------

validation_globale = all(
    resultat["contrat_valide"]
    for resultat in resultats.values()
)

rapport = {
    "verification": "Contrat technique des pipelines",
    "modeles": resultats,
    "validation_globale": validation_globale
}

chemin_rapport = DOSSIER / "rapport_contrat_pipeline.json"

with open(chemin_rapport, "w", encoding="utf-8") as fichier:
    json.dump(
        rapport,
        fichier,
        indent=4,
        ensure_ascii=False
    )

print("\n======================================")
print("BILAN FINAL")
print("======================================")

print(
    "VALIDATION GLOBALE :",
    "RÉUSSIE" if validation_globale else "ÉCHEC"
)

print(f"\nRapport enregistré : {chemin_rapport}")