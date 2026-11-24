import json
from pathlib import Path


# ============================================================
# 1. Définition du dossier de référence
# ============================================================

DOSSIER = Path(__file__).resolve().parent

DOSSIER.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Définition des tolérances de non-régression
# ============================================================

CRITERES = {

    "version_reference": "V1",

    "version_cible": "V2",

    "seuils": {

        # Variation maximale autorisée des probabilités
        "variation_probabilite_max": 0.01,

        # Baisse maximale autorisée du rappel
        "baisse_recall_max": 0.02,

        # Baisse maximale autorisée du F1-score
        "baisse_f1_max": 0.02,

        # Écart maximal après rechargement du modèle
        "ecart_rechargement_max": 1e-12
    },

    "tests_obligatoires": {

        "tests_fonctionnels": True,

        "contrat_pipeline": True,

        "comparaison_metriques": True,

        "controle_recall": True,

        "controle_f1_score": True,

        "verification_sauvegarde_rechargement": True
    },

    "regles_acceptation": {

        "tests_fonctionnels": "100 % de réussite",

        "contrat_pipeline": "Respect intégral",

        "probabilites": "Variation maximale de 0,01",

        "recall": "Baisse maximale de 0,02",

        "f1_score": "Baisse maximale de 0,02",

        "rechargement": "Prédictions identiques"
    }
}


# ============================================================
# 3. Enregistrement des critères
# ============================================================

fichier = DOSSIER / "criteres_non_regression.json"

with open(fichier, "w", encoding="utf-8") as f:

    json.dump(
        CRITERES,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# 4. Affichage des critères
# ============================================================

print("========================================")
print("CRITÈRES DE NON-RÉGRESSION")
print("========================================")

print("Version de référence :", CRITERES["version_reference"])

print("Nouvelle version :", CRITERES["version_cible"])

print("\nSeuils définis :")

for nom, valeur in CRITERES["seuils"].items():

    print(f"{nom} : {valeur}")

print("\nFichier enregistré :", fichier)

print("\nDéfinition des critères terminée.")