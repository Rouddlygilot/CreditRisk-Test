# -*- coding: utf-8 -*-

import json
import re
import subprocess
import sys
from pathlib import Path
from datetime import datetime


# ==================================================
# 1. INITIALISATION DES CHEMINS
# ==================================================

DOSSIER = Path(__file__).resolve().parent
RACINE = DOSSIER.parent

FICHIER_COMPARAISON = DOSSIER / "rapport_non_regression_v1_v2.json"
FICHIER_CONTRAT = DOSSIER / "rapport_contrat_pipeline.json"
FICHIER_CRITERES = DOSSIER / "criteres_non_regression.json"

FICHIER_RAPPORT_JSON = DOSSIER / "rapport_global_non_regression.json"
FICHIER_RAPPORT_TXT = DOSSIER / "rapport_global_non_regression.txt"


# ==================================================
# 2. FONCTION DE LECTURE DES FICHIERS JSON
# ==================================================

def lire_json(chemin):

    if not chemin.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {chemin}"
        )

    with open(chemin, "r", encoding="utf-8-sig") as fichier:
        return json.load(fichier)


# ==================================================
# 3. CHARGEMENT DES RAPPORTS
# ==================================================

comparaison = lire_json(FICHIER_COMPARAISON)
contrat = lire_json(FICHIER_CONTRAT)
criteres = lire_json(FICHIER_CRITERES)

print("\n============================================")
print("CONSOLIDATION DE LA NON-RÉGRESSION")
print("============================================")


# ==================================================
# 4. RELANCEMENT DES TESTS FONCTIONNELS
# ==================================================

print("\n[1] Exécution des tests fonctionnels...")

commande = [
    sys.executable,
    "-m",
    "pytest",
    "tests",
    "-v"
]

execution = subprocess.run(
    commande,
    cwd=RACINE,
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace"
)

sortie_tests = execution.stdout + "\n" + execution.stderr

correspondance = re.search(
    r"(\d+) passed",
    sortie_tests
)

nombre_reussis = (
    int(correspondance.group(1))
    if correspondance else 0
)

nombre_echecs = 0

for motif in [r"(\d+) failed", r"(\d+) error"]:
    resultat = re.search(motif, sortie_tests)

    if resultat:
        nombre_echecs += int(resultat.group(1))

tests_valides = (
    execution.returncode == 0
    and nombre_reussis > 0
    and nombre_echecs == 0
)

print(f"Tests réussis : {nombre_reussis}")
print(f"Tests en échec ou en erreur : {nombre_echecs}")

print(
    "Tests fonctionnels :",
    "VALIDÉS" if tests_valides else "ÉCHEC"
)


# ==================================================
# 5. VÉRIFICATION DE LA COMPARAISON V1 / V2
# ==================================================

print("\n[2] Vérification de la comparaison V1/V2...")

seuils = criteres["seuils"]

variation = comparaison["variation_probabilite_maximale"]
baisse_recall = comparaison["baisse_recall"]
baisse_f1 = comparaison["baisse_f1_score"]
ecart_rechargement = comparaison["ecart_rechargement"]

controles_comparaison = {

    "variation_probabilites": (
        variation <= seuils["variation_probabilite_max"]
    ),

    "baisse_recall": (
        baisse_recall <= seuils["baisse_recall_max"]
    ),

    "baisse_f1_score": (
        baisse_f1 <= seuils["baisse_f1_max"]
    ),

    "sauvegarde_rechargement": (
        ecart_rechargement <= seuils["ecart_rechargement_max"]
    ),

    "comparaison_initiale": (
        comparaison["non_regression_reussie"] is True
    )
}

for nom, resultat in controles_comparaison.items():

    print(
        f"{nom} :",
        "OK" if resultat else "ÉCHEC"
    )

comparaison_validee = all(controles_comparaison.values())


# ==================================================
# 6. VÉRIFICATION DU CONTRAT TECHNIQUE
# ==================================================

print("\n[3] Vérification du contrat technique...")

contrat_valide = contrat["validation_globale"] is True

for version, informations in contrat["modeles"].items():

    modele_valide = informations["contrat_valide"]

    print(
        f"Modèle {version} :",
        "VALIDÉ" if modele_valide else "ÉCHEC"
    )

    if not modele_valide:
        contrat_valide = False

print(
    "Contrat technique :",
    "VALIDÉ" if contrat_valide else "ÉCHEC"
)


# ==================================================
# 7. DÉCISION GLOBALE
# ==================================================

print("\n[4] Décision globale...")

validation_globale = (
    tests_valides
    and comparaison_validee
    and contrat_valide
)

decision = (
    "NON-RÉGRESSION VALIDÉE"
    if validation_globale
    else "NON-RÉGRESSION NON VALIDÉE"
)

print("\nRÉSULTAT FINAL :", decision)


# ==================================================
# 8. CONSTRUCTION DU RAPPORT JSON
# ==================================================

rapport = {

    "date_generation": datetime.now().isoformat(timespec="seconds"),

    "versions": {
        "reference": comparaison["version_reference"],
        "cible": comparaison["version_cible"]
    },

    "tests_fonctionnels": {
        "nombre_reussis": nombre_reussis,
        "nombre_echecs": nombre_echecs,
        "code_retour_pytest": execution.returncode,
        "validation": tests_valides
    },

    "comparaison_v1_v2": {
        "metriques": comparaison["comparaison_metriques"],
        "variation_probabilite_maximale": variation,
        "baisse_recall": baisse_recall,
        "baisse_f1_score": baisse_f1,
        "ecart_rechargement": ecart_rechargement,
        "seuils": seuils,
        "controles": controles_comparaison,
        "validation": comparaison_validee
    },

    "contrat_technique": contrat,

    "decision_finale": {
        "non_regression_validee": validation_globale,
        "decision": decision
    }
}


# ==================================================
# 9. ENREGISTREMENT DU RAPPORT JSON
# ==================================================

with open(
    FICHIER_RAPPORT_JSON,
    "w",
    encoding="utf-8"
) as fichier:

    json.dump(
        rapport,
        fichier,
        indent=4,
        ensure_ascii=False
    )


# ==================================================
# 10. CRÉATION DU RAPPORT TEXTE
# ==================================================

contenu = f"""
==================================================
RAPPORT GLOBAL DE NON-RÉGRESSION
==================================================

Date de génération : {rapport["date_generation"]}

Version de référence : V1
Version cible : V2

--------------------------------------------------
1. TESTS FONCTIONNELS
--------------------------------------------------

Nombre de tests réussis : {nombre_reussis}
Nombre d'échecs ou erreurs : {nombre_echecs}

Résultat :
{"VALIDÉ" if tests_valides else "NON VALIDÉ"}

--------------------------------------------------
2. COMPARAISON V1 / V2
--------------------------------------------------

Variation maximale des probabilités : {variation:.8f}
Baisse du Recall : {baisse_recall:.8f}
Baisse du F1-score : {baisse_f1:.8f}
Écart après rechargement : {ecart_rechargement:.8f}

Résultat :
{"VALIDÉ" if comparaison_validee else "NON VALIDÉ"}

--------------------------------------------------
3. CONTRAT TECHNIQUE
--------------------------------------------------

V1 : {"VALIDÉ" if contrat["modeles"]["V1"]["contrat_valide"] else "NON VALIDÉ"}
V2 : {"VALIDÉ" if contrat["modeles"]["V2"]["contrat_valide"] else "NON VALIDÉ"}

Résultat global :
{"VALIDÉ" if contrat_valide else "NON VALIDÉ"}

--------------------------------------------------
4. DÉCISION FINALE
--------------------------------------------------

{decision}

==================================================
FIN DU RAPPORT
==================================================
"""

with open(
    FICHIER_RAPPORT_TXT,
    "w",
    encoding="utf-8"
) as fichier:

    fichier.write(contenu)


# ==================================================
# 11. AFFICHAGE FINAL
# ==================================================

print("\n============================================")
print("RAPPORTS GÉNÉRÉS")
print("============================================")

print(f"Rapport JSON : {FICHIER_RAPPORT_JSON}")
print(f"Rapport texte : {FICHIER_RAPPORT_TXT}")