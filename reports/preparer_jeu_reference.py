import json
import numpy as np
import pandas as pd

from pathlib import Path


# ============================================================
# 1. Définition des chemins
# ============================================================

RACINE = Path(__file__).resolve().parent
DOSSIER = RACINE / "reference_v1"

DOSSIER.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Génération d'un jeu de test indépendant
# ============================================================

rng = np.random.default_rng(2026)

nombre_observations = 300

age = rng.integers(18, 75, nombre_observations)

revenu = rng.uniform(50000, 1000000, nombre_observations)

montant = rng.uniform(100000, 5000000, nombre_observations)

duree = rng.integers(3, 61, nombre_observations)

anciennete = rng.uniform(0, 30, nombre_observations)


# ============================================================
# 3. Construction des variables explicatives
# ============================================================

X_test = pd.DataFrame({

    "age": age,

    "revenu_mensuel": revenu,

    "montant_credit": montant,

    "duree_mois": duree,

    "anciennete_activite": anciennete

})


# ============================================================
# 4. Génération de la cible réelle
# ============================================================

risque = (
    0.8 * montant / revenu
    + 0.015 * duree
    - 0.04 * anciennete
    - 0.01 * age
)

probabilite = 1 / (1 + np.exp(-risque))

y_test = (
    rng.random(nombre_observations) < probabilite
).astype(int)


# ============================================================
# 5. Création du jeu de test
# ============================================================

jeu_test = X_test.copy()

jeu_test["valeur_reelle"] = y_test

fichier_csv = DOSSIER / "jeu_test_commun.csv"

jeu_test.to_csv(
    fichier_csv,
    index=False,
    encoding="utf-8"
)


# ============================================================
# 6. Sauvegarde des informations
# ============================================================

informations = {

    "nombre_observations": int(nombre_observations),

    "nombre_variables": int(X_test.shape[1]),

    "graine_aleatoire": 2026,

    "type_jeu": "Test indépendant",

    "repartition_classes": {

        "classe_0": int(np.sum(y_test == 0)),

        "classe_1": int(np.sum(y_test == 1))

    }

}

with open(
    DOSSIER / "informations_jeu_test.json",
    "w",
    encoding="utf-8"
) as fichier:

    json.dump(
        informations,
        fichier,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# 7. Affichage des résultats
# ============================================================

print("========================================")
print("JEU DE TEST INDÉPENDANT")
print("========================================")

print("Nombre d'observations :", len(X_test))

print("Nombre de variables :", X_test.shape[1])

print("\nRépartition des classes :")

print(pd.Series(y_test).value_counts().sort_index())

print("\nGraine aléatoire : 2026")

print("\nFichier créé :", fichier_csv)

print("\nPréparation terminée avec succès.")