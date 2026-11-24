# -*- coding: utf-8 -*-

"""
Mesure de la latence d'inférence du modèle CreditRisk.

Ce script mesure le temps nécessaire au modèle
pour produire une prédiction, sans inclure le temps HTTP.
"""

import json
import time
import sys
from pathlib import Path

import numpy as np

# Ajouter la racine du projet au chemin Python
RACINE = Path(__file__).resolve().parents[1]

if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

# Importer le modèle utilisé par l'API
from app.main import model


# ============================================================
# 1. Vérification du modèle
# ============================================================

if model is None:
    raise RuntimeError("Le modèle n'est pas disponible.")

print("=" * 55)
print("MESURE DE LA LATENCE D'INFERENCE")
print("=" * 55)

print("\nModèle utilisé : V1")
print("Nombre de variables : 5")


# ============================================================
# 2. Préparation d'une demande de crédit
# ============================================================

donnees = np.array([[
    35,         # Âge
    350000,     # Revenu mensuel
    1500000,    # Montant du crédit
    36,         # Durée en mois
    5           # Ancienneté de l'activité
]])

print("\nDimensions des données :", donnees.shape)


# ============================================================
# 3. Phase d'échauffement
# ============================================================

# Les premières prédictions peuvent être plus lentes.
# Nous les excluons donc des mesures.

nombre_echauffement = 100

for _ in range(nombre_echauffement):
    model.predict_proba(donnees)

print("\nÉchauffement terminé :", nombre_echauffement,
      "prédictions")


# ============================================================
# 4. Mesure des temps d'inférence
# ============================================================

nombre_mesures = 1000

temps_inference = []

for _ in range(nombre_mesures):

    debut = time.perf_counter_ns()

    model.predict_proba(donnees)

    fin = time.perf_counter_ns()

    duree_ms = (fin - debut) / 1_000_000

    temps_inference.append(duree_ms)


# Conversion en tableau NumPy
temps_inference = np.array(temps_inference)


# ============================================================
# 5. Calcul des indicateurs de performance
# ============================================================

latence_moyenne = float(np.mean(temps_inference))

latence_mediane = float(np.median(temps_inference))

latence_p95 = float(np.percentile(temps_inference, 95))

latence_p99 = float(np.percentile(temps_inference, 99))

latence_min = float(np.min(temps_inference))

latence_max = float(np.max(temps_inference))

debit = 1000 / latence_moyenne


# ============================================================
# 6. Affichage des résultats
# ============================================================

print("\n" + "=" * 55)
print("RÉSULTATS DES MESURES")
print("=" * 55)

print(f"Nombre de mesures : {nombre_mesures}")

print(f"Latence moyenne : {latence_moyenne:.6f} ms")

print(f"Latence médiane : {latence_mediane:.6f} ms")

print(f"Latence P95 : {latence_p95:.6f} ms")

print(f"Latence P99 : {latence_p99:.6f} ms")

print(f"Latence minimale : {latence_min:.6f} ms")

print(f"Latence maximale : {latence_max:.6f} ms")

print(f"Débit théorique : {debit:.2f} prédictions/seconde")


# ============================================================
# 7. Sauvegarde des résultats
# ============================================================

resultats = {
    "activite": "Tests de performance",
    "mesure": "Latence d'inférence",
    "version_modele": "V1",
    "nombre_mesures": nombre_mesures,
    "nombre_echauffement": nombre_echauffement,
    "latence_moyenne_ms": latence_moyenne,
    "latence_mediane_ms": latence_mediane,
    "latence_p95_ms": latence_p95,
    "latence_p99_ms": latence_p99,
    "latence_min_ms": latence_min,
    "latence_max_ms": latence_max,
    "debit_theorique_predictions_seconde": debit
}

dossier_resultats = RACINE / "reports"

fichier_json = dossier_resultats / "resultats_inference.json"

with open(fichier_json, "w", encoding="utf-8") as fichier:
    json.dump(
        resultats,
        fichier,
        ensure_ascii=False,
        indent=4
    )


# Sauvegarder également les 1000 mesures individuelles
fichier_csv = dossier_resultats / "latences_inference.csv"

np.savetxt(
    fichier_csv,
    temps_inference,
    delimiter=",",
    header="latence_ms",
    comments="",
    fmt="%.8f"
)


print("\nFichiers générés :")
print(f"- {fichier_json}")
print(f"- {fichier_csv}")

print("\nMESURE TERMINÉE")