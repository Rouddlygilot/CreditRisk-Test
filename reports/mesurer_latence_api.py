# -*- coding: utf-8 -*-

"""
Mesure de la latence HTTP de l'API CreditRisk.
"""

import json
import time
from pathlib import Path

import numpy as np
import requests


# ============================================================
# 1. Configuration
# ============================================================

URL_API = "http://127.0.0.1:8000/predict"

NOMBRE_ECHAUFFEMENT = 50
NOMBRE_REQUETES = 1000

DONNEES_CREDIT = {
    "age": 35,
    "revenu_mensuel": 350000,
    "montant_credit": 1500000,
    "duree_mois": 36,
    "anciennete_activite": 5
}


# ============================================================
# 2. Vérification de disponibilité
# ============================================================

print("=" * 55)
print("MESURE DE LA LATENCE HTTP DE L'API")
print("=" * 55)

reponse = requests.get(
    "http://127.0.0.1:8000/health",
    timeout=10
)

reponse.raise_for_status()

print("\nAPI disponible :", reponse.json())


# ============================================================
# 3. Phase d'échauffement
# ============================================================

print("\nPhase d'échauffement...")

for _ in range(NOMBRE_ECHAUFFEMENT):

    reponse = requests.post(
        URL_API,
        json=DONNEES_CREDIT,
        timeout=10
    )

    reponse.raise_for_status()

print("Échauffement terminé :", NOMBRE_ECHAUFFEMENT,
      "requêtes")


# ============================================================
# 4. Mesure des temps de réponse
# ============================================================

temps_reponses = []
nombre_erreurs = 0

debut_global = time.perf_counter()

for i in range(NOMBRE_REQUETES):

    debut = time.perf_counter()

    try:

        reponse = requests.post(
            URL_API,
            json=DONNEES_CREDIT,
            timeout=10
        )

        fin = time.perf_counter()

        if reponse.status_code == 200:

            duree_ms = (fin - debut) * 1000

            temps_reponses.append(duree_ms)

        else:

            nombre_erreurs += 1

    except requests.RequestException:

        nombre_erreurs += 1


fin_global = time.perf_counter()

duree_totale = fin_global - debut_global


# ============================================================
# 5. Calcul des indicateurs
# ============================================================

temps_reponses = np.array(temps_reponses)

if len(temps_reponses) == 0:
    raise RuntimeError("Aucune requête réussie.")

latence_moyenne = float(np.mean(temps_reponses))

latence_mediane = float(np.median(temps_reponses))

latence_p95 = float(np.percentile(temps_reponses, 95))

latence_p99 = float(np.percentile(temps_reponses, 99))

latence_min = float(np.min(temps_reponses))

latence_max = float(np.max(temps_reponses))

debit = len(temps_reponses) / duree_totale

taux_erreur = nombre_erreurs / NOMBRE_REQUETES * 100


# ============================================================
# 6. Affichage des résultats
# ============================================================

print("\n" + "=" * 55)
print("RÉSULTATS DES MESURES HTTP")
print("=" * 55)

print("Nombre de requêtes prévues :", NOMBRE_REQUETES)

print("Requêtes réussies :", len(temps_reponses))

print("Requêtes en erreur :", nombre_erreurs)

print(f"Durée totale : {duree_totale:.4f} secondes")

print(f"Latence moyenne : {latence_moyenne:.4f} ms")

print(f"Latence médiane : {latence_mediane:.4f} ms")

print(f"Latence P95 : {latence_p95:.4f} ms")

print(f"Latence P99 : {latence_p99:.4f} ms")

print(f"Latence minimale : {latence_min:.4f} ms")

print(f"Latence maximale : {latence_max:.4f} ms")

print(f"Débit réel séquentiel : {debit:.2f} requêtes/seconde")

print(f"Taux d'erreur : {taux_erreur:.2f} %")


# ============================================================
# 7. Sauvegarde des résultats
# ============================================================

resultats = {
    "activite": "Tests de performance",
    "mesure": "Latence HTTP de l'API",
    "version_modele": "V1",
    "nombre_requetes_prevues": NOMBRE_REQUETES,
    "nombre_requetes_reussies": len(temps_reponses),
    "nombre_erreurs": nombre_erreurs,
    "duree_totale_secondes": duree_totale,
    "latence_moyenne_ms": latence_moyenne,
    "latence_mediane_ms": latence_mediane,
    "latence_p95_ms": latence_p95,
    "latence_p99_ms": latence_p99,
    "latence_min_ms": latence_min,
    "latence_max_ms": latence_max,
    "debit_requetes_seconde": debit,
    "taux_erreur_pourcent": taux_erreur
}

RACINE = Path(__file__).resolve().parents[1]
DOSSIER = RACINE / "reports"

with open(
    DOSSIER / "resultats_latence_api.json",
    "w",
    encoding="utf-8"
) as fichier:

    json.dump(
        resultats,
        fichier,
        ensure_ascii=False,
        indent=4
    )

np.savetxt(
    DOSSIER / "latences_api.csv",
    temps_reponses,
    delimiter=",",
    header="latence_ms",
    comments="",
    fmt="%.8f"
)

print("\nFichiers générés :")
print("- reports/resultats_latence_api.json")
print("- reports/latences_api.csv")

print("\nMESURE TERMINÉE")