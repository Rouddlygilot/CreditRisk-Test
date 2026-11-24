import requests
import time
import json
import csv
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# Adresse de notre API
URL = "http://127.0.0.1:8000/predict"

# Données utilisées pour chaque prédiction
DONNEES = {
    "age": 35,
    "revenu_mensuel": 4500,
    "montant_credit": 15000,
    "duree_mois": 36,
    "anciennete_activite": 5
}


def envoyer_requete():
    """Envoie une requête à l'API et mesure sa durée."""

    debut = time.perf_counter()

    try:
        reponse = requests.post(
            URL,
            json=DONNEES,
            timeout=10
        )

        duree = (time.perf_counter() - debut) * 1000

        return {
            "latence_ms": duree,
            "statut": reponse.status_code,
            "succes": reponse.status_code == 200
        }

    except requests.RequestException:
        duree = (time.perf_counter() - debut) * 1000

        return {
            "latence_ms": duree,
            "statut": "ERREUR",
            "succes": False
        }


def tester_charge(nombre_utilisateurs, nombre_requetes=300):

    print("\n" + "=" * 55)
    print(f"TEST AVEC {nombre_utilisateurs} UTILISATEURS SIMULTANÉS")
    print("=" * 55)

    resultats = []

    debut_test = time.perf_counter()

    with ThreadPoolExecutor(
        max_workers=nombre_utilisateurs
    ) as executor:

        futures = [
            executor.submit(envoyer_requete)
            for _ in range(nombre_requetes)
        ]

        for future in as_completed(futures):
            resultats.append(future.result())

    duree_totale = time.perf_counter() - debut_test

    latences = [
        r["latence_ms"]
        for r in resultats
    ]

    reussites = sum(
        1 for r in resultats if r["succes"]
    )

    erreurs = nombre_requetes - reussites

    latences_triees = sorted(latences)

    p95 = latences_triees[
        int(0.95 * (len(latences_triees) - 1))
    ]

    moyenne = statistics.mean(latences)
    mediane = statistics.median(latences)

    debit = reussites / duree_totale

    taux_erreur = erreurs / nombre_requetes * 100

    mesures = {
        "utilisateurs_simultanes": nombre_utilisateurs,
        "requetes_prevues": nombre_requetes,
        "requetes_reussies": reussites,
        "erreurs": erreurs,
        "duree_totale_secondes": round(duree_totale, 4),
        "latence_moyenne_ms": round(moyenne, 4),
        "latence_mediane_ms": round(mediane, 4),
        "latence_p95_ms": round(p95, 4),
        "latence_maximale_ms": round(max(latences), 4),
        "debit_requetes_seconde": round(debit, 2),
        "taux_erreur_pourcentage": round(taux_erreur, 2)
    }

    for cle, valeur in mesures.items():
        print(f"{cle} : {valeur}")

    return mesures


# Vérification préalable de la disponibilité de l'API
try:
    verification = requests.get(
        "http://127.0.0.1:8000/health",
        timeout=5
    )
    verification.raise_for_status()

except requests.RequestException:
    print("ERREUR : l'API n'est pas disponible.")
    print("Vérifie que le serveur Uvicorn est démarré.")
    raise SystemExit(1)


# Tests progressifs
resultats_finaux = []

for utilisateurs in [5, 10, 20]:

    mesures = tester_charge(
        nombre_utilisateurs=utilisateurs,
        nombre_requetes=300
    )

    resultats_finaux.append(mesures)


# Création du dossier de résultats
dossier = Path("reports")
dossier.mkdir(exist_ok=True)


# Enregistrement JSON
with open(
    dossier / "resultats_charge_api.json",
    "w",
    encoding="utf-8"
) as fichier:

    json.dump(
        resultats_finaux,
        fichier,
        indent=4,
        ensure_ascii=False
    )


# Enregistrement CSV
with open(
    dossier / "resultats_charge_api.csv",
    "w",
    newline="",
    encoding="utf-8"
) as fichier:

    colonnes = resultats_finaux[0].keys()

    writer = csv.DictWriter(
        fichier,
        fieldnames=colonnes
    )

    writer.writeheader()
    writer.writerows(resultats_finaux)


print("\n" + "=" * 55)
print("TOUS LES TESTS DE CHARGE SONT TERMINÉS.")
print("Fichiers générés :")
print("reports/resultats_charge_api.json")
print("reports/resultats_charge_api.csv")
print("=" * 55)