import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

import app.main as api

client = TestClient(api.app)

donnees_valides = {
    "age": 35,
    "revenu_mensuel": 250000,
    "montant_credit": 1000000,
    "duree_mois": 12,
    "anciennete_activite": 4
}

scenarios = [
    (
        "1. Requête valide",
        donnees_valides
    ),
    (
        "2. Champ obligatoire absent",
        {k: v for k, v in donnees_valides.items() if k != "age"}
    ),
    (
        "3. Âge négatif",
        {**donnees_valides, "age": -5}
    ),
    (
        "4. Revenu sous forme de texte",
        {**donnees_valides, "revenu_mensuel": "250000 FCFA"}
    ),
    (
        "5. Montant du crédit nul",
        {**donnees_valides, "montant_credit": 0}
    ),
    (
        "6. Durée non entière",
        {**donnees_valides, "duree_mois": 12.5}
    ),
    (
        "7. Champ supplémentaire",
        {**donnees_valides, "adresse": "Dakar"}
    ),
    (
        "8. Valeurs extrêmes",
        {
            **donnees_valides,
            "age": 150,
            "revenu_mensuel": 200_000_000,
            "montant_credit": 200_000_000,
            "duree_mois": 500,
            "anciennete_activite": 100
        }
    )
]

print("\n========== INSPECTION DES RÉPONSES API ==========\n")

for nom, donnees in scenarios:
    response = client.post("/predict", json=donnees)

    print(nom)
    print("Code HTTP :", response.status_code)
    print("Réponse JSON :", response.json())
    print("-" * 65)

# Scénario 9 : modèle indisponible
print("9. Modèle indisponible")

ancien_modele = api.model

try:
    api.model = None
    response = client.post("/predict", json=donnees_valides)

    print("Code HTTP :", response.status_code)
    print("Réponse JSON :", response.json())

finally:
    api.model = ancien_modele

print("-" * 65)

# Scénario 10 : requête mal formée
print("10. Requête mal formée")

response = client.post(
    "/predict",
    content='{"age": 35, "revenu_mensuel": ',
    headers={"Content-Type": "application/json"}
)

print("Code HTTP :", response.status_code)
print("Réponse JSON :", response.json())

print("\n========== FIN DE L'INSPECTION ==========\n")