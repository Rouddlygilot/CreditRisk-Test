import math

import pytest
from fastapi.testclient import TestClient

from app.main import app

# Création du client permettant de tester l'API
client = TestClient(app)

# Jeu de données valide utilisé comme référence
donnees_valides = {
    "age": 35,
    "revenu_mensuel": 250000,
    "montant_credit": 1000000,
    "duree_mois": 12,
    "anciennete_activite": 4
}


# ---------------------------------------------------------
# TEST 1 : Vérification du contrat de sortie du modèle
# ---------------------------------------------------------

def test_probabilite_dans_intervalle():
    response = client.post("/predict", json=donnees_valides)

    assert response.status_code == 200

    resultat = response.json()
    probabilite = resultat["probabilite_defaut"]

    # La probabilité doit être un nombre fini compris entre 0 et 1
    assert math.isfinite(probabilite)
    assert 0.0 <= probabilite <= 1.0


# ---------------------------------------------------------
# TEST 2 : Valeurs manquantes
# ---------------------------------------------------------

def test_champ_manquant():
    donnees = donnees_valides.copy()
    del donnees["age"]

    response = client.post("/predict", json=donnees)

    assert response.status_code == 422
    assert "detail" in response.json()


def test_valeur_null():
    donnees = donnees_valides.copy()
    donnees["revenu_mensuel"] = None

    response = client.post("/predict", json=donnees)

    assert response.status_code == 422
    assert "detail" in response.json()


# ---------------------------------------------------------
# TEST 3 : Valeurs non numériques
# ---------------------------------------------------------

def test_age_non_numerique():
    donnees = donnees_valides.copy()
    donnees["age"] = "trente-cinq"

    response = client.post("/predict", json=donnees)

    assert response.status_code == 422
    assert "detail" in response.json()


def test_montant_non_numerique():
    donnees = donnees_valides.copy()
    donnees["montant_credit"] = "un million"

    response = client.post("/predict", json=donnees)

    assert response.status_code == 422
    assert "detail" in response.json()


# ---------------------------------------------------------
# TEST 4 : Valeurs infinies et non finies
# ---------------------------------------------------------

# ---------------------------------------------------------
# TEST 4 : Valeurs infinies et non finies
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "valeur",
    ["Infinity", "-Infinity", "NaN"]
)
def test_valeurs_non_finies(valeur):

    # Construction manuelle du contenu JSON
    contenu = (
        '{"age": 35,'
        '"revenu_mensuel": ' + valeur + ','
        '"montant_credit": 1000000,'
        '"duree_mois": 12,'
        '"anciennete_activite": 4}'
    )

    # Envoi du contenu directement à l'API
    response = client.post(
        "/predict",
        content=contenu,
        headers={"Content-Type": "application/json"}
    )

    # L'API doit rejeter les valeurs non finies
    assert response.status_code == 422
    assert "detail" in response.json()


# ---------------------------------------------------------
# TEST 5 : Valeurs aux limites autorisées
# ---------------------------------------------------------

def test_limites_minimales():
    donnees = {
        "age": 18,
        "revenu_mensuel": 1,
        "montant_credit": 1,
        "duree_mois": 1,
        "anciennete_activite": 0
    }

    response = client.post("/predict", json=donnees)

    assert response.status_code == 200

    probabilite = response.json()["probabilite_defaut"]

    assert math.isfinite(probabilite)
    assert 0 <= probabilite <= 1


def test_limites_maximales():
    donnees = {
        "age": 100,
        "revenu_mensuel": 100_000_000,
        "montant_credit": 100_000_000,
        "duree_mois": 360,
        "anciennete_activite": 80
    }

    response = client.post("/predict", json=donnees)

    assert response.status_code == 200

    probabilite = response.json()["probabilite_defaut"]

    assert math.isfinite(probabilite)
    assert 0 <= probabilite <= 1


# ---------------------------------------------------------
# TEST 6 : Entrée inhabituelle mais valide
# ---------------------------------------------------------

def test_entree_tres_eloignee_des_donnees_habituelles():
    donnees = {
        "age": 18,
        "revenu_mensuel": 1000,
        "montant_credit": 100_000_000,
        "duree_mois": 360,
        "anciennete_activite": 0
    }

    response = client.post("/predict", json=donnees)

    # Les valeurs respectent les limites du schéma :
    # l'API doit donc accepter la demande.
    assert response.status_code == 200

    resultat = response.json()
    probabilite = resultat["probabilite_defaut"]

    assert math.isfinite(probabilite)
    assert 0 <= probabilite <= 1


# ---------------------------------------------------------
# TEST 7 : Schéma de données incorrect
# ---------------------------------------------------------

def test_schema_incorrect():
    # Une liste est envoyée à la place d'un objet JSON
    response = client.post("/predict", json=[35, 250000, 1000000])

    assert response.status_code == 422
    assert "detail" in response.json()


def test_champ_supplementaire_non_autorise():
    donnees = donnees_valides.copy()
    donnees["adresse"] = "Dakar"

    response = client.post("/predict", json=donnees)

    assert response.status_code == 422
    assert "detail" in response.json()