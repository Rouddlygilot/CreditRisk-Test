from fastapi.testclient import TestClient

from app.main import app

# ============================================================
# 1. INITIALISATION DU CLIENT DE TEST
# ============================================================

client = TestClient(app)


# ============================================================
# 2. DONNEES VALIDES UTILISEES POUR LES TESTS
# ============================================================

donnees_valides = {
    "age": 35,
    "revenu_mensuel": 250000,
    "montant_credit": 1000000,
    "duree_mois": 12,
    "anciennete_activite": 4
}


# ============================================================
# TEST COMPLEMENTAIRE 1 : ENDPOINT /health
# ============================================================

def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    resultat = response.json()

    assert resultat["status"] == "ok"
    assert resultat["model_loaded"] is True


# ============================================================
# TEST COMPLEMENTAIRE 2 : ENDPOINT /model-info
# ============================================================

def test_model_info():

    response = client.get("/model-info")

    assert response.status_code == 200

    resultat = response.json()

    assert resultat["nom"] == "CreditRisk"
    assert resultat["version"] == "1.0.0"
    assert resultat["model_loaded"] is True


# ============================================================
# SCENARIO 1 : REQUETE VALIDE
# ============================================================

def test_01_requete_valide():

    response = client.post(
        "/predict",
        json=donnees_valides
    )

    assert response.status_code == 200

    resultat = response.json()

    assert "probabilite_defaut" in resultat

    assert 0 <= resultat["probabilite_defaut"] <= 1

    assert resultat["decision"] in [
        "Risque élevé",
        "Risque faible"
    ]

    assert resultat["version_modele"] == "1.0.0"


# ============================================================
# SCENARIO 2 : CHAMP OBLIGATOIRE ABSENT
# ============================================================

def test_02_champ_obligatoire_absent():

    donnees = donnees_valides.copy()

    del donnees["age"]

    response = client.post(
        "/predict",
        json=donnees
    )

    assert response.status_code == 422

    resultat = response.json()

    assert "detail" in resultat


# ============================================================
# SCENARIO 3 : AGE NEGATIF
# ============================================================

def test_03_age_negatif():

    donnees = donnees_valides.copy()

    donnees["age"] = -5

    response = client.post(
        "/predict",
        json=donnees
    )

    assert response.status_code == 422

    assert "detail" in response.json()


# ============================================================
# SCENARIO 4 : REVENU SOUS FORME DE TEXTE
# ============================================================

def test_04_revenu_texte():

    donnees = donnees_valides.copy()

    donnees["revenu_mensuel"] = "250000 FCFA"

    response = client.post(
        "/predict",
        json=donnees
    )

    assert response.status_code == 422

    assert "detail" in response.json()


# ============================================================
# SCENARIO 5 : MONTANT NUL OU NEGATIF
# ============================================================

def test_05_montant_invalide():

    montants_invalides = [0, -100000]

    for montant in montants_invalides:

        donnees = donnees_valides.copy()

        donnees["montant_credit"] = montant

        response = client.post(
            "/predict",
            json=donnees
        )

        assert response.status_code == 422

        assert "detail" in response.json()


# ============================================================
# SCENARIO 6 : DUREE NON ENTIERE
# ============================================================

def test_06_duree_non_entiere():

    donnees = donnees_valides.copy()

    donnees["duree_mois"] = 12.5

    response = client.post(
        "/predict",
        json=donnees
    )

    assert response.status_code == 422

    assert "detail" in response.json()


# ============================================================
# SCENARIO 7 : CHAMP SUPPLEMENTAIRE NON AUTORISE
# ============================================================

def test_07_champ_supplementaire():

    donnees = donnees_valides.copy()

    donnees["adresse"] = "Dakar"

    response = client.post(
        "/predict",
        json=donnees
    )

    assert response.status_code == 422

    assert "detail" in response.json()


# ============================================================
# SCENARIO 8 : VALEURS EXTREMEMENT ELEVEES
# ============================================================

def test_08_valeurs_extremes():

    donnees = donnees_valides.copy()

    donnees["age"] = 150
    donnees["revenu_mensuel"] = 200_000_000
    donnees["montant_credit"] = 200_000_000
    donnees["duree_mois"] = 500
    donnees["anciennete_activite"] = 100

    response = client.post(
        "/predict",
        json=donnees
    )

    assert response.status_code == 422

    assert "detail" in response.json()


# ============================================================
# SCENARIO 9 : MODELE INDISPONIBLE
# ============================================================

def test_09_modele_indisponible(monkeypatch):

    # Simuler l'absence du modèle
    monkeypatch.setattr(
        "app.main.model",
        None
    )

    response = client.post(
        "/predict",
        json=donnees_valides
    )

    assert response.status_code == 503

    resultat = response.json()

    assert resultat["detail"] == (
        "Modèle de prédiction indisponible"
    )


# ============================================================
# SCENARIO 10 : REQUETE MAL FORMEE
# ============================================================

def test_10_requete_mal_formee():

    response = client.post(
        "/predict",
        content='{"age": 35, "revenu_mensuel": ',
        headers={
            "Content-Type": "application/json"
        }
    )

    assert response.status_code == 422

    assert "detail" in response.json()