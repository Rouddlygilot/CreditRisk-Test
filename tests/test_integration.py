import numpy as np
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

import app.main as main_module

client = TestClient(main_module.app)

donnees_valides = {
    "age": 35,
    "revenu_mensuel": 250000,
    "montant_credit": 1000000,
    "duree_mois": 12,
    "anciennete_activite": 4
}

def test_01_pipeline_complet():

    # Envoyer une requête à l'API
    response = client.post(
        "/predict",
        json=donnees_valides
    )

    # Vérifier le code HTTP
    assert response.status_code == 200

    resultat = response.json()

    # Vérifier la structure de la réponse
    assert "probabilite_defaut" in resultat
    assert "decision" in resultat
    assert "version_modele" in resultat

    # Vérifier la probabilité
    assert 0 <= resultat["probabilite_defaut"] <= 1

    # Vérifier la décision
    assert resultat["decision"] in [
        "Risque élevé",
        "Risque faible"
    ]

    # Vérifier que le modèle utilise un pipeline
    assert isinstance(main_module.model, Pipeline)

    # Vérifier les étapes du pipeline
    assert "scaler" in main_module.model.named_steps
    assert "classifier" in main_module.model.named_steps

    # Vérifier les métadonnées
    infos = client.get("/model-info").json()

    assert resultat["version_modele"] == infos["version"]
    assert infos["model_loaded"] is True

def test_02_modele_non_charge(monkeypatch):

    # Simuler un modèle indisponible
    monkeypatch.setattr(
        main_module,
        "model",
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
def test_03_variable_absente():

    donnees = donnees_valides.copy()

    del donnees["revenu_mensuel"]

    response = client.post(
        "/predict",
        json=donnees
    )

    assert response.status_code == 422

    resultat = response.json()

    assert "detail" in resultat

def test_04_erreur_pretraitement(monkeypatch):

    def provoquer_erreur(*args, **kwargs):
        raise RuntimeError("Erreur technique simulée")

    # Remplacer temporairement la transformation du scaler
    monkeypatch.setattr(
        main_module.model.named_steps["scaler"],
        "transform",
        provoquer_erreur
    )

    response = client.post(
        "/predict",
        json=donnees_valides
    )

    assert response.status_code == 500

    resultat = response.json()

    assert resultat["detail"] == (
        "Une erreur interne est survenue pendant la prédiction"
    )

    # Vérifier que le détail technique n'est pas exposé
    assert "Erreur technique simulée" not in str(resultat)

def test_05_dimensions_incompatibles(monkeypatch):

    # Simuler un tableau contenant seulement quatre variables
    monkeypatch.setattr(
        main_module.np,
        "array",
        lambda *args, **kwargs: np.zeros((1, 4))
    )

    response = client.post(
        "/predict",
        json=donnees_valides
    )

    assert response.status_code == 500

    resultat = response.json()

    assert resultat["detail"] == (
        "Une erreur interne est survenue pendant la prédiction"
    )

def test_06_erreur_interne_prediction(monkeypatch):

    def provoquer_erreur(*args, **kwargs):
        raise RuntimeError("Erreur interne simulée")

    # Simuler une erreur du classificateur
    monkeypatch.setattr(
        main_module.model.named_steps["classifier"],
        "predict_proba",
        provoquer_erreur
    )

    response = client.post(
        "/predict",
        json=donnees_valides
    )

    assert response.status_code == 500

    resultat = response.json()

    assert resultat["detail"] == (
        "Une erreur interne est survenue pendant la prédiction"
    )

    # Vérifier l'absence de détails techniques
    assert "Erreur interne simulée" not in str(resultat)

