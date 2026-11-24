from typing import Annotated

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. Création et entraînement d'un modèle de test
# ============================================================

def creer_modele():

    rng = np.random.default_rng(42)

    nombre_observations = 1000

    age = rng.integers(18, 75, nombre_observations)

    revenu = rng.uniform(
        50000, 1000000, nombre_observations
    )

    montant = rng.uniform(
        100000, 5000000, nombre_observations
    )

    duree = rng.integers(
        3, 61, nombre_observations
    )

    anciennete = rng.uniform(
        0, 30, nombre_observations
    )

    X = np.column_stack([
        age,
        revenu,
        montant,
        duree,
        anciennete
    ])

    # Création d'une cible synthétique pour les tests

    risque = (
        0.8 * montant / revenu
        + 0.015 * duree
        - 0.04 * anciennete
        - 0.01 * age
    )

    probabilite = 1 / (1 + np.exp(-risque))

    y = (
        rng.random(nombre_observations) < probabilite
    ).astype(int)

    modele = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(random_state=42)
        )
    ])

    modele.fit(X, y)

    return modele


# Chargement du modèle
model = creer_modele()


# ============================================================
# 2. Création de l'application FastAPI
# ============================================================

app = FastAPI(
    title="CreditRisk API",
    description="API de prédiction du risque de défaut de crédit",
    version="1.0.0"
)


# ============================================================
# 3. Gestion personnalisée des erreurs de validation
# ============================================================

@app.exception_handler(RequestValidationError)
async def gerer_erreurs_validation(request, exc):

    erreurs = []

    for erreur in exc.errors():

        erreurs.append({
            "champ": ".".join(
                str(element) for element in erreur["loc"]
            ),
            "message": erreur["msg"],
            "type": erreur["type"]
        })

    return JSONResponse(
        status_code=422,
        content={
            "message": (
                "Les données envoyées ne respectent "
                "pas le schéma attendu."
            ),
            "detail": erreurs
        }
    )


# ============================================================
# 4. Définition des données attendues
# ============================================================

class DemandeCredit(BaseModel):

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False
    )

    age: Annotated[
        StrictInt,
        Field(ge=18, le=100)
    ]

    revenu_mensuel: Annotated[
        StrictInt | StrictFloat,
        Field(gt=0, le=100_000_000)
    ]

    montant_credit: Annotated[
        StrictInt | StrictFloat,
        Field(gt=0, le=100_000_000)
    ]

    duree_mois: Annotated[
        StrictInt,
        Field(ge=1, le=360)
    ]

    anciennete_activite: Annotated[
        StrictInt | StrictFloat,
        Field(ge=0, le=80)
    ]


# ============================================================
# 5. Endpoint de vérification de disponibilité
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": model is not None
    }


# ============================================================
# 6. Endpoint de prédiction
# ============================================================

@app.post("/predict")
def predict(demande: DemandeCredit):

    # Vérifier si le modèle est disponible
    if model is None:

        raise HTTPException(
            status_code=503,
            detail="Modèle de prédiction indisponible"
        )

    try:

        # Préparer les données reçues
        donnees = np.array([[
            demande.age,
            demande.revenu_mensuel,
            demande.montant_credit,
            demande.duree_mois,
            demande.anciennete_activite
        ]])

        # Vérifier les dimensions des données
        if donnees.shape != (1, 5):

            raise ValueError(
                "Dimensions des données incompatibles"
            )

        # Exécuter la prédiction
        probabilite = float(
            model.predict_proba(donnees)[0][1]
        )

        # Vérifier la validité de la probabilité produite
        if not np.isfinite(probabilite):

            raise ValueError(
                "La probabilité calculée n'est pas finie"
            )

        if not 0.0 <= probabilite <= 1.0:

            raise ValueError(
                "La probabilité calculée est hors intervalle"
            )

        # Déterminer la décision
        decision = (
            "Risque élevé"
            if probabilite >= 0.5
            else "Risque faible"
        )

        # Construire la réponse
        return {
            "probabilite_defaut": probabilite,
            "decision": decision,
            "version_modele": "1.0.0"
        }

    except Exception:  # noqa: BLE001

        # Ne pas exposer les détails techniques au client
        raise HTTPException(
            status_code=500,
            detail=(
                "Une erreur interne est survenue "
                "pendant la prédiction"
            )
        ) from None


# ============================================================
# 7. Endpoint des métadonnées du modèle
# ============================================================

@app.get("/model-info")
def model_info():

    return {
        "nom": "CreditRisk",
        "version": "1.0.0",
        "type": "Régression logistique",
        "model_loaded": model is not None
    }