# CreditRisk-Test



## 1. Présentation du projet



**CreditRisk-Test** est un projet consacré aux tests logiciels et à l'assurance qualité d'un modèle d'intelligence artificielle dédié à la prédiction du risque de crédit.



Le projet vise à évaluer la fiabilité, la robustesse, le fonctionnement et les performances d'une API de prédiction.



## 2. Technologies utilisées



- Python

- FastAPI

- Scikit-learn

- NumPy

- Pydantic

- Pytest

- Pytest-cov

- Ruff

- Uvicorn

- Matplotlib

- Pandas



## 3. Structure du projet



```text

CreditRisk-Test/

├── app/

│   ├── main.py

│   ├── validation.py

│   ├── preprocessing.py

│   └── prediction.py

├── tests/

│   ├── test_api.py

│   ├── test_integration.py

│   ├── test_robustesse.py

│   ├── test_validation.py

│   └── test_regression.py

├── reports/

├── .github/

│   └── workflows/

│       └── tests.yml

├── requirements.txt

├── .gitignore

└── README.md

```



## 4. Installation



### 4.1 Cloner le dépôt



```bash

git clone URL_DU_DEPOT

cd CreditRisk-Test

```



### 4.2 Créer un environnement virtuel



```bash

python -m venv .venv

```



### 4.3 Activer l'environnement sous Windows



```powershell

.venv\\Scripts\\Activate.ps1

```



### 4.4 Installer les dépendances



```bash

python -m pip install -r requirements.txt

```



## 5. Exécution de l'API



### 5.1 Démarrer le serveur



```bash

python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

```



### 5.2 Documentation interactive



La documentation interactive de l'API est accessible à l'adresse suivante :



http://127.0.0.1:8000/docs



### 5.3 Principaux endpoints



| Méthode | Endpoint | Description |

|---|---|---|

| GET | `/health` | Vérification de l'état de l'API |

| POST | `/predict` | Prédiction du risque de crédit |

| GET | `/model-info` | Informations sur le modèle |



## 6. Exécution des tests automatisés



### 6.1 Lancer tous les tests



```bash

python -m pytest -v

```



### 6.2 Lancer les tests avec couverture de code



```bash

python -m pytest --cov=app --cov-report=term-missing --cov-report=html

```



Le rapport détaillé de couverture est généré dans le dossier `htmlcov/`.



### 6.3 Consulter le rapport HTML



Ouvrir le fichier :



```text

htmlcov/index.html

```



## 7. Analyse de la qualité du code avec Ruff



### 7.1 Vérifier la qualité du code



```bash

python -m ruff check .

```



### 7.2 Corriger automatiquement les erreurs corrigibles



```bash

python -m ruff check . --fix

```



## 8. Intégration continue avec GitHub Actions



Le projet utilise GitHub Actions pour automatiser les vérifications à chaque modification du dépôt et lors des Pull Requests.



La chaîne d'intégration continue a pour objectifs :



- Installer les dépendances du projet.

- Exécuter les tests automatisés avec Pytest.

- Vérifier la qualité du code avec Ruff.

- Générer un rapport de couverture des tests.

- Détecter automatiquement les échecs de validation.



Cette automatisation permet de vérifier régulièrement la qualité et la fiabilité du projet.



## 9. Mesure des performances



Des scripts présents dans le dossier `reports/` permettent d'évaluer les performances de l'inférence et de l'API.



Les principales mesures réalisées sont :



- La latence moyenne des prédictions.

- Le temps de réponse de l'API.

- Le débit de traitement des requêtes.

- Le comportement de l'API sous charge.

- Le taux d'erreur.



Les résultats et les graphiques sont conservés dans le dossier `reports/`.



## 10. Conclusion



Le projet CreditRisk-Test met en place une démarche d'assurance qualité appliquée à une API de prédiction du risque de crédit.



Il combine les tests fonctionnels, les tests d'intégration, les tests de robustesse, les tests de non-régression et les tests de performance.



L'intégration de GitHub Actions permettra d'automatiser les contrôles et de faciliter la validation du projet à chaque modification du code.

