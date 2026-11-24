import csv
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# 1. Chargement des résultats
# --------------------------------------------------

fichier_csv = Path("reports/resultats_charge_api.csv")

with open(fichier_csv, "r", encoding="utf-8") as fichier:
    lecteur = csv.DictReader(fichier)
    donnees = list(lecteur)

# Extraction des données
utilisateurs = [
    int(ligne["utilisateurs_simultanes"])
    for ligne in donnees
]

latence_moyenne = [
    float(ligne["latence_moyenne_ms"])
    for ligne in donnees
]

latence_p95 = [
    float(ligne["latence_p95_ms"])
    for ligne in donnees
]

debit = [
    float(ligne["debit_requetes_seconde"])
    for ligne in donnees
]

# Création du dossier de sortie
dossier = Path("reports")
dossier.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Graphique : évolution de la latence
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    utilisateurs,
    latence_moyenne,
    marker="o",
    label="Latence moyenne"
)

plt.plot(
    utilisateurs,
    latence_p95,
    marker="s",
    label="Latence P95"
)

plt.title("Évolution de la latence selon la charge")
plt.xlabel("Nombre d'utilisateurs simultanés")
plt.ylabel("Latence (ms)")
plt.xticks(utilisateurs)
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    dossier / "graphique_evolution_latence.png",
    dpi=300
)

plt.show()
plt.close()


# --------------------------------------------------
# 3. Graphique : évolution du débit
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    utilisateurs,
    debit,
    marker="o"
)

plt.title("Évolution du débit selon la charge")
plt.xlabel("Nombre d'utilisateurs simultanés")
plt.ylabel("Débit (requêtes/seconde)")
plt.xticks(utilisateurs)
plt.grid(True)

plt.tight_layout()

plt.savefig(
    dossier / "graphique_evolution_debit.png",
    dpi=300
)

plt.show()
plt.close()


# --------------------------------------------------
# 4. Confirmation
# --------------------------------------------------

print("\nGRAPHIQUES GÉNÉRÉS AVEC SUCCÈS")
print("--------------------------------")
print("reports/graphique_evolution_latence.png")
print("reports/graphique_evolution_debit.png")