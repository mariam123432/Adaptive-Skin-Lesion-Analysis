# Choix du prétraitement approuvé après audit

Objectif : deux boutons exclusifs dans Streamlit, Shades of Gray par défaut,
et une comparaison appariée sur les identifiants sélectionnés.

Contraintes : mêmes modèles 2021, aucun poids modifié, aucun entraînement,
aucun double passage Shades of Gray, aucune modification des modules C1–C5.
Politique approuvée : C1 automatique, détections C2–C5 signalées ; corrections
expérimentales désactivées et aucune correction C5 prétendue.

- [x] Tester la préparation des images et l'appel des vraies fonctions adaptatives.
- [x] Créer un adaptateur isolé avec chargement strict des versions sklearn.
- [x] Brancher le choix dans les deux pipelines d'inférence sans changer les modèles.
- [x] Ajouter les deux boutons, l'invalidation et les informations affichées/exportées.
- [x] Réutiliser les ensembles selected_images.csv et comparer les méthodes sur le même lot.
- [x] Tester imports, compilation, régression et navigation Streamlit.
- [x] Tester les inférences réelles possibles ; documenter les blocages de version.

Validation : 39 tests réussis, 1 intégration ignorée ; imports et compilation OK.
Navigation AppTest validée sur les deux méthodes et les deux tâches (30/35 IDs).
Régression réelle Shades : mêmes cartes de probabilités sur ISIC_0000034.jpg.
Détecteurs C3/C4 exécutés sur cette image. Adaptatif complet et diagnostic bloqués
explicitement par sklearn 1.6.1 exporté contre 1.9.1 installé, sans prétendre à une
inférence adaptative réelle. Aucun poids ou module C1–C5 d'origine modifié.
