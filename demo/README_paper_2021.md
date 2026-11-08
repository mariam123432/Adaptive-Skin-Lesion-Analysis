# Démonstration locale du pipeline 2021

Cette interface séparée de `demo/app.py` expose l'adaptation expérimentale
existante. Le mode adaptatif appelle les modules C1–C5 sans modifier leur code. Aucun entraînement
ni téléchargement de poids n'est effectué.

## Lancement PowerShell

Depuis la racine `Adaptive-Skin-Lesion-Analysis-ready` :

```powershell
.\.venv\Scripts\python.exe -m streamlit run demo/paper_2021_app.py
```

L'environnement `.venv` inspecté contient torch, torchvision,
segmentation-models-pytorch, numpy, OpenCV, Pillow, scikit-learn, joblib,
Streamlit et pytest. Le Python global n'a pas toutes ces dépendances.
Ne pas remplacer `requirements.txt` ; conserver l'environnement compatible
avec les checkpoints et les objets sklearn exportés.

**Écart détecté lors de la validation :** les objets joblib ont été exportés
avec scikit-learn **1.6.1**, mais `.venv` utilise **1.9.1**. L'application
refuse explicitement cette différence de version. Pour une inférence réelle,
utiliser un environnement avec scikit-learn 1.6.1 et une version de Python
compatible, ainsi que les dépendances scientifiques du projet. Aucun
changement de l'environnement installé n'est effectué automatiquement.

## Fichiers nécessaires

Placer les artefacts **de confiance** exportés par le projet dans
`models/paper_2021/` (les fichiers joblib sont des objets Python sérialisés) :

- `vgg19_unet_best.pth`, `deeplabv3plus_best.pth` ;
- `hybrid_resnet50_best.pth`, `hybrid_densenet201_best.pth` ;
- `handcrafted_scaler.joblib`, `hybrid_final_svm.joblib` ;
- `hybrid_final_svm_config.json`.

Les fichiers historiques d'entraînement ne sont pas nécessaires.
Les checkpoints sont chargés avec `weights_only=True` et `strict=True`,
sous forme de state_dict direct ou de dictionnaire contenant `state_dict`.
Une incompatibilité de clés, de dimensions, de classes ou de configuration
interrompt l'analyse. Aucune prédiction de substitution n'est produite.
Les constructeurs scientifiques acceptent désormais `pretrained=False` pour
éviter de télécharger ImageNet avant de charger un checkpoint complet ;
leur valeur par défaut et leurs calculs sont conservés.

## Images et utilisation

Déposer vos propres images `.jpg`, `.jpeg`, `.png` dans `test_images/`,
ou choisir l'import dans la barre latérale. Le dossier est volontairement vide.
Les images sont vérifiées avant inférence : JPEG/PNG réel, maximum 30 Mo
et 40 millions de pixels. La lecture retourne du RGB uint8 comme celle du projet.
Choisir CPU, CUDA, ou auto (CUDA si disponible). CUDA explicitement demandé
mais indisponible produit une erreur.

Cliquer sur **Analyser l'image** : ce clic seul déclenche le chargement et
l'inférence. `st.cache_resource` partage les modèles entre sessions ; un verrou
sérialise les inférences. Changer d'image ou de device efface l'ancien résultat.
Les téléchargements et sauvegardes réutilisent le résultat en mémoire.

## Deux boutons de prétraitement

Dans la barre latérale, **Méthode de prétraitement** présente exactement deux
boutons : **Shades of Gray (papier 2021)** et **Mon prétraitement adaptatif C1–C5**.
Shades of Gray est sélectionné par défaut ; le bouton actif est mis en évidence.
Cliquer sur un bouton sélectionne une méthode, puis **Analyser l'image** lance
l'inférence. Le choix seul ne charge pas les modèles et ne traite aucune image.
Changer de méthode efface les résultats précédents dans les deux modes de l'interface.

Le chemin de référence est conservé : resize INTER_AREA 224², Shades of Gray
p=6, normalisation ImageNet. Le mode adaptatif appelle le vrai
`src.pipeline.process_image` à la taille native : C1 utilise `HairPredictor`
(SVM RBF configuré) et applique DullRazor/Telea seulement si des poils sont détectés.
Les détecteurs C2, C3, C4 et C5 sont également exécutés sur l'image originale et
leurs décisions sont affichées. Conformément à l'audit approuvé, les corrections
expérimentales C2/C3/C4 restent désactivées et C5 n'a pas de correction disponible.
La comparaison mesure donc cette politique conservatrice, pas cinq corrections
automatiques. Il n'y a aucune simulation d'étape manquante.

Après la correction C1, le résultat conserve les dimensions natives, est
redimensionné à 224² puis normalisé ImageNet, **sans Shades of Gray supplémentaire**.
La même image alimente les segmentateurs, handcrafted200 et les CNN hybrides.
Les checkpoints, le scaler et le SVM diagnostique restent identiques entre les
méthodes. Les classifieurs d'artefacts ne remplacent jamais le SVM diagnostique.
Le mode adaptatif exige au moins 16 pixels par dimension pour le détecteur C3.

L'onglet Prétraitement affiche l'image réellement utilisée avant normalisation,
la méthode, les étapes appliquées, le temps et les états de C1–C5. Les exports
JSON incluent ces informations ainsi que les fichiers de modèles adaptatifs.

Les modèles adaptatifs ont aussi été exportés avec sklearn 1.6.1. Dans
l'environnement actuel 1.9.1, l'adaptateur refuse explicitement leur chargement
au lieu de poursuivre avec une compatibilité non garantie. Aucun changement
d'environnement ni téléchargement de poids n'est effectué automatiquement.

## Traitements et interprétation

L'image est redimensionnée en 224 × 224 par INTER_AREA puis corrigée par
Shades of Gray p=6 et normalisée avec les constantes ImageNet du projet.
VGG19-UNet et DeepLabV3+ (encodeur ResNet50) produisent chacun des logits ;
leur sigmoid donne deux cartes dont la moyenne est seuillée à 0,5.
Les masques sont ensuite agrandis à la résolution native par voisin proche,
comme dans le script local existant. Les cartes brutes restent à 224 × 224.

`handcrafted200` reçoit l'image corrigée et le masque à 224 × 224, puis le
scaler existant transforme les 200 valeurs. Les CNN hybrides reçoivent le
même tenseur image et les valeurs standardisées. Leurs embeddings de
512 valeurs sont concaténés **ResNet50 puis DenseNet201**, donnant les
1024 entrées du SVM ; les valeurs manuelles ne sont pas ajoutées une seconde fois.
Le fichier `hybrid_final_svm.joblib` est lui-même un pipeline
`StandardScaler → SVC(C=0.1, kernel='rbf', probability=False)` ; il est appelé
dans son intégralité, conservant la seconde standardisation des embeddings.
Cet ordre est celui du script local et la configuration confirme `512x2`.
Le code d'entraînement du SVM n'est pas présent : la provenance complète de
l'export ne peut donc pas être auditée à partir de ce seul dépôt.

La classe entière 0–6 correspond à MEL, NV, BCC, AKIEC, BKL, DF, VASC
dans cet ordre. Ce résultat n'est pas une probabilité médicale calibrée.
Sans annotation réelle, aucun Dice, IoU ou accuracy n'est affiché.
Un masque vide est signalé : le comportement de secours interne du
descripteur d'origine est conservé.

Le descripteur est explicitement une **adaptation**, pas celui exact du papier.
La démo illustre les modèles fournis ; elle ne valide ni la reproduction des
performances publiées ni un usage clinique.

**Prototype expérimental de recherche — ne constitue pas un diagnostic médical.**

## Export et sauvegarde

L'onglet Export propose les PNG, `prediction.json` et une archive ZIP complète.
**Sauvegarder cette exécution sur disque** crée un dossier unique dans
`results/paper_2021/demo_runs/<horodatage_UTC_UUID>/`, sans écraser les précédents.
Il contient les six images (originale, prétraitée, trois masques, superposition),
le JSON technique, `probabilities.npz` et `manual_features.npy`.
Les masques PNG valent 0/255 ; les tableaux binaires internes valent 0/1.
Les images importées ne sont enregistrées que lors de cette sauvegarde explicite.

## Validation

```powershell
.\.venv\Scripts\python.exe -m compileall -q src/inference demo/paper_2021_app.py tests/test_paper_2021_inference.py
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

Les tests unitaires utilisent des doubles légers pour les réseaux et le SVM.
Test facultatif avec les vrais modèles et une image fournie localement :

```powershell
$env:PAPER2021_INTEGRATION = '1'
.\.venv\Scripts\python.exe -m pytest tests/test_paper_2021_inference.py -k real_local_models -q -p no:cacheprovider
Remove-Item Env:PAPER2021_INTEGRATION
```

Ce test est ignoré si les artefacts ou les images manquent. Le premier chargement
des quatre réseaux peut prendre du temps et plusieurs Go de mémoire CPU/GPU.

## Évaluation Ground Truth — ISIC 2018

Choisir **Évaluation Ground Truth** dans la barre latérale. La démonstration
précédente reste accessible. Aucun résultat réel n'est fourni avec cette extension.

### Préparer les véritables données

Pour Task 1, placer les images dans `test_images/segmentation/images/` et les
annotations dans `test_images/segmentation/masks/`, par exemple
`ISIC_0024306.jpg` et `ISIC_0024306_segmentation.png`. Les identifiants doivent
correspondre exactement ; les doublons sont refusés et les fichiers sans
correspondance sont signalés. Aucun fichier image d'exemple n'est généré.

Pour Task 3, placer les images dans `test_images/classification/images/` et
**remplacer** `test_images/classification/ground_truth.csv` par le CSV officiel.
Le fichier créé contient uniquement les en-têtes, sans annotation fictive.
Colonnes : `image` (ou `image_id`), puis MEL, NV, BCC, AKIEC, BKL, DF, VASC.
Chaque ligne doit contenir une seule classe à 1 et les autres à 0. Une image
sans annotation n'est pas proposée à l'évaluation ; un doublon ou un encodage
ambigu provoque une erreur. Conserver la provenance des annotations :
l'application valide leur structure, pas leur authenticité auprès d'ISIC.

### Vérifier la validité scientifique

Les CSV découverts et utilisés par défaut sont :

- `results/paper_2021/task1_splits.csv` : 1 815 train, 389 val, 390 test ;
- `results/paper_2021/task3_splits.csv` : 7 010 train, 1 502 val, 1 503 test.

Il s'agit des partitions locales exportées à partir des données d'entraînement
ISIC, **pas d'une preuve d'appartenance au test officiel caché du challenge**.
Les chemins `/content/...` historiques sont ignorés pour charger les images.
Les fichiers C2–C5 ne sont pas utilisés comme partitions des modèles 2021.

L'audit précède toute évaluation et est enregistré dans le manifeste. Une
sélection train/val/inconnue ou ambiguë porte explicitement une restriction
à la démonstration. Si le CSV manque, le message exact est :
**Partition non vérifiée : résultat de démonstration uniquement**.
La case « Proposer uniquement les identifiants du test exporté » facilite
la sélection ; elle n'affirme pas l'absence de fuite par lésion.

Aucun CSV de regroupement par lésion n'a été trouvé dans le dépôt inspecté.
Le champ optionnel des métadonnées accepte un CSV contenant `image_id`
(ou `image`) et `lesion_id`. Fournir les métadonnées de **toutes** les partitions
pour détecter les lésions partagées avec train/val. Les métadonnées incomplètes
et les regroupements non vérifiés restent signalés. Les doublons d'identifiants
de partition sont recensés ; les images aux pixels identiques dans le lot sont
signalées et excluent toute interprétation comme généralisation indépendante.
Cela ne détecte pas toutes les images quasi identiques ni les patients communs
sans métadonnées. Même un sous-ensemble test vérifié n'est pas le test complet.

### Calcul et visualisations

Sélectionner un ensemble local, puis cliquer **Évaluer la sélection et sauvegarder**.
Les modèles utilisent `st.cache_resource` ; aucun GT n'est transmis au pipeline.

Si `selected_images.csv` est présent dans le dossier Task 1 ou Task 3, sa colonne
`image_id` sert à limiter les choix disponibles (option désactivable). La colonne
`split` de cette sélection ne remplace jamais l'audit des véritables partitions
exportées. Les doublons sont refusés et les images/annotations manquantes signalées.

L'évaluation utilise la méthode choisie par les deux boutons. Cocher
**Comparer les deux méthodes sur exactement les mêmes images** exécute les
deux prétraitements avec la même instance des modèles 2021 et les mêmes identifiants.
Les empreintes des images, annotations et partitions doivent correspondre entre
les deux passages ; sinon la comparaison est refusée. Le tableau présente les
scores et la différence **adaptatif − Shades of Gray**, sans supposer une amélioration.
Chaque comparaison crée un dossier `comparison_<UTC_UUID>` avec `comparison.csv`,
un résumé JSON, un manifeste et les résultats distincts de chaque méthode.
Si une méthode échoue, la comparaison est marquée `failed` sans tableau final.
Les matrices et rapports restent consultables pour chaque méthode.

Chaque ligne de métriques exportée conserve la méthode, les étapes, les modèles
scientifiques et les temps. Les temps par image excluent le chargement initial
des modèles ; ils incluent lecture/validation et inférence, pas l'écriture des exports.
Ce mode constitue un changement de distribution par rapport à l'entraînement
avec Shades of Gray ; les petits ensembles locaux ne prouvent pas une amélioration
généralisable. Les GT restent réservés à l'évaluation.

Task 1 charge seulement les deux segmentateurs et fonctionne indépendamment
des objets sklearn. Les mêmes calculs de probabilités que dans la démonstration
sont réutilisés. Le seuil 0,5 est appliqué à 224² avant agrandissement natif par
voisin proche. Les PNG GT binaires 0/1 ou 0/255 sont lus séparément. Un GT de
taille différente est aligné par voisin proche si le rapport de forme concorde ;
sinon il est refusé. Les images avec rotation/miroir EXIF implicite sont refusées.
Vérifier visuellement l'orientation : aucun recalage anatomique n'est réalisé.

Dice et IoU sont calculés sans lissage à la résolution native, séparément pour
VGG19-UNet, DeepLabV3+ et l'ensemble. Deux masques vides ont un score de 1.
Les moyennes sont arithmétiques par image, pas des scores micro sur tous les pixels.
Les superpositions montrent TP en vert, FP en rouge et FN en bleu ; les TN
conservent l'image originale.

Task 3 exécute le pipeline complet existant, compare la classe réelle à la
prédiction et affiche accuracy, balanced accuracy, macro-F1, précision/rappel,
rapport et matrice de confusion (lignes réelles, colonnes prédites). Balanced
accuracy est le rappel moyen des classes réellement présentes. Macro-F1 porte
sur les sept classes, avec les scores indéfinis fixés à zéro ; le support est
affiché pour éviter une interprétation trompeuse d'un petit lot. Aucun score
de confiance n'est inventé. Le blocage de version sklearn décrit plus haut
reste applicable à la classification.

### Résultats sauvegardés

Chaque clic d'évaluation crée un dossier unique dans
`results/paper_2021/evaluation/<UTC_UUID>/`, sans écrasement :

- `metrics.csv`, `summary.json`, `manifest.json` ;
- pour Task 3 : `classification_report.csv` et `confusion_matrix.csv` ;
- par image : originale, métadonnées d'inférence et, pour Task 1, GT aligné,
  trois masques et trois superpositions d'erreurs.

Le manifeste conserve la sélection, l'audit, les empreintes SHA-256 des entrées
et les images terminées. Un lot interrompu est marqué `failed` et ne reçoit pas
de résumé final ; les fichiers partiels restent identifiables. Le navigateur
permet de télécharger les métriques et les JSON sans relancer les modèles.

Tests supplémentaires :

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_paper_2021_evaluation.py -q -p no:cacheprovider
```
