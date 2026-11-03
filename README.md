<p align="center">
  <img src="assets/banner.png" alt="PulmoScan AI — classification de coupes de scanner thoracique" width="100%">
</p>

<h1 align="center">PulmoScan AI</h1>

<p align="center">
  <strong>Classification de coupes de scanner thoracique avec TensorFlow/Keras et MobileNetV3-Large</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/TensorFlow-2.21-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow 2.21">
  <img src="https://img.shields.io/badge/Keras-3.15-D00000?logo=keras&logoColor=white" alt="Keras 3.15">
  <img src="https://img.shields.io/badge/Architecture-MobileNetV3--Large-0E7C86" alt="MobileNetV3-Large">
  <img src="https://img.shields.io/badge/Licence-MIT-22B8CF" alt="Licence MIT">
  <img src="https://img.shields.io/badge/Statut-Pédagogique-F59F00" alt="Projet pédagogique">
</p>

<p align="center">
  <strong>Mohammed Oughlane</strong> · Marrakech, Maroc · novembre 2026
</p>

---

## Résumé

PulmoScan AI classe une coupe axiale de scanner thoracique en **normal**, **COVID-19** ou **non informatif**.
Le modèle adapte **MobileNetV3-Large**, pré-entraîné sur ImageNet, par transfer learning avec **TensorFlow/Keras**.
Projet pédagogique : il ne constitue pas un dispositif médical et ne doit pas guider une décision de santé.

## Table des matières

- [Présentation](#présentation)
- [Données](#données)
- [Modèle](#modèle)
- [Entraînement](#entraînement)
- [Résultats](#résultats)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du dépôt](#structure-du-dépôt)
- [Analyse critique](#analyse-critique)
- [Ma contribution](#ma-contribution)
- [Crédits et attribution](#crédits-et-attribution)
- [À propos de l'auteur](#à-propos-de-lauteur)
- [Avertissement](#avertissement)

## Présentation

**PulmoScan AI** est un projet de vision par ordinateur qui classe une image de coupe axiale de scanner thoracique dans l'une des trois catégories suivantes :

- **Normal (`nCT`)** : les poumons sont visibles, sans signe associé à la COVID-19 dans le jeu de données.
- **Positif (`pCT`)** : la coupe présente des anomalies associées à une pneumonie à COVID-19.
- **Non informatif (`NiCT`)** : la coupe ne montre pas suffisamment les poumons pour permettre une classification pertinente.

Le modèle utilise **MobileNetV3-Large**, pré-entraîné sur ImageNet, puis adapté à cette tâche par *transfer learning* avec **TensorFlow/Keras**.

> **À retenir :** PulmoScan AI est un projet développé indépendamment. Il est dérivé du projet open source d'origine et s'appuie sur MobileNetV3-Large pré-entraîné sur ImageNet, adapté à une classification en trois classes.

### Aperçu

![Exemples de coupes de scanner par classe](assets/ct-examples.png)

*Exemples de coupes issus du jeu de données utilisé dans le projet.*

## Données

Les données proviennent du jeu Kaggle [CT Scans for COVID-19 Classification](https://www.kaggle.com/datasets/azaemon/preprocessed-ct-scans-for-covid19), publié par **Abu Zahid Bin Aziz** sous licence **CC BY 4.0**. Les images originales sont issues de la ressource [iCTCF](http://ictcf.biocuckoo.cn/) (Ning et al., 2020).

Le jeu contient **19 685 coupes** :

| Classe | Description | Nombre d'images |
|---|---|---:|
| `nCT` | Scanner normal | 9 979 |
| `pCT` | Scanner positif à la COVID-19 | 4 001 |
| `NiCT` | Scanner non informatif | 5 705 |
| **Total** | | **19 685** |

![Répartition des classes](assets/class-distribution-fr.png)

### Répartition des données

Les images de chaque classe sont mélangées avec une graine fixe (`42`), puis réparties en ensembles d'entraînement, de validation et de test. Les effectifs sont fixés dans le notebook, ce qui rend le découpage identique à chaque exécution.

| Classe | Entraînement | Validation | Test |
|---|---:|---:|---:|
| `nCT` | 6 985 | 1 496 | 1 498 |
| `NiCT` | 3 993 | 855 | 857 |
| `pCT` | 2 800 | 600 | 601 |
| **Total** | **13 778** | **2 951** | **2 956** |

**Limite importante :** le découpage est effectué par coupe et non par patient. Des images d'un même patient peuvent donc se retrouver dans plusieurs ensembles. Ce risque de fuite de données est détaillé dans la section [Analyse critique](#analyse-critique).

### Prétraitement

- Redimensionnement à 256 px.
- Recadrage central en 224 × 224 pour la validation et le test.
- Recadrage aléatoire, retournement horizontal et zoom aléatoire pendant l'entraînement, avec des couches de prétraitement Keras.
- **Aucune normalisation ImageNet et aucune division par 255** : le MobileNetV3 de Keras intègre sa propre normalisation et attend des pixels bruts entre 0 et 255.

### Ordre des classes

Le tri alphabétique des noms de dossiers, majuscules avant minuscules, conformément au comportement de `image_dataset_from_directory`, donne **`NiCT`, `nCT`, `pCT`**. Cet ordre est enregistré explicitement dans `outputs/checkpoints/class_names.json` et réutilisé dans le notebook, l'évaluation et l'application Streamlit.

## Modèle

![Architecture du classifieur PulmoScan AI](assets/model-architecture-fr.png)

Le backbone MobileNetV3-Large extrait **960 caractéristiques** de chaque image. Sa tête de classification d'origine est remplacée par une tête adaptée aux trois classes :

```text
Image 224 × 224 × 3 (pixels bruts 0-255)
      ↓
MobileNetV3-Large (normalisation interne Keras)
      ↓
960 caractéristiques
      ↓
Dense(1280) → ReLU → Dropout(0.2)
      ↓
Dense(3) → Softmax
      ↓
NiCT / nCT / pCT
```

| Élément | Configuration |
|---|---|
| Backbone | MobileNetV3-Large pré-entraîné sur ImageNet |
| Taille d'entrée | 224 × 224 × 3 |
| Nombre de caractéristiques | 960 |
| Tête de classification | `Dense(1280)` → ReLU → `Dropout(0.2)` → `Dense(3, softmax)` |
| Fonction de sortie | Softmax |
| Fonction de perte | `SparseCategoricalCrossentropy` |
| Stratégie | Fine-tuning du réseau entier (`base_model.trainable = True`) |

## Entraînement

| Paramètre | Valeur |
|---|---|
| Optimiseur | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `1e-4` |
| Batch size | `16` |
| Nombre d'époques | `25` |
| Perte | Entropie croisée catégorielle parcimonieuse |
| Précision | float32 |
| Callbacks | `ModelCheckpoint` (meilleure exactitude de validation), `EarlyStopping` (patience 5, restauration des meilleurs poids) |

Le modèle est enregistré dans `outputs/checkpoints/pulmoscan_mobilenetv3.keras`. En mode smoke test, le fichier porte le suffixe `_smoke` pour ne jamais écraser un vrai modèle.

**Mode smoke test.** `SMOKE_TEST = True` limite l'entraînement à une époque et à un petit sous-ensemble des données. Il sert uniquement à vérifier que le pipeline fonctionne. Pour l'entraînement complet, passez à `False`.

**Vérification effectuée :** le notebook a été exécuté de bout en bout en mode smoke test sur un jeu de données synthétique, sous Windows avec TensorFlow 2.21.0 en CPU. Aucun entraînement complet sur le vrai jeu de données n'a été réalisé.

## Résultats

> **Aucun entraînement complet TensorFlow/Keras sur le vrai jeu de données n'a été exécuté.** Le tableau ci-dessous est volontairement laissé incomplet. Il ne doit être rempli qu'avec les valeurs produites par votre propre exécution.

| Ensemble | Images | Exactitude | Perte |
|---|---:|---:|---:|
| Entraînement | 13 778 | à compléter après l'entraînement complet | à compléter après l'entraînement complet |
| Validation | 2 951 | à compléter après l'entraînement complet | à compléter après l'entraînement complet |
| **Test** | **2 956** | **à compléter après l'entraînement complet** | **à compléter après l'entraînement complet** |

Après un entraînement complet, le notebook produit :

- `assets/courbes_validation.png` : courbes de perte et d'exactitude ;
- `assets/matrice_confusion.png` : matrice de confusion sur le jeu de test ;
- `assets/predictions.png` : une prédiction par classe ;
- `outputs/metrics.json` : exactitude, perte, matrice de confusion et métriques par classe.

Les métriques par classe (précision, rappel, F1, support) sont calculées avec `classification_report` de scikit-learn.

### Comparaison avec le projet d'origine

| Élément | PyTorch (projet d'origine) | TensorFlow/Keras (ma version) |
|---|---|---|
| Exactitude test | **99,76 %** — résultat du projet d'origine | à compléter après l'entraînement complet |
| F1 par classe | non fourni dans le projet d'origine | à compléter après l'entraînement complet |
| Durée d'entraînement | **41 min 07 s** (25 époques) — résultat du projet d'origine | à mesurer |
| Matériel | MacBook Pro M5 Pro, 48 Go, MPS — projet d'origine | à renseigner selon l'exécution |

Les valeurs PyTorch sont reprises telles quelles du README du projet d'origine et clairement identifiées comme telles. **Ce ne sont pas les résultats de cette migration.** Aucune métrique, durée ou configuration matérielle TensorFlow n'est inventée : la colonne de droite reste vide tant qu'un entraînement complet n'a pas été exécuté.

![Courbes de l'exécution d'origine](assets/training-curves-fr.png)

*Figure du projet d'origine (PyTorch), conservée pour comparaison. Elle ne constitue pas un résultat TensorFlow.*

![Résultats du projet d'origine](assets/results-fr.png)

*Métriques héritées du projet d'origine. Elles n'ont pas été recalculées pour cette version TensorFlow/Keras.*

## Installation

### Prérequis

- Python 3.11
- Git
- Environ 4 Go d'espace disque pour les données
- Un compte Kaggle pour télécharger le jeu de données
- GPU Apple Silicon (Metal) ou NVIDIA (CUDA) recommandé pour l'entraînement ; le CPU reste possible, mais plus lent.

### 1. Cloner le dépôt

Remplacez `[URL_DE_MON_REPO]` par l'URL de votre dépôt GitHub :

```bash
git clone [URL_DE_MON_REPO]
cd Pulmoscan-ai
```

### 2. Créer l'environnement et installer les dépendances

```bash
python3.11 -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Sur un Mac Apple Silicon, décommentez la ligne `tensorflow-metal` dans `requirements.txt` pour activer l'accélération GPU Metal.

### 3. Télécharger le jeu de données

Depuis Kaggle, créez un jeton API dans **Settings → API → Create New Token**. Placez ensuite le fichier `kaggle.json` dans le dossier `.kaggle` de votre répertoire utilisateur.

**macOS / Linux :**

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download \
  -d azaemon/preprocessed-ct-scans-for-covid19 \
  -p ~/datasets/covid-ct \
  --unzip
```

**Windows PowerShell :**

```powershell
New-Item -ItemType Directory -Force "$HOME\.kaggle"
Move-Item "$HOME\Downloads\kaggle.json" "$HOME\.kaggle\kaggle.json"

kaggle datasets download `
  -d azaemon/preprocessed-ct-scans-for-covid19 `
  -p "$HOME\datasets\covid-ct" `
  --unzip
```

Copiez les dossiers des **coupes originales** dans le répertoire du projet :

```text
Pulmoscan-ai/
└── CT Scans/
    ├── nCT/
    ├── NiCT/
    └── pCT/
```

Pour utiliser un autre emplacement, définissez la variable d'environnement `PULMOSCAN_DATA_ROOT`.

## Utilisation

### Lancer le notebook

Ouvrez `classifier.ipynb` dans Jupyter ou VS Code, puis exécutez les cellules dans l'ordre.

Pour vérifier rapidement l'installation, gardez `SMOKE_TEST = True` dans la cellule des paramètres. Ce mode utilise un sous-ensemble réduit des données et une seule époque.

Pour lancer l'entraînement complet de 25 époques :

1. ouvrez la cellule **Paramètres** ;
2. remplacez `SMOKE_TEST = True` par `SMOKE_TEST = False` ;
3. exécutez toutes les cellules dans l'ordre.

Le notebook enregistre le modèle dans `outputs/checkpoints/pulmoscan_mobilenetv3.keras` et l'ordre des classes dans `outputs/checkpoints/class_names.json`.

### Utiliser le modèle entraîné

```python
import json

import numpy as np
import tensorflow as tf
from PIL import Image

model = tf.keras.models.load_model("outputs/checkpoints/pulmoscan_mobilenetv3.keras")
class_names = json.load(open("outputs/checkpoints/class_names.json", encoding="utf-8"))

image = Image.open("chemin/vers/coupe.jpg").convert("RGB").resize((256, 256), Image.BILINEAR)
left = (256 - 224) // 2
image = image.crop((left, left, left + 224, left + 224))

# Pixels bruts 0-255 : ne pas diviser par 255, MobileNetV3 normalise en interne.
x = np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)
probs = model.predict(x, verbose=0)[0]

for name, p in sorted(zip(class_names, probs), key=lambda t: t[1], reverse=True):
    print(f"{name}: {p:.1%}")
```

### Lancer la démo Streamlit (optionnel)

Installez les dépendances dédiées :

```bash
pip install -r requirements-demo.txt
streamlit run app.py
```

Ouvrez ensuite l'adresse locale affichée dans le terminal (généralement `http://localhost:8501`).

La démo charge `pulmoscan_mobilenetv3.keras` et `class_names.json`, puis affiche les probabilités prédites pour les trois classes. Si le modèle est absent, l'application affiche un message d'erreur explicite.

### Docker (optionnel)

Le modèle doit d'abord être entraîné et placé dans `outputs/checkpoints/`.

```bash
docker build -t pulmoscan-ai .
docker run --rm -p 8501:8501 pulmoscan-ai
```

Puis ouvrez `http://localhost:8501` dans votre navigateur.

## Structure du dépôt

```text
Pulmoscan-ai/
├── app.py
├── assets/
│   ├── banner.png
│   ├── ct-examples.png
│   ├── class-distribution-fr.png
│   ├── model-architecture-fr.png
│   ├── training-curves-fr.png
│   └── results-fr.png
├── classifier.ipynb
├── classifier.html
├── outputs/
│   └── checkpoints/
│       └── class_names.json
├── reports/
│   └── training_history_original_run.json
├── scripts/
│   └── make_figures.py
├── Dockerfile
├── requirements.txt
├── requirements-demo.txt
└── README.md
```

Après un entraînement complet, les fichiers suivants s'ajoutent :

```text
outputs/
├── checkpoints/
│   └── pulmoscan_mobilenetv3.keras
└── metrics.json

assets/
├── courbes_validation.png
├── matrice_confusion.png
└── predictions.png
```

## Analyse critique

Les performances du projet d'origine sont élevées, mais elles ne suffisent pas à démontrer que le modèle serait fiable dans un contexte médical réel.

1. **Découpage par coupe et non par patient** — des coupes d'un même patient peuvent apparaître dans les ensembles d'entraînement et de test. Les performances de test peuvent donc être surestimées.
2. **Source de données limitée** — les images proviennent de deux hôpitaux de Wuhan. La capacité du modèle à généraliser à d'autres hôpitaux, populations et appareils n'est pas établie.
3. **Absence de contexte 3D** — le modèle analyse chaque coupe indépendamment, sans exploiter le volume complet ni les informations cliniques du patient.
4. **Déséquilibre des classes** — la classe `nCT` représente environ la moitié des images. Une exactitude globale élevée peut masquer de mauvaises performances sur `pCT` ou `NiCT` ; la précision, le rappel, le F1 et la matrice de confusion sont donc nécessaires.
5. **Risque d'apprentissage de raccourcis** — le modèle pourrait exploiter des artefacts visuels ou des indices d'acquisition plutôt que les structures anatomiques pertinentes. Une analyse de saillance, par exemple avec Grad-CAM, serait utile.
6. **Calibration et validation externe absentes** — les probabilités prédites ne sont pas nécessairement bien calibrées et aucune validation sur un jeu de données externe n'est documentée.
7. **Résultats TensorFlow non produits** — le code d'évaluation est prêt, mais aucun entraînement complet n'a été exécuté lors de la migration. Les résultats restent à produire et à vérifier.
8. **Pas d'interprétabilité** — le notebook ne montre pas encore de Grad-CAM ni de carte de saillance.

### Améliorations possibles

- Effectuer un découpage strict par patient.
- Évaluer le modèle sur un jeu de données externe.
- Examiner les prédictions avec Grad-CAM.
- Mesurer la calibration et fournir des intervalles de confiance.
- Publier les métriques détaillées par classe après un entraînement complet.

## Ma contribution

Cette version vise à rendre le projet plus accessible, documenté et exploitable en français. Les changements réellement apportés comprennent :

- **Migration PyTorch vers TensorFlow/Keras** : notebook, pipeline `tf.data`, modèle MobileNetV3-Large, entraînement, sauvegarde et rechargement.
- **Nouvelle évaluation** : ajout de la matrice de confusion et du rapport précision/rappel/F1 par classe avec scikit-learn. Le code est prêt ; les résultats restent à produire après un entraînement complet.
- **Application Streamlit** : chargement du modèle `.keras` et de `class_names.json`, prétraitement 224 × 224 en pixels bruts, prédiction softmax.
- **Documentation française** : README, notebook, commentaires, messages et libellés de figures.
- **Rebranding** sous le nom **PulmoScan AI** et création des visuels français.
- **Mode smoke test** et variable `PULMOSCAN_DATA_ROOT` pour vérifier le pipeline sans le vrai jeu de données.
- **Docker** : mise à jour du conteneur pour TensorFlow/Keras.
- **Clarification des crédits, de l'attribution et de la provenance des résultats**

Aucun entraînement complet sur le vrai jeu de données n'a été réalisé. Aucune métrique TensorFlow n'est présentée comme mesurée.

## Crédits et attribution

Ce projet est dérivé du projet open source d'origine :

- **Projet d'origine :** `covid-ct-classifier`, auteur **chrispathway**, [URL_DU_REPO_ORIGINAL].
- Le fichier [`LICENSE`](LICENSE) de ce dépôt est conservé.

### Jeu de données

- **Abu Zahid Bin Aziz**, [CT Scans for COVID-19 Classification](https://www.kaggle.com/datasets/azaemon/preprocessed-ct-scans-for-covid19), Kaggle, licence **CC BY 4.0**.
- **Ning, W. et al.**, *iCTCF: an integrative resource of chest computed tomography images and clinical features of patients with COVID-19 pneumonia* (2020).

### Architecture et outils

- **Howard, A. et al.**, [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244), ICCV 2019.
- [TensorFlow](https://www.tensorflow.org/) et [Keras](https://keras.io/).
- NumPy, pandas, scikit-learn, Matplotlib, Seaborn, Pillow, Jupyter et Streamlit.

Les ressources et bibliothèques tierces utilisées sont référencées ci-dessus.

## À propos de l'auteur

**Mohammed Oughlane**  
Marrakech, Maroc

- GitHub : [LIEN_GITHUB]
- LinkedIn : [LIEN_LINKEDIN]

**Période de réalisation :** 30/10/2026 – 03/11/2026

---

## Avertissement

**PulmoScan AI est un projet pédagogique et ne constitue pas un dispositif médical.** Il ne doit pas être utilisé pour établir un diagnostic, trier des patients ou prendre une décision de santé. Les résultats présentés ne constituent pas une validation clinique et ne remplacent pas l'avis d'un professionnel de santé.

Les données utilisées sont publiques et anonymisées. Aucune donnée patient identifiable n'est incluse dans ce dépôt.

---

<p align="center">
  <sub>Projet pédagogique de deep learning · Computer Vision · TensorFlow/Keras</sub>
</p>
