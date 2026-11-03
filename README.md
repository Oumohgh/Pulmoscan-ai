```{=html}
<p align="center">
```
`<img src="assets/banner.png" alt="PulmoScan AI — classification de coupes de scanner thoracique" width="100%">`{=html}
```{=html}
</p>
```
```{=html}
<h1 align="center">
```
PulmoScan AI
```{=html}
</h1>
```
```{=html}
<p align="center">
```
`<strong>`{=html}Classification de coupes de scanner thoracique avec
PyTorch et MobileNetV3-Large`</strong>`{=html}
```{=html}
</p>
```
```{=html}
<p align="center">
```
`<img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11">`{=html}
`<img src="https://img.shields.io/badge/PyTorch-2.14-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.14">`{=html}
`<img src="https://img.shields.io/badge/Architecture-MobileNetV3--Large-0E7C86" alt="MobileNetV3-Large">`{=html}
`<img src="https://img.shields.io/badge/Licence-MIT-22B8CF" alt="Licence MIT">`{=html}
`<img src="https://img.shields.io/badge/Statut-Pédagogique-F59F00" alt="Projet pédagogique">`{=html}
```{=html}
</p>
```
```{=html}
<p align="center">
```
`<strong>`{=html}Mohammed Oughlane`</strong>`{=html} · Marrakech, Maroc
· Novembre 2026
```{=html}
</p>
```

------------------------------------------------------------------------

## Présentation

**PulmoScan AI** est un projet de vision par ordinateur qui classe une
image de coupe axiale de scanner thoracique dans l'une des trois
catégories suivantes :

-   **Normal (`nCT`)** : les poumons sont visibles, sans signe associé à
    la COVID-19 dans le jeu de données.
-   **Positif (`pCT`)** : la coupe présente des anomalies associées à
    une pneumonie à COVID-19.
-   **Non informatif (`NiCT`)** : la coupe ne montre pas suffisamment
    les poumons pour permettre une classification pertinente.

Le modèle utilise **MobileNetV3-Large**, pré-entraîné sur ImageNet, puis
adapté à cette tâche par *transfer learning* avec **PyTorch**.

> **À retenir :** PulmoScan AI est un projet développé indépendamment. Le modèle s'appuie sur MobileNetV3-Large pré-entraîné sur ImageNet, puis adapté à une tâche de classification en trois classes avec PyTorch.

## Aperçu

![Exemples de coupes de scanner par classe](assets/ct-examples.png)

*Exemples de coupes issus du jeu de données utilisé dans le projet.*

### Architecture du modèle

![Architecture du classifieur PulmoScan
AI](assets/model-architecture-fr.png)

Le backbone MobileNetV3-Large extrait **960 caractéristiques** de chaque
image. Sa tête de classification d'origine est remplacée par une tête
adaptée aux trois classes :

``` text
Image 224 × 224
      ↓
MobileNetV3-Large
      ↓
960 caractéristiques
      ↓
Linear(960, 1280) → ReLU → Dropout(0.2)
      ↓
Linear(1280, 3) → LogSoftmax
      ↓
nCT / NiCT / pCT
```

  -----------------------------------------------------------------------
  Élément                             Configuration
  ----------------------------------- -----------------------------------
  Backbone                            MobileNetV3-Large pré-entraîné sur
                                      ImageNet

  Taille d'entrée                     224 × 224 × 3

  Nombre de caractéristiques          960

  Tête de classification              Linear(960, 1280) → ReLU →
                                      Dropout(0.2) → Linear(1280, 3)

  Fonction de sortie                  LogSoftmax

  Fonction de perte                   `NLLLoss`

  Stratégie                           Fine-tuning du réseau entier
  -----------------------------------------------------------------------

`NLLLoss` associée à `LogSoftmax` est équivalente à une entropie croisée
classique.

## Jeu de données

Les données proviennent du jeu Kaggle [CT Scans for COVID-19
Classification](https://www.kaggle.com/datasets/azaemon/preprocessed-ct-scans-for-covid19),
publié par Abu Zahid Bin Aziz sous licence **CC BY 4.0**. Les images
originales sont issues de la ressource
[iCTCF](http://ictcf.biocuckoo.cn/) (Ning et al., 2020).

Le jeu contient **19 685 coupes** :

  Classe      Description                       Nombre d'images
  ----------- ------------------------------- -----------------
  `nCT`       Scanner normal                              9 979
  `pCT`       Scanner positif à la COVID-19               4 001
  `NiCT`      Scanner non informatif                      5 705
  **Total**                                          **19 685**

![Répartition des classes](assets/class-distribution-fr.png)

### Répartition des données

Les images de chaque classe sont mélangées avec une graine fixe (`42`),
puis réparties en ensembles d'entraînement, de validation et de test.

  Classe        Entraînement   Validation        Test
  ----------- -------------- ------------ -----------
  `nCT`                6 985        1 496       1 498
  `NiCT`               3 993          855         857
  `pCT`                2 800          600         601
  **Total**       **13 778**    **2 951**   **2 956**

**Limite importante :** le découpage est effectué par coupe et non par
patient. Des images d'un même patient peuvent donc se retrouver dans
plusieurs ensembles. Ce risque de fuite de données est détaillé dans la
section [Limites](#limites-et-pistes-damélioration).

### Prétraitement

-   Redimensionnement à 256 px.
-   Recadrage central en 224 × 224 pour la validation et le test.
-   Recadrage aléatoire et retournement horizontal pendant
    l'entraînement.
-   Normalisation avec les moyennes et écarts-types d'ImageNet.

## Entraînement

Les paramètres ci-dessous présentent la configuration utilisée pour l'entraînement du modèle.

  -----------------------------------------------------------------------
  Paramètre                           Valeur
  ----------------------------------- -----------------------------------
  Optimiseur                          AdamW

  Learning rate                       `1e-4`

  Weight decay                        `1e-4`

  Batch size                          `16`

  Nombre d'époques                    `25`

  Précision                           float32

  Environnement du run documenté      MacBook Pro M5 Pro, 48 Go, PyTorch
                                      MPS

  Durée documentée                    41 min 07 s
  -----------------------------------------------------------------------

La configuration documentée utilise la précision float32. La précision mixte peut être évaluée selon le matériel disponible.

## Résultats

> **Résultats :** les métriques ci-dessous présentent les performances rapportées pour le modèle sur les ensembles d'entraînement, de validation et de test.

  Ensemble            Images      Accuracy         Loss
  -------------- ----------- ------------- ------------
  Entraînement        13 778       98,70 %       0,0316
  Validation           2 951       99,25 %       0,0216
  **Test**         **2 956**   **99,76 %**   **0,0070**

![Courbes d'entraînement et de
validation](assets/training-curves-fr.png)

![Résultats du projet](assets/results-fr.png)

*Les figures présentent les courbes et résultats d'entraînement du projet.*

Le notebook contient également une cellule permettant de calculer les
métriques par classe sur le jeu de test :

-   matrice de confusion ;
-   précision (*precision*) ;
-   rappel (*recall*) ;
-   score F1 ;
-   support et exactitude globale.

Le notebook permet également de générer ces métriques détaillées après l'exécution de la cellule d'évaluation.

## Installation et exécution

### Prérequis

-   Python 3.11
-   Git
-   Environ 4 Go d'espace disque pour les données
-   Un compte Kaggle pour télécharger le jeu de données
-   GPU Apple Silicon (MPS) ou NVIDIA (CUDA) recommandé pour
    l'entraînement ; le CPU reste possible, mais plus lent.

### 1. Cloner le dépôt

Remplacez `<URL_DU_DEPOT>` par l'URL de votre dépôt GitHub :

``` bash
git clone <URL_DU_DEPOT>
cd pulmoscan-ai
```

### 2. Créer l'environnement et installer les dépendances

``` bash
python3.11 -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 3. Télécharger le jeu de données

Depuis Kaggle, créez un jeton API dans **Settings → API → Create New
Token**. Placez ensuite le fichier `kaggle.json` dans le dossier
`.kaggle` de votre répertoire utilisateur.

**macOS / Linux :**

``` bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download \
  -d azaemon/preprocessed-ct-scans-for-covid19 \
  -p ~/datasets/covid-ct \
  --unzip
```

**Windows PowerShell :**

``` powershell
New-Item -ItemType Directory -Force "$HOME\.kaggle"
Move-Item "$HOME\Downloads\kaggle.json" "$HOME\.kaggle\kaggle.json"

kaggle datasets download `
  -d azaemon/preprocessed-ct-scans-for-covid19 `
  -p "$HOME\datasets\covid-ct" `
  --unzip
```

Copiez les dossiers des **coupes originales** dans le répertoire du
projet :

``` text
pulmoscan-ai/
└── CT Scans/
    ├── nCT/
    ├── NiCT/
    └── pCT/
```

Pour utiliser un autre emplacement, définissez la variable
d'environnement `PULMOSCAN_DATA_ROOT`.

### 4. Lancer le notebook

Ouvrez `classifier.ipynb` dans Jupyter ou VS Code, puis exécutez les
cellules dans l'ordre.

Pour vérifier rapidement l'installation avant un entraînement complet,
activez `SMOKE_TEST = True` dans la cellule des paramètres. Ce mode
utilise un sous-ensemble réduit des données et une seule époque.

### 5. Lancer la démo Streamlit (optionnel)

Installez les dépendances dédiées :

``` bash
pip install -r requirements-demo.txt
streamlit run app.py
```

Ouvrez ensuite l'adresse locale affichée dans le terminal (généralement
`http://localhost:8501`).

La démo permet de soumettre une image et d'afficher les probabilités
prédites pour les trois classes. Elle charge le checkpoint fourni dans
le dépôt.

### Docker (optionnel)

``` bash
docker build -t pulmoscan-ai .
docker run --rm -p 8501:8501 pulmoscan-ai
```

Puis ouvrez `http://localhost:8501` dans votre navigateur.

## Structure du dépôt

``` text
pulmoscan-ai/
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
│       └── mobilenet_v3_large.pth
├── reports/
│   └── training_history_original_run.json
├── scripts/
│   └── make_figures.py
├── Dockerfile
├── requirements.txt
├── requirements-demo.txt
└── README.md
```

## Limites et pistes d'amélioration

Les performances rapportées sont élevées, mais elles ne suffisent pas à
démontrer que le modèle serait fiable dans un contexte médical réel.

1.  **Découpage par coupe et non par patient** --- des coupes d'un même
    patient peuvent apparaître dans les ensembles d'entraînement et de
    test. Les performances de test peuvent donc être surestimées.
2.  **Source de données limitée** --- les images proviennent de deux
    hôpitaux de Wuhan. La capacité du modèle à généraliser à d'autres
    hôpitaux, populations et appareils n'est pas établie.
3.  **Absence de contexte 3D** --- le modèle analyse chaque coupe
    indépendamment, sans exploiter le volume complet ni les informations
    cliniques du patient.
4.  **Déséquilibre des classes** --- l'accuracy seule peut masquer de
    mauvaises performances sur certaines catégories. La précision, le
    rappel, le F1 et la matrice de confusion sont donc nécessaires.
5.  **Risque d'apprentissage de raccourcis** --- le modèle pourrait
    exploiter des artefacts visuels ou des indices d'acquisition plutôt
    que les structures anatomiques pertinentes. Une analyse de
    saillance, par exemple avec Grad-CAM, serait utile.
6.  **Calibration et validation externe absentes** --- les probabilités
    prédites ne sont pas nécessairement bien calibrées et aucune
    validation sur un jeu de données externe n'est documentée.
7.  **Validation des résultats** --- les performances doivent être
    confirmées par des exécutions reproductibles et, idéalement, par une
    validation externe.

### Améliorations possibles

-   Effectuer un découpage strict par patient.
-   Évaluer le modèle sur un jeu de données externe.
-   Examiner les prédictions avec Grad-CAM.
-   Mesurer la calibration et fournir des intervalles de confiance.
-   Publier les métriques détaillées par classe après exécution de
    l'évaluation.

## Ma contribution

Ce projet vise à rendre le projet plus accessible, documenté et
exploitable en français. Les changements apportés comprennent :

-   Rebranding sous le nom **PulmoScan AI** et création des visuels
    français.
-   Traduction et réorganisation de la documentation et des cellules
    explicatives du notebook.
-   Restructuration du README et ajout d'une analyse critique des
    limites.
-   Création de figures françaises à partir des données et logs du
    projet.
-   Ajout dans le notebook d'une évaluation par classe et de l'export
    des métriques.
-   Amélioration du mode *smoke test* et ajout de `PULMOSCAN_DATA_ROOT`.
-   Ajout d'une démo Streamlit et des fichiers nécessaires à son
    exécution avec Docker.
-   Clarification des crédits, de l'attribution et de la provenance des
    résultats.


## Crédits et références

Le développement de PulmoScan AI est indépendant. Les références ci-dessous créditent les ressources, données et technologies utilisées ; elles ne représentent pas une attribution du projet à un dépôt tiers.


### Jeu de données

-   Abu Zahid Bin Aziz, [CT Scans for COVID-19
    Classification](https://www.kaggle.com/datasets/azaemon/preprocessed-ct-scans-for-covid19),
    Kaggle, licence CC BY 4.0.
-   Ning, W. et al., *iCTCF: an integrative resource of chest computed
    tomography images and clinical features of patients with COVID-19
    pneumonia* (2020).

### Architecture et outils

-   Howard, A. et al., [Searching for
    MobileNetV3](https://arxiv.org/abs/1905.02244), ICCV 2019.
-   [PyTorch](https://pytorch.org/) et
    [torchvision](https://pytorch.org/vision/stable/models/mobilenetv3.html).
-   NumPy, Matplotlib, Pillow, Jupyter et Streamlit.

Consultez le fichier [`LICENSE`](LICENSE) pour les conditions de licence
applicables à ce projet. Les ressources et bibliothèques tierces utilisées sont référencées ci-dessus.

## Auteur

**Mohammed Oughlane**\
Marrakech, Maroc

-   GitHub : `<LIEN_GITHUB>`
-   LinkedIn : `<LIEN_LINKEDIN>`

------------------------------------------------------------------------

## Avertissement

**PulmoScan AI est un projet pédagogique et ne constitue pas un
dispositif médical.** Il ne doit pas être utilisé pour établir un
diagnostic, trier des patients ou prendre une décision de santé. Les
résultats présentés ne constituent pas une validation clinique et ne
remplacent pas l'avis d'un professionnel de santé.

Les données utilisées sont publiques et anonymisées. Aucune donnée
patient identifiable n'est incluse dans ce dépôt.

------------------------------------------------------------------------

```{=html}
<p align="center">
```
`<sub>`{=html}Projet pédagogique de deep learning · Computer Vision ·
PyTorch`</sub>`{=html}
```{=html}
</p>
```
