#### PulmoScan AI

<img width="1100" height="902" alt="ct-examples" src="https://github.com/user-attachments/assets/e88c776b-71c8-4f00-9e49-65c6ccd2c9b9" />

Classification de coupes de scanner thoracique par deep learning

Période de réalisation : 30/10/2026 – 03/11/2026

Résumé

PulmoScan AI est un classifieur d'images PyTorch qui trie des coupes de scanner thoracique en trois classes : normal, COVID-19 positif et non informatif. Il repose sur du transfer learning avec MobileNetV3-Large. Ce projet est une version personnalisée, documentée en français et étendue d'un projet open source existant.

Avertissement : projet pédagogique. Il ne constitue pas un dispositif médical et ne doit en aucun cas servir à prendre des décisions concernant la santé de quelqu'un.

Exemples de coupes CT par classe




Chaque image est une coupe axiale unique issue d'un scanner thoracique. Le modèle l'affecte à l'une des trois classes :

Classe	Signification	Images
nCT	Normal : les poumons sont visibles et ne montrent aucun signe de COVID-19.	9 979
pCT	Positif : les poumons présentent des anomalies typiques d'une pneumonie COVID-19.	4 001
NiCT	Non informatif : la coupe ne montre pas assez des poumons pour juger.	5 705

La troisième classe est importante en pratique : un scanner contient de nombreuses coupes au-dessus et au-dessous des poumons, et un modèle utile doit les reconnaître au lieu de les forcer vers « normal » ou « positif ».

Résultats

Les chiffres ci-dessous sont les résultats du projet d'origine. Remplacez-les par ceux de votre propre entraînement si vous relancez le notebook.

Jeu de données	Images	Précision	Perte
Entraînement	13 778	98,70 %	0,0316
Validation	2 951	99,25 %	0,0216
Test	2 956	99,76 %	0,0070

Précision et perte de validation par epoch



La précision d'entraînement est légèrement inférieure à celle de validation et de test, car elle est mesurée avec les augmentations (recadrages aléatoires et retournements) encore actives.

Prédictions sur une coupe de test par classe :

Prédictions du modèle



Matrice de confusion et métriques par classe (à ajouter après votre propre évaluation) :

Matrice de confusion


Fonctionnement
Données

Le jeu de données est séparé par classe en environ 70 % d'entraînement, 15 % de validation et 15 % de test, avec une graine fixe (42) et des effectifs fixes par classe : le découpage est identique à chaque exécution. Les images sont redimensionnées et normalisées avec les statistiques ImageNet attendues par le réseau pré-entraîné. Les images d'entraînement subissent en plus des recadrages aléatoires et des retournements horizontaux.

Modèle

Schéma du modèle



Le backbone est MobileNetV3-Large, un réseau convolutif compact conçu pour les appareils mobiles et pré-entraîné sur ImageNet. Il transforme chaque image 224 × 224 en un vecteur de 960 caractéristiques. Sa tête d'origine à 1 000 classes est remplacée par un petit classifieur : une couche cachée de 1 280 unités, ReLU, dropout de 20 % et trois sorties. Tout le réseau est ré-entraîné (fine-tuning), soit environ 4,2 millions de paramètres entraînables.

Entraînement
Paramètre	Valeur
Fonction de perte	Log-vraisemblance négative sur sorties log-softmax (équivalent à l'entropie croisée)
Optimiseur	AdamW, taux d'apprentissage 1e-4, weight decay 1e-4
Taille de batch	16
Epochs	25
Précision numérique	float32
Matériel	(à renseigner selon votre machine)
Installation
Prérequis
Python 3.11
GPU recommandé (Apple Silicon via MPS, ou NVIDIA via CUDA). Sinon repli sur le CPU, plus lent.
Environ 4 Go d'espace disque pour le jeu de données
Mise en place
bash
git clone [URL_DE_MON_REPO]
cd pulmoscan-ai

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Récupération des données

Les images viennent de Kaggle. Créez un jeton d'API dans Kaggle > Settings > API > Create New Token, puis :

bash
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d azaemon/preprocessed-ct-scans-for-covid19 -p ~/datasets/covid-ct --unzip

L'archive contient des coupes originales et prétraitées. Ce projet utilise les originales. Copiez leurs dossiers nCT, NiCT et pCT dans un dossier CT Scans à la racine du projet :

text
CT Scans/
├── nCT/
├── NiCT/
└── pCT/
Entraînement

Ouvrez classifier.ipynb dans Jupyter ou VS Code et exécutez toutes les cellules. Le notebook vérifie le nombre d'images, entraîne pendant 25 epochs, évalue les trois jeux et enregistre le modèle dans outputs/checkpoints/mobilenet_v3_large.pth. Avec SMOKE_TEST = True dans la cellule de paramètres, un seul epoch est exécuté, pratique pour vérifier l'installation.

Utilisation du modèle entraîné
python
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

ckpt = torch.load("outputs/checkpoints/mobilenet_v3_large.pth", map_location="cpu")

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(960, 1280), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1280, 3))

    def forward(self, x):
        return self.net(x)

model = models.mobilenet_v3_large(weights=None)
model.classifier = Head()
model.load_state_dict(ckpt["state_dict"])
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

image = preprocess(Image.open("chemin/vers/coupe.jpg").convert("RGB")).unsqueeze(0)
with torch.no_grad():
    probs = torch.softmax(model(image), dim=1)[0]

for name, idx in ckpt["class_to_idx"].items():
    print(f"{name} : {probs[idx]:.1%}")
Structure du dépôt
text
.
├── classifier.ipynb                         Notebook : données, entraînement, évaluation, inférence
├── classifier.html                          Export statique du notebook avec ses sorties
├── outputs/checkpoints/
│   └── mobilenet_v3_large.pth               Modèle entraîné
├── assets/                                  Images utilisées dans ce README
│   ├── exemples_classes.png
│   ├── courbes_validation.png
│   ├── predictions.png
│   ├── matrice_confusion.png
│   └── schema_modele.png
├── requirements.txt
├── LICENSE
└── README.md
Analyse critique
Découpage par coupe et non par patient. Le jeu de données contient de nombreuses coupes par patient, et les coupes voisines se ressemblent beaucoup. Des coupes d'une même personne peuvent se retrouver à la fois en entraînement et en test : la précision de test est donc probablement supérieure à ce qu'on obtiendrait sur de nouveaux patients.
Sources de données limitées. Tous les scanners proviennent de deux hôpitaux de Wuhan. Des images issues d'autres appareils, protocoles ou populations peuvent différer, et le modèle n'a pas été évalué dessus.
Coupes isolées. Le modèle juge une coupe à la fois. Une évaluation clinique considère le scanner complet, les symptômes et les analyses.
Ma contribution

(Ne listez que ce que vous avez réellement fait.)

Documentation complète en français (README, notebook)
Rebranding et restructuration du dépôt (PulmoScan AI)
Analyse critique des limites du projet
 Matrice de confusion et métriques par classe (précision, rappel, F1)
 Visualisation Grad-CAM
 Démo Streamlit / Dockerfile
Crédits et attribution

PulmoScan AI est basé sur le projet open source [URL_DU_REPO_ORIGINAL] de son auteur d'origine, utilisé conformément à sa licence (voir le fichier LICENSE).

Jeu de données : CT Scans for COVID-19 Classification, Abu Zahid Bin Aziz, licence CC BY 4.0. Images originales issues de iCTCF : Ning, W. et al., « iCTCF: an integrative resource of chest computed tomography images and clinical features of patients with COVID-19 pneumonia » (2020).
Modèle : Howard, A. et al., « Searching for MobileNetV3 », ICCV 2019. Poids pré-entraînés fournis par torchvision.
À propos de l'auteur

Mohammed Oughlane, Marrakech, Maroc. Développeur développeur AI/ServiceNow chez UX4MATION
