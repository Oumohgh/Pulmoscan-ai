"""Démo Streamlit de PulmoScan AI.

Lancement :
    streamlit run app.py

La démo charge le checkpoint inclus dans le dépôt et classe une coupe de
scanner thoracique en trois catégories : normal, COVID-19 ou non informatif.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

st.set_page_config(
    page_title="PulmoScan AI",
    page_icon="🫁",
    layout="wide",
)

CHECKPOINT_PATH = Path("outputs/checkpoints/mobilenet_v3_large.pth")
IMG_SHAPE = 224
CLASS_NAMES = {
    "nCT": "Scanner normal",
    "NiCT": "Scanner non informatif",
    "pCT": "Scanner positif à la COVID-19",
}
CLASS_DESCRIPTIONS = {
    "nCT": "Les poumons sont visibles et ne montrent pas de signe de COVID-19.",
    "NiCT": "La coupe ne montre pas assez de poumons pour conclure.",
    "pCT": "Les poumons présentent des anomalies typiques d'une pneumonie à COVID-19.",
}


class Classifier(nn.Module):
    """Tête de classification utilisée dans le notebook."""

    def __init__(self, input_size=960, output_size=3, hidden_layers=(1280,), drop_p=0.2):
        super().__init__()
        layers = []
        prev = input_size
        for hidden in hidden_layers:
            layers.extend([
                nn.Linear(prev, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=drop_p),
            ])
            prev = hidden
        layers.append(nn.Linear(prev, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.log_softmax(self.net(x), dim=1)


@st.cache_resource(show_spinner=False)
def load_model(path: str, device_name: str):
    """Charge le checkpoint et reconstruit le modèle."""
    device = torch.device(device_name)
    ckpt = torch.load(path, map_location=device, weights_only=False)

    model = models.mobilenet_v3_large(weights=None)
    model.classifier = Classifier(
        input_size=ckpt["input_size"],
        output_size=ckpt["output_size"],
        hidden_layers=ckpt["hidden_layers"],
        drop_p=ckpt["drop_p"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    return model, idx_to_class, device


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def preprocess(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SHAPE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return transform(image.convert("RGB")).unsqueeze(0)


def predict(image: Image.Image, model, idx_to_class, device):
    tensor = preprocess(image).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu()
    rows = [
        {
            "Classe": CLASS_NAMES[idx_to_class[i]],
            "Probabilité": float(probs[i]),
        }
        for i in range(len(probs))
    ]
    return sorted(rows, key=lambda row: row["Probabilité"], reverse=True)


def main() -> None:
    st.title("🫁 PulmoScan AI")
    st.caption(
        "Classification de coupes de scanner thoracique par deep learning — "
        "MobileNetV3-Large, transfer learning, PyTorch"
    )

    with st.sidebar:
        st.header("À propos")
        st.markdown(
            "**PulmoScan AI** classe une coupe axiale de scanner thoracique en "
            "trois catégories : normal, COVID-19 ou non informatif."
        )
        st.markdown(
            "**Auteur :** Mohammed Oughlane  \n"
            "**Période :** 30/10/2026 – 03/11/2026  \n"
            "**Modèle :** MobileNetV3-Large  \n"
            "**Jeu de données :** CT Scans for COVID-19 (CC BY 4.0)"
        )
        st.divider()
        st.warning(
            "Projet pédagogique. Cette démo ne constitue pas un dispositif médical "
            "et ne doit pas servir à prendre des décisions de santé."
        )

    if not CHECKPOINT_PATH.exists():
        st.error(
            f"Checkpoint introuvable : `{CHECKPOINT_PATH}`. "
            "Vérifiez que le fichier `mobilenet_v3_large.pth` est bien présent."
        )
        st.stop()

    device_name = choose_device()
    with st.spinner("Chargement du modèle…"):
        model, idx_to_class, device = load_model(str(CHECKPOINT_PATH), device_name)

    left, right = st.columns([1.1, 1.0], gap="large")

    with left:
        uploaded = st.file_uploader(
            "Choisissez une coupe de scanner (PNG, JPG ou JPEG)",
            type=["png", "jpg", "jpeg"],
        )
        if uploaded is not None:
            image = Image.open(uploaded)
            st.image(image, caption="Coupe chargée", use_container_width=True)

    with right:
        if uploaded is None:
            st.info(
                "Chargez une image pour lancer la prédiction. "
                "Le modèle attend une coupe axiale, pas un scanner complet."
            )
            st.markdown("### Classes")
            for key, name in CLASS_NAMES.items():
                st.markdown(f"**{name}** (`{key}`) — {CLASS_DESCRIPTIONS[key]}")
            return

        with st.spinner("Prédiction…"):
            rows = predict(image, model, idx_to_class, device)

        top = rows[0]
        st.metric(
            label="Classe prédite",
            value=top["Classe"],
            delta=f"{top['Probabilité'] * 100:.1f} % de confiance",
        )

        st.markdown("### Probabilités par classe")
        df = pd.DataFrame(rows).set_index("Classe")
        st.bar_chart(df, height=260)

        st.markdown("### Détail")
        st.dataframe(
            pd.DataFrame([
                {
                    "Classe": row["Classe"],
                    "Probabilité": f"{row['Probabilité'] * 100:.2f} %",
                }
                for row in rows
            ]),
            hide_index=True,
            use_container_width=True,
        )

    st.divider()
    with st.expander("Comment fonctionne le modèle ?"):
        st.markdown(
            "Le modèle reprend **MobileNetV3-Large** pré-entraîné sur ImageNet. "
            "Sa tête de classification est remplacée par une couche linéaire de "
            "1 280 neurones (ReLU, dropout 20 %), puis une couche de sortie à "
            "trois classes. L'ensemble du réseau est affiné par transfer learning."
        )
        st.markdown(
            "Les images sont redimensionnées à 256 px, recadrées au centre en "
            "224 × 224, converties en tenseurs et normalisées avec les "
            "statistiques ImageNet."
        )

    with st.expander("Limites"):
        st.markdown(
            "- Le découpage du jeu de données est fait par coupe, pas par patient : "
            "les performances sont probablement optimistes.\n"
            "- Les images proviennent de deux hôpitaux de Wuhan ; le modèle n'a pas "
            "été validé sur d'autres scanners ou populations.\n"
            "- Le modèle juge une coupe isolée, pas un scanner complet.\n"
            "- Les métriques du projet d'origine n'ont pas été reproduites ici."
        )


if __name__ == "__main__":
    main()
