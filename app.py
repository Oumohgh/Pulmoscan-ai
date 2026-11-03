"""Démo Streamlit de PulmoScan AI : un poste de lecture de coupes thoraciques.

Lancement :
    streamlit run app.py

La démo charge le checkpoint inclus dans le dépôt et classe une coupe de
scanner thoracique en trois catégories : normal, COVID-19 ou non informatif.
"""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

APP_NAME = "PulmoScan AI"  # un seul endroit à modifier si le projet est renommé

st.set_page_config(page_title=APP_NAME, page_icon="🫁", layout="wide", initial_sidebar_state="collapsed")

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
CLASS_COLORS = {"nCT": "#7fb88f", "NiCT": "#9aa7b8", "pCT": "#cf5c6e"}
LOW_CONFIDENCE = 0.70

# ---------------------------------------------------------------------------
# Style : une salle de lecture de radiologie. Fond sombre neutre pour laisser
# l'image en niveaux de gris au premier plan, texte d'annotation ambre comme
# dans les visionneuses DICOM, compte rendu en texte serif.
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --console: #1a1d21;
    --panel: #20242a;
    --film: #08090a;
    --line: #333940;
    --ivory: #e8e4da;
    --muted: #8b9099;
    --amber: #e0b45a;
}
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.stApp { background: var(--console); color: var(--ivory); font-family: "Newsreader", Georgia, serif; }
.block-container { max-width: 1180px !important; padding: 28px 24px 60px !important; }
h1, h2, h3, p { font-family: "Newsreader", Georgia, serif; }

.topbar { display: flex; justify-content: space-between; align-items: baseline;
          padding-bottom: 14px; border-bottom: 1px solid var(--line); margin-bottom: 26px; }
.topbar .name { font-size: 1.35rem; letter-spacing: .01em; }
.topbar .note { font-family: "IBM Plex Mono", monospace; font-size: .75rem; color: var(--muted); }

.intro { max-width: 640px; font-size: 1.2rem; line-height: 1.55; color: var(--ivory); margin: 0 0 26px; }

/* Visionneuse */
.viewport { position: relative; aspect-ratio: 1 / 1; background: var(--film);
            border: 1px solid var(--line); overflow: hidden; }
.viewport img { width: 100%; height: 100%; object-fit: contain; display: block; }
.viewport .empty { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
                   color: var(--muted); font-size: 1.05rem; text-align: center; padding: 0 12%; }
.viewport .cross::before, .viewport .cross::after { content: ""; position: absolute; background: #2c3138; }
.viewport .cross::before { left: 50%; top: 18%; bottom: 18%; width: 1px; }
.viewport .cross::after { top: 50%; left: 18%; right: 18%; height: 1px; }
.ov { position: absolute; font-family: "IBM Plex Mono", monospace; font-size: .72rem; line-height: 1.6;
      color: var(--amber); text-shadow: 0 0 3px #000; pointer-events: none; }
.ov.tl { top: 12px; left: 14px; } .ov.tr { top: 12px; right: 14px; text-align: right; }
.ov.bl { bottom: 12px; left: 14px; } .ov.br { bottom: 12px; right: 14px; text-align: right; }

/* Compte rendu */
.report { background: var(--panel); border: 1px solid var(--line); padding: 30px 30px 26px; min-height: 100%; }
.report .title { font-family: "IBM Plex Mono", monospace; font-size: .75rem; color: var(--muted); margin-bottom: 18px; }
.report .finding { font-size: 2.1rem; line-height: 1.15; font-weight: 500; margin: 0 0 8px; }
.report .why { font-size: 1.08rem; line-height: 1.55; color: #c9c5ba; margin: 0 0 26px; }
.row { margin: 0 0 16px; }
.row .head { display: flex; justify-content: space-between; align-items: baseline; font-size: 1.02rem; }
.row .pct { font-family: "IBM Plex Mono", monospace; font-size: .85rem; color: var(--ivory); }
.row .track { height: 3px; background: #2e333a; margin-top: 7px; }
.row .fill { height: 100%; transition: width .5s ease; }
.warn { margin-top: 22px; padding-top: 16px; border-top: 1px solid var(--line); font-size: 1rem;
        line-height: 1.5; color: var(--amber); }
.small { margin-top: 22px; font-size: .92rem; line-height: 1.5; color: var(--muted); }

/* Notes sous la visionneuse */
.notes { display: grid; grid-template-columns: 1fr 1fr; gap: 56px; margin-top: 60px;
         padding-top: 30px; border-top: 1px solid var(--line); }
.notes h3 { font-size: 1.3rem; font-weight: 500; margin: 0 0 10px; color: var(--ivory); }
.notes p { font-size: 1.05rem; line-height: 1.65; color: #c0bcb1; max-width: 520px; margin: 0 0 12px; }
.foot { margin-top: 48px; font-family: "IBM Plex Mono", monospace; font-size: .72rem; line-height: 1.8; color: var(--muted); }
@media (max-width: 800px) { .notes { grid-template-columns: 1fr; gap: 30px; } }

/* Widget de dépôt */
[data-testid="stFileUploaderDropzone"] { background: var(--panel); border: 1px dashed #4a525c; border-radius: 0; }
[data-testid="stFileUploaderDropzone"] * { color: var(--muted) !important; font-family: "IBM Plex Mono", monospace; font-size: .8rem; }
[data-testid="stFileUploaderDropzone"] button { border-radius: 0; border: 1px solid #4a525c; background: transparent; }
[data-testid="stFileUploader"] small { display: none; }
[data-testid="stFileUploaderFile"] * { color: var(--ivory) !important; }
</style>
"""


# ---------------------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------------------
class Classifier(nn.Module):
    """Tête de classification utilisée dans le notebook."""

    def __init__(self, input_size=960, output_size=3, hidden_layers=(1280,), drop_p=0.2):
        super().__init__()
        layers = []
        prev = input_size
        for hidden in hidden_layers:
            layers.extend([nn.Linear(prev, hidden), nn.ReLU(inplace=True), nn.Dropout(p=drop_p)])
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

    idx_to_class = {idx: name for name, idx in ckpt["class_to_idx"].items()}
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image.convert("RGB")).unsqueeze(0)


def predict(image: Image.Image, model, idx_to_class, device):
    tensor = preprocess(image).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu()
    rows = [
        {"key": idx_to_class[i], "Classe": CLASS_NAMES[idx_to_class[i]], "Probabilité": float(probs[i])}
        for i in range(len(probs))
    ]
    return sorted(rows, key=lambda row: row["Probabilité"], reverse=True)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
def image_to_data_uri(image: Image.Image, max_side: int = 900) -> str:
    preview = image.convert("RGB")
    preview.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    preview.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def render_viewport(image: Image.Image | None, filename: str, device_name: str, top: dict | None) -> None:
    device = html.escape(device_name)
    model_info = f"""
        <div class="ov bl">Modèle MobileNetV3-Large<br>Calcul : {device}</div>
        <div class="ov br">{('Résultat : ' + html.escape(top['Classe'])) if top else 'Projet pédagogique'}</div>
    """
    if image is None:
        body = '<div class="cross"></div><div class="empty">Déposez une coupe axiale (PNG ou JPG) pour lancer la lecture.</div>'
        corners = '<div class="ov tl">Aucune image</div>'
    else:
        body = f'<img src="{image_to_data_uri(image)}" alt="Coupe de scanner thoracique chargée">'
        corners = (
            f'<div class="ov tl">Coupe axiale<br>{html.escape(filename)}</div>'
            f'<div class="ov tr">Image source {image.width} × {image.height} px<br>'
            f'Entrée du modèle {IMG_SHAPE} × {IMG_SHAPE} px</div>'
        )
    st.markdown(f'<div class="viewport">{body}{corners}{model_info}</div>', unsafe_allow_html=True)


def render_report(rows: list[dict] | None) -> None:
    if rows is None:
        st.markdown(
            """
            <div class="report">
                <div class="title">Compte rendu automatique</div>
                <p class="finding" style="color:#8b9099">En attente d'une image</p>
                <p class="why">Le modèle attribue à la coupe une probabilité pour chacune des trois classes :
                scanner normal, scanner positif à la COVID-19, ou scanner non informatif lorsque
                la coupe ne montre pas assez de poumons.</p>
                <p class="small">Projet pédagogique. Ce résultat ne constitue pas un avis médical.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    top = rows[0]
    bars = "".join(
        f"""
        <div class="row">
            <div class="head"><span>{row['Classe']}</span><span class="pct">{row['Probabilité'] * 100:.1f} %</span></div>
            <div class="track"><div class="fill" style="width:{row['Probabilité'] * 100:.1f}%;background:{CLASS_COLORS[row['key']]}"></div></div>
        </div>
        """
        for row in rows
    )
    warning = ""
    if top["Probabilité"] < LOW_CONFIDENCE:
        warning = (
            '<div class="warn">Confiance faible : le modèle hésite entre plusieurs classes. '
            "Ne retenez pas ce résultat.</div>"
        )
    st.markdown(
        f"""
        <div class="report">
            <div class="title">Compte rendu automatique</div>
            <p class="finding" style="color:{CLASS_COLORS[top['key']]}">{top['Classe']}</p>
            <p class="why">{CLASS_DESCRIPTIONS[top['key']]}</p>
            {bars}
            {warning}
            <p class="small">Projet pédagogique. Ce résultat ne constitue pas un avis médical et ne doit pas
            servir à prendre une décision de santé.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_notes() -> None:
    st.markdown(
        """
        <div class="notes">
            <div>
                <h3>Comment le modèle lit l'image</h3>
                <p>L'image est redimensionnée, recadrée au centre en 224 × 224 pixels, puis normalisée
                avec les statistiques d'ImageNet. MobileNetV3-Large en extrait 960 caractéristiques, et une
                petite tête de classification (1 280 neurones, dropout 20 %) en déduit trois probabilités.</p>
                <p>Dans le projet d'origine, dont ce travail est dérivé, le modèle atteint 99,76 % de précision
                sur 2 956 coupes de test. Ce chiffre n'a pas été reproduit ici.</p>
            </div>
            <div>
                <h3>Ce que ce modèle ne sait pas faire</h3>
                <p>Le jeu de données est découpé par coupe et non par patient : des coupes très proches ont pu
                se retrouver à la fois en entraînement et en test, donc le score est probablement trop optimiste.</p>
                <p>Toutes les images viennent de deux hôpitaux de Wuhan, et le modèle juge une coupe isolée,
                jamais un scanner complet ni le dossier du patient.</p>
            </div>
        </div>
        <div class="foot">
            {app} réalisé du 30/10/2026 au 03/11/2026 par Mohammed Oughlane, Marrakech.<br>
            Données : CT Scans for COVID-19, licence CC BY 4.0. Modèle : MobileNetV3, Howard et al., ICCV 2019.
        </div>
        """.replace("{app}", html.escape(APP_NAME)),
        unsafe_allow_html=True,
    )


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="topbar">
            <div class="name">{html.escape(APP_NAME)}</div>
            <div class="note">Projet pédagogique</div>
        </div>
        <p class="intro">Déposez une coupe de scanner thoracique. Le modèle indique s'il la juge normale,
        positive à la COVID-19 ou inexploitable, et à quel point il est sûr de lui.</p>
        """,
        unsafe_allow_html=True,
    )

    if not CHECKPOINT_PATH.exists():
        st.error(
            f"Modèle introuvable : `{CHECKPOINT_PATH}`. "
            "Placez le fichier `mobilenet_v3_large.pth` à cet emplacement, puis relancez l'application."
        )
        st.stop()

    device_name = choose_device()
    with st.spinner("Chargement du modèle"):
        model, idx_to_class, device = load_model(str(CHECKPOINT_PATH), device_name)

    left, right = st.columns([1.15, 1.0], gap="large")
    with left:
        uploaded = st.file_uploader(
            "Coupe de scanner",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )
        image = Image.open(uploaded) if uploaded is not None else None
        rows = predict(image, model, idx_to_class, device) if image is not None else None
        render_viewport(
            image,
            uploaded.name if uploaded is not None else "",
            device_name,
            rows[0] if rows else None,
        )
    with right:
        render_report(rows)

    render_notes()


if __name__ == "__main__":
    main()