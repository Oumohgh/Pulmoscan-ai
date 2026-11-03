"""Génère les figures françaises de PulmoScan AI.

Les courbes de validation sont reconstruites à partir des logs du projet
d'origine (`reports/training_history_original_run.json`). Les autres figures
sont dérivées des effectifs du jeu de données et des métriques publiées dans
le README d'origine. Aucune métrique n'est inventée.

Usage :
    python scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
REPORTS = ROOT / "reports"
ASSETS.mkdir(exist_ok=True)

# Palette de la nouvelle identité visuelle
NAVY = "#0B1F33"
TEAL = "#0E7C86"
CYAN = "#22B8CF"
CORAL = "#FF6B6B"
AMBER = "#F59F00"
LIGHT = "#F4F7FA"
GREY = "#5B6B7C"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#C9D4DF",
    "axes.labelcolor": NAVY,
    "axes.titlecolor": NAVY,
    "xtick.color": GREY,
    "ytick.color": GREY,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def save(fig, name: str) -> None:
    path = ASSETS / name
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure écrite : {path.relative_to(ROOT)}")


def make_banner() -> None:
    fig, ax = plt.subplots(figsize=(16, 4.2))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Dégradé horizontal
    grad = plt.imshow(
        [[i / 500 for i in range(500)]],
        extent=(0, 1, 0, 1),
        aspect="auto",
        cmap=plt.cm.colors.LinearSegmentedColormap.from_list("pulmo", [NAVY, TEAL]),
        alpha=0.98,
        zorder=0,
    )

    # Cercles décoratifs inspirés des poumons / des coupes scanner
    for x, y, r, a in [(0.88, 0.45, 0.32, 0.08), (0.08, 0.55, 0.28, 0.06), (0.95, 0.15, 0.18, 0.10)]:
        ax.add_patch(plt.Circle((x, y), r, color=CYAN, alpha=a, lw=0, zorder=1))

    ax.text(0.055, 0.63, "PulmoScan AI", fontsize=54, fontweight="bold",
            color="white", va="center", zorder=2)
    ax.text(0.057, 0.36, "Classification de coupes de scanner thoracique par deep learning",
            fontsize=21, color="#D8F3F9", va="center", zorder=2)
    ax.text(0.057, 0.16, "PyTorch  ·  MobileNetV3-Large  ·  Transfer learning",
            fontsize=14, color="#9FDDE6", va="center", zorder=2)
    ax.text(0.945, 0.16, "30/10/2026 – 03/11/2026",
            fontsize=13, color="#9FDDE6", va="center", ha="right", zorder=2)

    save(fig, "banner.png")


def make_class_distribution() -> None:
    counts = {"Normal (nCT)": 9979, "Non informatif (NiCT)": 5705, "COVID-19 (pCT)": 4001}
    colors = [TEAL, AMBER, CORAL]
    total = sum(counts.values())

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = list(counts.keys())
    values = list(counts.values())
    bars = ax.barh(labels, values, color=colors, height=0.55)

    for bar, value in zip(bars, values):
        pct = value / total * 100
        ax.text(value + 150, bar.get_y() + bar.get_height() / 2,
                f"{value:,}  ({pct:.1f} %)".replace(",", " "),
                va="center", fontsize=11, color=NAVY)

    ax.set_xlim(0, max(values) * 1.22)
    ax.set_xlabel("Nombre de coupes")
    ax.set_title("Répartition des classes dans le jeu de données", fontsize=16, pad=14)
    ax.grid(axis="x", color="#E3EAF1", lw=1)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.text(0.01, -0.02, f"Total : {total:,} coupes — dataset Kaggle CT Scans for COVID-19 (CC BY 4.0).".replace(",", " "),
             fontsize=9, color=GREY)
    save(fig, "class-distribution-fr.png")


def make_training_curves() -> None:
    data = json.loads((REPORTS / "training_history_original_run.json").read_text(encoding="utf-8"))
    epochs = [row["epoch"] for row in data["epochs"]]
    losses = [row["val_loss"] for row in data["epochs"]]
    accs = [row["val_acc"] * 100 for row in data["epochs"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    ax1.plot(epochs, losses, color=CORAL, lw=2.2, marker="o", ms=3.5)
    ax1.set_title("Perte de validation", fontsize=14)
    ax1.set_xlabel("Époque")
    ax1.set_ylabel("Perte (NLLLoss)")
    ax1.grid(color="#E3EAF1", lw=1)
    ax1.set_axisbelow(True)
    ax1.annotate(f"{losses[-1]:.4f}", (epochs[-1], losses[-1]),
                 textcoords="offset points", xytext=(-38, 10), fontsize=10, color=CORAL)

    ax2.plot(epochs, accs, color=TEAL, lw=2.2, marker="o", ms=3.5)
    ax2.set_title("Exactitude de validation", fontsize=14)
    ax2.set_xlabel("Époque")
    ax2.set_ylabel("Exactitude (%)")
    ax2.set_ylim(97.8, 100.0)
    ax2.grid(color="#E3EAF1", lw=1)
    ax2.set_axisbelow(True)
    ax2.annotate(f"{accs[-1]:.2f} %", (epochs[-1], accs[-1]),
                 textcoords="offset points", xytext=(-58, -18), fontsize=10, color=TEAL)

    for ax in (ax1, ax2):
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.suptitle("Courbes de validation — run d'origine (25 époques)", fontsize=16, y=1.03)
    fig.text(0.01, -0.04,
             "Données extraites des logs du projet d'origine (MacBook Pro M5 Pro). "
             "Elles n'ont pas été reproduites pour cette adaptation.",
             fontsize=9, color=GREY)
    fig.tight_layout()
    save(fig, "training-curves-fr.png")


def make_results() -> None:
    splits = ["Entraînement", "Validation", "Test"]
    accuracy = [98.70, 99.25, 99.76]
    loss = [0.0316, 0.0216, 0.0070]
    colors = [TEAL, CYAN, CORAL]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    bars1 = ax1.bar(splits, accuracy, color=colors, width=0.55)
    ax1.set_ylim(97.5, 100.1)
    ax1.set_title("Exactitude par découpage", fontsize=14)
    ax1.set_ylabel("Exactitude (%)")
    for bar, value in zip(bars1, accuracy):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.08,
                 f"{value:.2f} %", ha="center", fontsize=11, color=NAVY)

    bars2 = ax2.bar(splits, loss, color=colors, width=0.55)
    ax2.set_title("Perte par découpage", fontsize=14)
    ax2.set_ylabel("Perte (NLLLoss)")
    for bar, value in zip(bars2, loss):
        ax2.text(bar.get_x() + bar.get_width() / 2, value + 0.001,
                 f"{value:.4f}", ha="center", fontsize=11, color=NAVY)

    for ax in (ax1, ax2):
        ax.grid(axis="y", color="#E3EAF1", lw=1)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.suptitle("Résultats du projet d'origine — métriques héritées", fontsize=16, y=1.03)
    fig.text(0.01, -0.04,
             "Ces valeurs proviennent du run d'origine (MacBook Pro M5 Pro, 25 époques). "
             "Elles n'ont pas été recalculées pour PulmoScan AI.",
             fontsize=9, color=GREY)
    fig.tight_layout()
    save(fig, "results-fr.png")


def make_architecture() -> None:
    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.03, 0.36, 0.14, 0.28, "Entrée\n224 × 224 × 3", LIGHT, NAVY),
        (0.21, 0.30, 0.20, 0.40, "MobileNetV3-Large\npré-entraîné ImageNet", TEAL, "white"),
        (0.45, 0.36, 0.13, 0.28, "960\ncaractéristiques", LIGHT, NAVY),
        (0.62, 0.36, 0.13, 0.28, "Linear 1280\n+ ReLU\n+ Dropout 0.2", LIGHT, NAVY),
        (0.79, 0.36, 0.10, 0.28, "Linear 3\n+ LogSoftmax", LIGHT, NAVY),
        (0.91, 0.36, 0.07, 0.28, "NLLLoss", CORAL, "white"),
    ]

    for x, y, w, h, label, face, text_color in boxes:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.4, edgecolor="#C9D4DF", facecolor=face,
        ))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=11, color=text_color, fontweight="bold")

    for x1, x2 in [(0.17, 0.21), (0.41, 0.45), (0.58, 0.62), (0.75, 0.79), (0.89, 0.91)]:
        ax.add_patch(FancyArrowPatch(
            (x1, 0.50), (x2, 0.50),
            arrowstyle="-|>", mutation_scale=16, lw=1.8, color=GREY,
        ))

    ax.text(0.5, 0.90, "Architecture du classifieur PulmoScan AI", ha="center",
            fontsize=18, color=NAVY, fontweight="bold")
    ax.text(0.5, 0.08,
            "Transfer learning : 4 205 875 paramètres entraînables, réseau entier affiné.",
            ha="center", fontsize=11, color=GREY)

    save(fig, "model-architecture-fr.png")


def main() -> None:
    make_banner()
    make_class_distribution()
    make_training_curves()
    make_results()
    make_architecture()


if __name__ == "__main__":
    main()
