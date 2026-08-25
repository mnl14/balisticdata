"""
Grafico de lamina individual: dispersion de impactos sobre el objetivo,
elipse robusta (MCD + Mahalanobis), centroide, outliers marcados aparte,
y punto de puntas.

Titulo: ARMA - DISTANCIA - FECHA
Debajo del titulo: datos de referencia (distancia, offset de apuntando,
tiempo de serie).

Genera un PNG por sesion.

Uso:
    python target_plot.py --sessions_dir ../../data/sessions --out_dir ../../data/processed/laminas
    python target_plot.py --sessions_dir ../../data/sessions --session sesion_001 --out_dir ../../data/processed/laminas
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from sklearn.covariance import MinCovDet
from scipy.stats import chi2


MAHALANOBIS_ALPHA = 0.975


def load_session(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def robust_ellipse_and_outliers(points):
    n = len(points)
    if n < 4:
        center = points.mean(axis=0)
        cov = np.cov(points, rowvar=False) if n > 1 else np.eye(2)
        inlier_mask = np.ones(n, dtype=bool)
    else:
        mcd = MinCovDet(support_fraction=0.75).fit(points)
        center = mcd.location_
        cov = mcd.covariance_
        d2 = mcd.mahalanobis(points)
        threshold = chi2.ppf(MAHALANOBIS_ALPHA, df=2)
        inlier_mask = d2 <= threshold
    return center, cov, inlier_mask


def ellipse_params(center, cov, chi2_val):
    """Devuelve ancho, alto y angulo (grados) de la elipse de confianza."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * np.sqrt(np.clip(eigenvalues, 0, None) * chi2_val)
    return width, height, angle


def plot_session(session, out_path, aim_point=(0.0, 0.0)):
    impacts = session["impacts_cm"]
    width_cm = session["target_width_cm"]
    height_cm = session["target_height_cm"]
    points = np.array([[p["x"], p["y"]] for p in impacts])

    center, cov, inlier_mask = robust_ellipse_and_outliers(points)
    chi2_val = chi2.ppf(MAHALANOBIS_ALPHA, df=2)
    e_width, e_height, e_angle = ellipse_params(center, cov, chi2_val)

    aim = np.array(aim_point)
    apuntando_offset = float(np.linalg.norm(center - aim))

    fig, ax = plt.subplots(figsize=(6, 7))

    # Objetivo (rectangulo) centrado en 0,0
    ax.add_patch(
        Rectangle(
            (-width_cm / 2, -height_cm / 2), width_cm, height_cm,
            fill=False, edgecolor="black", linewidth=1.5,
        )
    )

    # Elipse robusta de confianza
    ax.add_patch(
        Ellipse(
            center, e_width, e_height, angle=e_angle,
            fill=True, facecolor="tab:blue", alpha=0.15,
            edgecolor="tab:blue", linewidth=1.5, linestyle="--",
        )
    )

    # Impactos: inliers vs outliers
    ax.scatter(
        points[inlier_mask, 0], points[inlier_mask, 1],
        c="tab:blue", s=60, label="Impacto", zorder=3,
    )
    if (~inlier_mask).any():
        ax.scatter(
            points[~inlier_mask, 0], points[~inlier_mask, 1],
            c="tab:red", marker="x", s=70, linewidths=2,
            label="Outlier", zorder=3,
        )

    # Centroide y punto de puntas
    ax.scatter(*center, c="black", marker="+", s=150, linewidths=2, label="Centroide", zorder=4)
    ax.scatter(*aim, c="green", marker="*", s=150, label="Punto de puntería", zorder=4)

    margin = 5
    ax.set_xlim(-width_cm / 2 - margin, width_cm / 2 + margin)
    ax.set_ylim(-height_cm / 2 - margin, height_cm / 2 + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)

    titulo = f"{session['arma']} - {session['distancia_m']}m - {session['date']}"
    subtitulo = (
        f"Distancia: {session['distancia_m']}m  |  "
        f"Apuntando: {apuntando_offset:.1f}cm  |  "
        f"Tiempo serie: {session.get('tiempo_serie_seg', 'N/A')}s"
    )
    ax.set_title(f"{titulo}\n{subtitulo}", fontsize=11)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Grafica cada lamina de tiro por separado")
    parser.add_argument("--sessions_dir", default="data/sessions")
    parser.add_argument("--session", default=None, help="session_id especifico (si se omite, procesa todas)")
    parser.add_argument("--out_dir", default="data/processed/laminas")
    parser.add_argument("--aim_x", type=float, default=0.0)
    parser.add_argument("--aim_y", type=float, default=0.0)
    args = parser.parse_args()

    sessions_dir = Path(args.sessions_dir)
    out_dir = Path(args.out_dir)

    if args.session:
        paths = [sessions_dir / f"{args.session}.json"]
    else:
        paths = sorted(sessions_dir.glob("sesion_*.json"))

    for path in paths:
        session = load_session(path)
        out_path = out_dir / f"{session['session_id']}.png"
        plot_session(session, out_path, aim_point=(args.aim_x, args.aim_y))


if __name__ == "__main__":
    main()