"""
Calculo de metricas por sesion: Precision, Control, Apuntando.

- Precision = aciertos / disparos totales (aciertos = impactos dentro del
  objetivo, es decir dentro de +-width/2 y +-height/2).
- Control = excentricidad de la elipse robusta de impactos (MCD + distancia
  de Mahalanobis para descartar outliers antes de ajustar la elipse).
  Excentricidad = sqrt(1 - (eje_menor/eje_mayor)^2). 0 = agrupacion circular
  perfecta (buen control), cercano a 1 = agrupacion muy alargada (mal control
  direccional).
- Apuntando = distancia euclidiana del centroide de la nube (tras descartar
  outliers) respecto al punto de puntas (por defecto el centro del objetivo,
  0,0), en cm. Cuantifica el sesgo/offset de puntería.

Uso:
    python compute_metrics.py --sessions_dir ../../data/sessions --out ../../data/processed/metrics.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.covariance import MinCovDet
from scipy.stats import chi2


MAHALANOBIS_ALPHA = 0.975  # umbral de descarte de outliers (97.5% chi2, 2 gl)


def load_session(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def robust_ellipse_and_outliers(points):
    """
    Ajusta un estimador robusto de covarianza (MCD) sobre los puntos,
    calcula distancia de Mahalanobis de cada punto respecto al centro
    robusto, y descarta como outliers los que superan el umbral chi2.

    Devuelve: centroide robusto, matriz de covarianza robusta,
    mascara de inliers, excentricidad de la elipse.
    """
    n = len(points)
    if n < 4:
        # Muy pocos puntos para MCD; usar centroide y covarianza simples
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

    eigenvalues, _ = np.linalg.eigh(cov)
    eigenvalues = np.clip(eigenvalues, a_min=1e-9, a_max=None)
    minor, major = np.sort(eigenvalues)
    eccentricity = float(np.sqrt(1 - (minor / major))) if major > 0 else 0.0

    return center, cov, inlier_mask, eccentricity


def compute_session_metrics(session, aim_point=(0.0, 0.0)):
    impacts = session["impacts_cm"]
    width = session["target_width_cm"]
    height = session["target_height_cm"]
    n_total = len(impacts)

    points = np.array([[p["x"], p["y"]] for p in impacts])

    # Precision: aciertos dentro del objetivo
    within = (
        (np.abs(points[:, 0]) <= width / 2)
        & (np.abs(points[:, 1]) <= height / 2)
    )
    n_aciertos = int(within.sum())
    precision = n_aciertos / n_total if n_total else 0.0

    # Control + Apuntando: sobre nube robusta (descartando outliers)
    center, cov, inlier_mask, eccentricity = robust_ellipse_and_outliers(points)
    n_outliers = int((~inlier_mask).sum())

    aim = np.array(aim_point)
    apuntando_offset_cm = float(np.linalg.norm(center - aim))

    return {
        "session_id": session["session_id"],
        "date": session["date"],
        "arma": session["arma"],
        "distancia_m": session["distancia_m"],
        "n_disparos": n_total,
        "n_aciertos": n_aciertos,
        "precision": round(precision, 4),
        "control_excentricidad": round(eccentricity, 4),
        "apuntando_offset_cm": round(apuntando_offset_cm, 3),
        "centroide_cm": {"x": round(float(center[0]), 3), "y": round(float(center[1]), 3)},
        "n_outliers_descartados": n_outliers,
        "velocidad_tiempo_serie_seg": session.get("tiempo_serie_seg"),
        "reflejos_tiempo_primer_disparo_seg": session.get("tiempo_primer_disparo_seg"),
    }


def main():
    parser = argparse.ArgumentParser(description="Calcula metricas por sesion")
    parser.add_argument("--sessions_dir", default="data/sessions")
    parser.add_argument("--out", default="data/processed/metrics.json")
    parser.add_argument("--aim_x", type=float, default=0.0)
    parser.add_argument("--aim_y", type=float, default=0.0)
    args = parser.parse_args()

    sessions_dir = Path(args.sessions_dir)
    session_files = sorted(sessions_dir.glob("sesion_*.json"))

    results = []
    for path in session_files:
        session = load_session(path)
        metrics = compute_session_metrics(session, aim_point=(args.aim_x, args.aim_y))
        results.append(metrics)
        print(
            f"{metrics['session_id']}: precision={metrics['precision']:.2f} "
            f"control(exc)={metrics['control_excentricidad']:.2f} "
            f"apuntando={metrics['apuntando_offset_cm']:.2f}cm "
            f"velocidad={metrics['velocidad_tiempo_serie_seg']:.1f}s "
            f"reflejos={metrics['reflejos_tiempo_primer_disparo_seg']:.2f}s "
            f"outliers={metrics['n_outliers_descartados']}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()