"""
Grafico radar (pentagono) de las 5 metricas: Precision, Control, Apuntando,
Velocidad, Reflejos.

Normalizacion: min-max 0-100 contra el propio historial del tirador
(las 7 sesiones de data/processed/metrics.json). Se invierte el sentido
en las metricas donde "menor valor crudo = mejor desempeno":
Control (excentricidad), Apuntando (offset), Velocidad (tiempo), Reflejos
(tiempo). Precision ya es "mayor = mejor", no se invierte.

Uso:
    python radar_chart.py --metrics ../../data/processed/metrics.json --session sesion_007 --out ../../data/processed/radar_sesion_007.png
    python radar_chart.py --metrics ../../data/processed/metrics.json --session sesion_007 --compare sesion_001 --out ../../data/processed/radar_comparacion.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


EJES = ["Precisión", "Control", "Apuntando", "Velocidad", "Reflejos"]

# Para cada eje: True si el valor crudo mayor es mejor (no se invierte),
# False si el valor crudo menor es mejor (se invierte al normalizar).
MAYOR_ES_MEJOR = {
    "Precisión": True,
    "Control": False,
    "Apuntando": False,
    "Velocidad": False,
    "Reflejos": False,
}

CAMPO_CRUDO = {
    "Precisión": "precision",
    "Control": "control_excentricidad",
    "Apuntando": "apuntando_offset_cm",
    "Velocidad": "velocidad_tiempo_serie_seg",
    "Reflejos": "reflejos_tiempo_primer_disparo_seg",
}


def load_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_all_sessions(all_sessions):
    """
    Devuelve dict: session_id -> {eje: valor_normalizado_0_100}
    normalizado min-max contra el historial completo (all_sessions).
    """
    raw_by_axis = {
        eje: np.array([s[CAMPO_CRUDO[eje]] for s in all_sessions], dtype=float)
        for eje in EJES
    }

    normalized = {s["session_id"]: {} for s in all_sessions}

    for eje in EJES:
        values = raw_by_axis[eje]
        vmin, vmax = values.min(), values.max()
        rango = vmax - vmin

        for s, v in zip(all_sessions, values):
            if rango == 0:
                score = 100.0
            else:
                score = (v - vmin) / rango * 100
                if not MAYOR_ES_MEJOR[eje]:
                    score = 100 - score
            normalized[s["session_id"]][eje] = round(float(score), 1)

    return normalized


def plot_radar(session_scores_list, labels, out_path, title="BlisticData - Radar de desempeño"):
    """
    session_scores_list: lista de dicts {eje: score}, uno por serie a graficar.
    labels: nombres de cada serie (para la leyenda).
    """
    n_axes = len(EJES)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(EJES, fontsize=11)
    ax.set_rlabel_position(0)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8, color="gray")

    colors = plt.cm.tab10.colors

    for i, (scores, label) in enumerate(zip(session_scores_list, labels)):
        values = [scores[eje] for eje in EJES]
        values += values[:1]
        color = colors[i % len(colors)]
        ax.plot(angles, values, linewidth=2, label=label, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_title(title, fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Guardado: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Genera grafico radar de metricas")
    parser.add_argument("--metrics", default="data/processed/metrics.json")
    parser.add_argument("--session", required=True, help="session_id principal a graficar")
    parser.add_argument("--compare", default=None, help="session_id adicional para comparar")
    parser.add_argument("--out", default="data/processed/radar.png")
    args = parser.parse_args()

    all_sessions = load_metrics(args.metrics)
    normalized = normalize_all_sessions(all_sessions)

    if args.session not in normalized:
        raise ValueError(f"session_id '{args.session}' no encontrado en {args.metrics}")

    session_scores_list = [normalized[args.session]]
    labels = [args.session]

    if args.compare:
        if args.compare not in normalized:
            raise ValueError(f"session_id '{args.compare}' no encontrado en {args.metrics}")
        session_scores_list.append(normalized[args.compare])
        labels.append(args.compare)

    plot_radar(session_scores_list, labels, args.out)


if __name__ == "__main__":
    main()