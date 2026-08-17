import argparse
import math
import os
import pandas as pd


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    """Konwertuje kąt obrotu 2D (yaw w radianach) na kwaternion (qx, qy, qz, qw)."""
    qx = 0.0
    qy = 0.0
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    return qx, qy, qz, qw


def convert_csv_to_evo_tum(
    csv_filepath: str,
    output_dir: str = None,
    save_gt: bool = True,
    save_pred: bool = True,
) -> dict[str, str]:
    """Konwertuje plik CSV z odometrią do formatu TUM akceptowanego przez pakiet evo.

    :param csv_filepath: Ścieżka do wejściowego pliku CSV.
    :param output_dir: Katalog docelowy na pliki wynikowe (domyślnie katalog
        pliku wejściowego).
    :param save_gt: Czy wygenerować plik trajektorii Ground Truth.
    :param save_pred: Czy wygenerować plik trajektorii Predykcji.
    :return: Słownik ze ścieżkami do wygenerowanych plików.
    """
    if not os.path.exists(csv_filepath):
        raise FileNotFoundError(f"Nie znaleziono pliku: {csv_filepath}")

    # Wczytanie pliku CSV
    df = pd.read_csv(csv_filepath)

    base_dir = (
        output_dir if output_dir else os.path.dirname(os.path.abspath(csv_filepath))
    )
    base_name = os.path.splitext(os.path.basename(csv_filepath))[0]
    generated_files = {}

    # Generowanie pliku dla Ground Truth
    if save_gt and "gt_x" in df.columns:
        gt_filename = os.path.join(base_dir, f"{base_name}_gt_tum.txt")
        with open(gt_filename, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                timestamp = float(row["frame_id"])  # Stosujemy frame_id jako czas
                x, y, theta = (
                    float(row["gt_x"]),
                    float(row["gt_y"]),
                    float(row["gt_theta"]),
                )
                qx, qy, qz, qw = yaw_to_quaternion(theta)
                f.write(
                    f"{timestamp:.6f} {x:.6f} {y:.6f} 0.000000 {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
                )
        generated_files["gt"] = gt_filename
        print(f"[OK] Zapisano Ground Truth do: {gt_filename}")

    # Generowanie pliku dla Predykcji
    if save_pred and "pred_x" in df.columns:
        pred_filename = os.path.join(base_dir, f"{base_name}_pred_tum.txt")
        with open(pred_filename, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                timestamp = float(row["frame_id"])
                x, y, theta = (
                    float(row["pred_x"]),
                    float(row["pred_y"]),
                    float(row["pred_theta"]),
                )
                qx, qy, qz, qw = yaw_to_quaternion(theta)
                f.write(
                    f"{timestamp:.6f} {x:.6f} {y:.6f} 0.000000 {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
                )
        generated_files["pred"] = pred_filename
        print(f"[OK] Zapisano Predykcję do: {pred_filename}")

    return generated_files



root_dir = 'C:/Users/janis/Projekty/Magisterka/SonarOdometry'
pth_in = 'SonarOdometryDataset/selfsupervised/val/seq_15/results/noise_comparie/eval_seq15_mid_noise_artefacts.csv'
pth_out = 'SonarOdometryDataset/selfsupervised/val/seq_15/results/noise_comparie'

convert_csv_to_evo_tum(os.path.join(root_dir, pth_in), os.path.join(root_dir, pth_out))