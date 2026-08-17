#!/bin/bash

INPUT_PTH="C:\Users\janis\Projekty\Magisterka\SonarOdometry\SonarOdometryDataset\selfsupervised\val\seq_15\results\noise_comparie\eval_seq15_mid_noise_artefacts.csv"
OUTPUT_PTH="C:\Users\janis\Projekty\Magisterka\SonarOdometry\SonarOdometryDataset\selfsupervised\val\seq_15\results\noise_comparie\eval_seq15_mid_noise_artefacts_evo.csv"
PYTHON_FORMATIN_SCRIPT_PTH="C:\Users\janis\Projekty\Magisterka\SonarOdometry\notebooks\evaluation\format_to_evo.py"

python3 "$PYTHON_FORMATIN_SCRIPT_PTH" "$PLIK_CSV" -o "$KATALOG_WYJSCIOWY"

echo "Succes!"

python ./notebooks/evaluation/format_to_evo.py ./SonarOdometryDataset/selfsupervised/val/seq_15/results/noise_comparie/eval_seq15_mid_noise_artefacts.csv ./SonarOdometryDataset/selfsupervised/val/seq_15/results/noise_comparie/eval_seq15_mid_noise_artefacts_evo.csv




# evo_ape \ 
# C:/Users/janis/Projekty/Magisterka/SonarOdometry/SonarOdometryDataset/selfsupervised/val/seq_15/results/noise_comparie/eval_seq15_mid_noise_artefacts_gt_tum.txt \
# C:/Users/janis/Projekty/Magisterka/SonarOdometry/SonarOdometryDataset/selfsupervised/val/seq_15/results/noise_comparie/eval_seq15_mid_noise_artefacts_pred_tum.txt \
#   --mode 2d \
#   --pose_relation trans_part \
#   --stats \
#   --save_results C:/Users/janis/Projekty/Magisterka/SonarOdometry/SonarOdometryDataset/selfsupervised/val/seq_15/results/noise_comparie/eval_seq15_mid_noise_artefacts_evoout.zip \
#   --save_plot C:/Users/janis/Projekty/Magisterka/SonarOdometry/SonarOdometryDataset/selfsupervised/val/seq_15/results/noise_comparie/eval_seq15_mid_noise_artefacts_evoout.png \
#   --plot --plot_mode xy

evo_ape tum `
  C:/Users/janis/Projekty/Magisterka/SonarOdometry/SonarOdometryDataset/selfsupervised/val/seq_15/results/noise_comparie/eval_seq15_mid_noise_artefacts_gt_tum.txt `
  C:/Users/janis/Projekty/Magisterka/SonarOdometry/SonarOdometryDataset/selfsupervised/val/seq_15/results/noise_comparie/eval_seq15_mid_noise_artefacts_pred_tum.txt `
  --mode 2d `
  --pose_relation trans_part `
  --stats `
  --save_results C:/Users/janis/Projekty/Magisterka/SonarOdometry/SonarOdometryDataset/selfsupervised/val/seq_15/results/noise_comparie/eval_seq15_mid_noise_artefacts_evoout.zip `
  --save_plot C:/Users/janis/Projekty/Magisterka/SonarOdometry/SonarOdometryDataset/selfsupervised/val/seq_15/results/noise_comparie/eval_seq15_mid_noise_artefacts_evoout.png `
  --plot --plot_mode xy