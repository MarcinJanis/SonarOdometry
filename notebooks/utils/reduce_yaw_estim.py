import pandas as pd
import math

def recalculate_trajectory_with_gt_yaw(input_csv, output_csv, start_n, use_frame_id, inv_dx, inv_dy, swap_xy, inv_gt_yaw):
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Błąd podczas wczytywania pliku: {e}")
        return

    # Ustalenie indeksu początkowego
    if use_frame_id:
        idx_list = df.index[df['frame_id'] == start_n].tolist()
        if not idx_list:
            print(f"Błąd: Nie znaleziono frame_id == {start_n} w danych.")
            return
        start_idx = idx_list[0]
    else:
        start_idx = int(start_n)
        if start_idx < 0 or start_idx >= len(df):
            print(f"Błąd: Indeks wiersza poza zakresem.")
            return
    
    print(f"Odcinam trajektorię przed: {start_idx} (frame_id: {df.loc[start_idx, 'frame_id']})")
    print(f"Konfiguracja -> Inwersje X: {inv_dx}, Y: {inv_dy} | Zamiana XY: {swap_xy} | Odwrócony Yaw GT: {inv_gt_yaw}")
    print("UWAGA: Kąt obrotu (Yaw) jest narzucony z danych Ground Truth (GT)!")

    # Całkowite usunięcie rzędów przed wybranym krokiem (zarówno GT jak i Pred)
    if start_idx > 0:
        df = df.iloc[start_idx:].reset_index(drop=True)
    
    # Po ucięciu rzędów, nasz nowy punkt startowy w DataFrame ma zawsze indeks 0
    start_idx = 0

    # Reset w punkcie startowym do wartości Ground Truth
    df.loc[start_idx, 'pred_x'] = df.loc[start_idx, 'gt_x']
    df.loc[start_idx, 'pred_y'] = df.loc[start_idx, 'gt_y']
    df.loc[start_idx, 'pred_theta'] = df.loc[start_idx, 'gt_theta']

    # Dynamiczne odnalezienie kolumn
    pred_dx_col = 'step_pred_dx' if 'step_pred_dx' in df.columns else [c for c in df.columns if c.startswith('step_pred_d')][0]
    pred_dy_col = 'step_pred_dy' if 'step_pred_dy' in df.columns else [c for c in df.columns if c.startswith('step_pred_d')][1]
    
    for i in range(start_idx + 1, len(df)):
        # Pobieramy idealny kąt obrotu z Ground Truth z poprzedniego kroku
        theta_prev = df.loc[i - 1, 'gt_theta'] 
        
        # ODWRÓCENIE KIERUNKU OBROTU GT
        if inv_gt_yaw:
            theta_prev = -theta_prev
        
        # Pobranie lokalnych odczytów przesunięcia (tylko translacja)
        dx_local = df.loc[i, pred_dx_col]
        dy_local = df.loc[i, pred_dy_col]
        
        # ZAMIANA OSI X i Y
        if swap_xy:
            dx_local, dy_local = dy_local, dx_local

        # Aplikacja wybranych inwersji
        if inv_dx: dx_local = -dx_local
        if inv_dy: dy_local = -dy_local
        
        # Standardowa transformacja rotacyjna 2D
        dx_global = dx_local * math.cos(theta_prev) - dy_local * math.sin(theta_prev)
        dy_global = dx_local * math.sin(theta_prev) + dy_local * math.cos(theta_prev)
        
        # Integracja
        df.loc[i, 'pred_x'] = df.loc[i - 1, 'pred_x'] + dx_global
        df.loc[i, 'pred_y'] = df.loc[i - 1, 'pred_y'] + dy_global
        
        # Zastępujemy predykcję kąta wartością z GT (oryginalną, bez minusa, dla wykresu)
        df.loc[i, 'pred_theta'] = df.loc[i, 'gt_theta']

    df.to_csv(output_csv, index=False)
    print(f"Zapisano do: {output_csv}")

# --- for aracati2017 --- 
input_csv = r'C:\Users\janis\Projekty\Magisterka\SonarOdometry\data\aracati2017\eval_5_raw.csv'
output_csv = r'C:\Users\janis\Projekty\Magisterka\SonarOdometry\data\aracati2017\eval_5_gt_yaw.csv'
start_n = 130 
use_frame_id = False

# Nowa konfiguracja - przetestuj to!
inv_dx = True
inv_dy = True
swap_xy = True
inv_gt_yaw = True

recalculate_trajectory_with_gt_yaw(input_csv, output_csv, start_n, use_frame_id, inv_dx, inv_dy, swap_xy, inv_gt_yaw)