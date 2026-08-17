import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def fix_trajectory_axes(input_csv, output_csv):
    print(f"Naprawiam plik: {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Zapisujemy stare predykcje do późniejszej naprawy estymat w oknie
    old_pred_x = df['pred_x'].copy()
    old_pred_y = df['pred_y'].copy()
    old_pred_th = df['pred_theta'].copy()
    
    # 1. NAPRAWA LOKALNYCH KROKÓW (zgodnie z wykresem krok-po-kroku)
    # Wykres pokazuje, że 'step_pred_dy' (błąd boczny) zawierał ruch do przodu o odwróconym znaku
    fixed_step_dx = -df['step_pred_dy']
    # 'step_pred_dx' zawierał szum boczny
    fixed_step_dy = -df['step_pred_dx'] 
    # Kąt był lustrzanym odbiciem
    fixed_step_dtheta = -df['step_pred_dtheta']
    
    # Podmiana kroków w DataFrame
    df['step_pred_dx'] = fixed_step_dx
    df['step_pred_dy'] = fixed_step_dy
    df['step_pred_dtheta'] = fixed_step_dtheta
    
    # Przeliczenie błędów lokalnych względem GT
    df['err_step_dx'] = df['step_pred_dx'] - df['step_gt_dx']
    df['err_step_dy'] = df['step_pred_dy'] - df['step_gt_dy']
    df['err_step_dtheta'] = df['step_pred_dtheta'] - df['step_gt_dtheta']
    
    # 2. CAŁKOWANIE (RE-INTEGRACJA) GLOBALNEJ TRAJEKTORII
    pred_x = np.zeros(len(df))
    pred_y = np.zeros(len(df))
    pred_theta = np.zeros(len(df))
    
    # Startujemy z idealnego punktu startowego (tak jak robi to oryginalny kod)
    pred_x[0] = df['gt_x'].iloc[0]
    pred_y[0] = df['gt_y'].iloc[0]
    pred_theta[0] = df['gt_theta'].iloc[0]
    
    for i in range(1, len(df)):
        # Kąt
        pred_theta[i] = pred_theta[i-1] + fixed_step_dtheta.iloc[i]
        
        # Rotacja lokalnego ruchu na mapę globalną
        yaw = pred_theta[i-1]
        dx_glob = np.cos(yaw) * fixed_step_dx.iloc[i] - np.sin(yaw) * fixed_step_dy.iloc[i]
        dy_glob = np.sin(yaw) * fixed_step_dx.iloc[i] + np.cos(yaw) * fixed_step_dy.iloc[i]
        
        # Akumulacja pozycji
        pred_x[i] = pred_x[i-1] + dx_glob
        pred_y[i] = pred_y[i-1] + dy_glob
        
    df['pred_x'] = pred_x
    df['pred_y'] = pred_y
    df['pred_theta'] = np.arctan2(np.sin(pred_theta), np.cos(pred_theta))
    
    # 3. NAPRAWA ESTYMAT W OKNIE (BOXPLOTY)
    # Skoro zamieniliśmy X z Y, odchylenie standardowe też musi zostać zamienione!
    temp_std_x = df['est_std_x'].copy()
    df['est_std_x'] = df['est_std_y']
    df['est_std_y'] = temp_std_x
    
    # Rekonstrukcja estymat (string z CSV) by zachować poprawne rozsunięcie kropek
    fixed_estimates_list = []
    for i in range(len(df)):
        est_str = df['individual_estimates_str'].iloc[i]
        if pd.isna(est_str) or not str(est_str).strip():
            fixed_estimates_list.append(est_str)
            continue
            
        estimates = str(est_str).split(';')
        fixed_ests = []
        for est in estimates:
            vals = est.split(',')
            if len(vals) == 3:
                x, y, th = float(vals[0]), float(vals[1]), float(vals[2])
                
                # Różnica względem STAREJ predykcji
                old_diff_x = x - old_pred_x.iloc[i]
                old_diff_y = y - old_pred_y.iloc[i]
                old_diff_th = th - old_pred_th.iloc[i]
                
                # Aplikacja na NOWĄ predykcję z zamianą osi
                new_x = df['pred_x'].iloc[i] - old_diff_y
                new_y = df['pred_y'].iloc[i] - old_diff_x
                new_th = df['pred_theta'].iloc[i] - old_diff_th
                new_th = np.arctan2(np.sin(new_th), np.cos(new_th))
                
                fixed_ests.append(f"{new_x:.4f},{new_y:.4f},{new_th:.4f}")
                
        fixed_estimates_list.append(";".join(fixed_ests))
        
    df['individual_estimates_str'] = fixed_estimates_list
    
    # Zapis do pliku
    df.to_csv(output_csv, index=False)
    print(f"Zakończono! Zapisano naprawiony plik jako: {output_csv}")


def fix_trajectory_csv(input_csv, output_csv):
    """
    Obraca globalną estymację trajektorii o 180 stopni względem punktu startowego.
    Nie narusza lokalnych błędów krok-po-kroku (RTE/RRE) ani odchyleń standardowych,
    ponieważ są one niezmiennicze dla transformacji sztywnej (rotacji globalnej).
    """
    print(f"Wczytywanie pliku: {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # 1. Pobranie punktu startowego (środek obrotu)
    init_x = df['pred_x'].iloc[0]
    init_y = df['pred_y'].iloc[0]
    
    # 2. Odwrócenie globalnych predykcji o 180 stopni
    # Wzór na symetrię środkową: X_nowe = X_start - (X_stare - X_start) = 2*X_start - X_stare
    df['pred_x'] = 2 * init_x - df['pred_x']
    df['pred_y'] = 2 * init_y - df['pred_y']
    
    # 3. Obrót kąta o 180 stopni (pi radianów) i normalizacja do zakresu [-pi, pi]
    df['pred_theta'] = df['pred_theta'] + np.pi
    df['pred_theta'] = np.arctan2(np.sin(df['pred_theta']), np.cos(df['pred_theta']))
    
    # 4. Naprawa indywidualnych estymat w oknie (aby wykresy pudełkowe i scatter działały)
    def fix_estimates_str(est_str):
        if pd.isna(est_str) or not str(est_str).strip():
            return est_str
        
        fixed_estimates = []
        for est in str(est_str).split(';'):
            vals = est.split(',')
            if len(vals) == 3:
                x, y, th = float(vals[0]), float(vals[1]), float(vals[2])
                
                # Dokładnie ta sama transformacja dla estymat w oknie
                new_x = 2 * init_x - x
                new_y = 2 * init_y - y
                new_th = np.arctan2(np.sin(th + np.pi), np.cos(th + np.pi))
                
                fixed_estimates.append(f"{new_x:.4f},{new_y:.4f},{new_th:.4f}")
        
        return ";".join(fixed_estimates)

    df['individual_estimates_str'] = df['individual_estimates_str'].apply(fix_estimates_str)
    
    # Zapis do nowego pliku
    df.to_csv(output_csv, index=False)
    print(f"Sukces! Naprawiono trajektorię i zapisano jako: {output_csv}")

# --- Wywołanie skryptu ---
# Podmień nazwy plików na swoje właściwe
pth_org = 'C:/Users/janis/Projekty/Magisterka/SonarOdometry/data/aracati2017/eval_2.csv'
pth_fixed = 'C:/Users/janis/Projekty/Magisterka/SonarOdometry/data/aracati2017/eval_3.csv'
fix_trajectory_axes(pth_org, pth_fixed)