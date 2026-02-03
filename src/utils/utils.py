import os

def get_size(path, mode=0):
    """
    Returns total size of directory.
    :param path: Path to directory or file. 
    :param mode: Display mode (0: B, 1: kB, 2: MB, 3: GB).
    :return: Size of directory (float).
    """
    total_size = 0

    if not os.path.exists(path):
        raise FileNotFoundError(f"Ścieżka '{path}' nie istnieje.")

    # Sprawdzenie czy to pojedynczy plik czy folder
    if os.path.isfile(path):
        total_size = os.path.getsize(path)
    else:
        # Przechodzenie przez wszystkie podfoldery i pliki
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                # Pominięcie linków symbolicznych, które mogą prowadzić w nieskończoność
                if not os.path.islink(file_path):
                    total_size += os.path.getsize(file_path)

    # Logika konwersji jednostek
    # Formuła: rozmiar / (1024 ^ mode)
    divisor = 1024 ** mode
    
    return total_size / divisor