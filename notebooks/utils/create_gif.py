import os
from PIL import Image

folder = r"C:\Users\janis\Downloads\output_frames-20260818T074106Z-1-001\output_frames"
output_gif = r"C:\Users\janis\Projekty\Magisterka\Praca Magisterska\obrazy\seq_15_visu.gif"

# Pobierz pliki PNG
files = [f for f in os.listdir(folder) if f.endswith(".png")]
files_sorted = sorted(files)

# --- Parametry optymalizacji ---
scale_factor = 0.75  # 75% oryginalnego rozmiaru (możesz zmienić np. na 0.65)
step = 1             # Co ile klatek bierzemy (1 = wszystkie, 2 = co druga klatka)
base_duration = 200  # Czas wyświetlania klatki w ms

images = []

print(f"Przetwarzanie klatek (łącznie: {len(files_sorted[::step])})...")

for f in files_sorted[::step]:
    img_path = os.path.join(folder, f)
    with Image.open(img_path) as img:
        # Zmniejszenie rozdzielczości z zachowaniem wysokiej ostrości
        new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Konwersja palety barw dla formatu GIF
        img_quantized = img_resized.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        images.append(img_quantized)

# Zapis do zoptymalizowanego GIF-a
if images:
    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=base_duration * step,
        loop=0,
        optimize=True
    )
    
    size_mb = os.path.getsize(output_gif) / (1024 * 1024)
    print(f"GIF zapisany jako: {output_gif}")
    print(f"Rozmiar wyjściowy: {size_mb:.2f} MB")